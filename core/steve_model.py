import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class RevGradFunc(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


revgrad = RevGradFunc.apply


class RevGradLayer(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_, p):
        alpha = self._alpha / np.power((1 + 10 * p), 0.75)
        return revgrad(input_, alpha)


def cal_cheb_polynomial(laplacian, K):
    N = laplacian.size(0)
    multi_order_laplacian = torch.zeros([K, N, N], device=laplacian.device, dtype=torch.float)
    multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)
    if K == 1:
        return multi_order_laplacian
    multi_order_laplacian[1] = laplacian
    if K == 2:
        return multi_order_laplacian
    for k in range(2, K):
        multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k - 1]) - multi_order_laplacian[k - 2]
    return multi_order_laplacian


def cal_laplacian(graph):
    I = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
    graph = graph + I
    D = torch.diag(torch.sum(graph, dim=-1) ** (-0.5))
    L = I - torch.mm(torch.mm(D, graph), D)
    return L


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x


class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super().__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)


class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out, device):
        super().__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks).to(device))
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1).to(device))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        x_in = self.align(x)
        return torch.relu(x_gc + x_in)


class STConvBlock(nn.Module):
    def __init__(self, ks, kt, n, c, p, device):
        super().__init__()
        self.tconv1 = TemporalConvLayer(kt, c[0], c[1], "GLU")
        self.sconv = SpatioConvLayer(ks, c[1], c[1], device)
        self.tconv2 = TemporalConvLayer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x, graph):
        x_t1 = self.tconv1(x)
        x_s = self.sconv(x_t1, graph)
        x_t2 = self.tconv2(x_s)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)


class MLPAttention(nn.Module):
    def __init__(self, d_models):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_models, d_models),
            nn.ReLU(),
            nn.Linear(d_models, 1),
        )
        self.softmax = nn.Softmax(-1)
        self.tau = 4

    def forward(self, Q, K, V):
        k = K.shape[1]
        n = Q.shape[1]
        b = Q.shape[0]
        Q = Q.unsqueeze(2).repeat(1, 1, k, 1)
        K = K.unsqueeze(1).repeat(1, n, 1, 1)
        input = torch.stack([Q, K], -1).reshape(b, n, k, -1)
        res = self.mlp(input).squeeze(-1)
        att = self.softmax(res / self.tau)
        out = torch.bmm(att, V)
        return out, att


def pca_whitening(data):
    data = data - data.mean(axis=0)
    cov = np.dot(data.T, data) / data.shape[0]
    _, eigenvalues, eigenvectors = np.linalg.svd(cov)
    samples = np.dot(data, eigenvectors.T)
    samples_white = samples / np.sqrt(eigenvalues + 1e-5)
    return samples_white


class CLUB(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super().__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim)
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim), nn.Tanh()
        )

    def get_mu_logvar(self, x_samples):
        return self.p_mu(x_samples), self.p_logvar(x_samples)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        positive = -(mu - y_samples) ** 2 / logvar.exp()
        negative = -(mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)


class ST_encoder(nn.Module):
    def __init__(self, num_nodes, d_input, d_output, Ks, Kt, blocks, input_window, drop_prob, device):
        super().__init__()
        self.Ks = Ks
        self.Kt = Kt
        self.blocks = blocks
        self.blocks[0][0] = d_output
        self.device = device
        self.input_conv = nn.Conv2d(d_input, d_output, 1)
        self.st_conv1 = STConvBlock(Ks, Kt, num_nodes, blocks[0], drop_prob, device)
        self.st_conv2 = STConvBlock(Ks, Kt, num_nodes, blocks[1], drop_prob, device)

    def forward(self, x, graph):
        lap_mx = cal_laplacian(graph)
        Lk = cal_cheb_polynomial(lap_mx, self.Ks)
        x = self.input_conv(x)
        x_st1 = self.st_conv1(x, Lk)
        x_st2 = self.st_conv2(x_st1, Lk)
        return x_st2

    def variant_encode(self, x, graph):
        x = self.input_conv(x)
        x_st1 = self.st_conv1(x, graph)
        x_st2 = self.st_conv2(x_st1, graph)
        return x_st2


class STEVE(nn.Module):
    def __init__(self, num_nodes, input_dim=3, embed_size=64, input_length=12, output_dim=1,
                 dropout=0.1, device="cuda", mi_w=2, kw=0.5, bank_gamma=0.9):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_size = embed_size
        self.mi_w = mi_w
        self.bank_gamma = bank_gamma
        self.device = device

        T_dim = input_length - 4 * (3 - 1)
        self.T_dim = T_dim
        self.K = int(embed_size * kw)
        self.spatial_label = torch.tensor(list(range(num_nodes)), device=device)

        blocks = [[embed_size, embed_size // 2, embed_size], [embed_size, embed_size // 2, embed_size]]
        self.st_encoder4variant = ST_encoder(num_nodes, input_dim, embed_size, 3, 3, 
                                              [b.copy() for b in blocks], input_length, dropout, device)
        self.st_encoder4invariant = ST_encoder(num_nodes, input_dim, embed_size, 3, 3,
                                                [b.copy() for b in blocks], input_length, dropout, device)

        self.node_embeddings_1 = nn.Parameter(torch.randn(3, num_nodes, embed_size))
        self.node_embeddings_2 = nn.Parameter(torch.randn(3, embed_size, num_nodes))

        self.tcl4c = nn.Conv2d(T_dim, 1, 1, bias=True)
        self.tcl4h = nn.Conv2d(T_dim, 1, 1, bias=True)
        self.variant_predict_conv_2 = nn.Conv2d(embed_size, output_dim, 1)
        self.invariant_predict_conv_2 = nn.Conv2d(embed_size, output_dim, 1)

        self.revgrad = RevGradLayer()
        self.mse_loss = nn.MSELoss()

        self.mi_net = CLUB(embed_size, embed_size, embed_size * mi_w)
        self.optimizer_mi_net = torch.optim.Adam(self.mi_net.parameters(), lr=0.1)

        bank_temp = np.random.randn(self.K, embed_size)
        bank_temp = pca_whitening(bank_temp)
        self.Bank = nn.Parameter(torch.tensor(bank_temp, dtype=torch.float), requires_grad=False)
        self.mlp4bank = nn.Linear(T_dim * num_nodes, self.K)
        self.att4bank = MLPAttention(embed_size)
        self.W_weight = nn.Parameter(torch.randn(embed_size, output_dim))

        self.variant_end_temproal = nn.Sequential(nn.Linear(embed_size, embed_size * 2), nn.ReLU(), nn.Linear(embed_size * 2, 48))
        self.variant_end_spacial = nn.Sequential(nn.Linear(embed_size, embed_size * 2), nn.ReLU(), nn.Linear(embed_size * 2, num_nodes))
        self.variant_end_congest = nn.Sequential(nn.Linear(embed_size, embed_size // 2), nn.ReLU(), nn.Linear(embed_size // 2, output_dim))
        self.invariant_end_temporal = nn.Sequential(nn.Linear(embed_size, embed_size * 2), nn.ReLU(), nn.Linear(embed_size * 2, 48))
        self.invariant_end_spatial = nn.Sequential(nn.Linear(embed_size, embed_size * 2), nn.ReLU(), nn.Linear(embed_size * 2, num_nodes))
        self.invariant_end_congest = nn.Sequential(nn.Linear(embed_size, embed_size // 2), nn.ReLU(), nn.Linear(embed_size // 2, output_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x, adj):
        x = x.permute(0, 3, 1, 2)
        invariant_output = self.st_encoder4invariant(x, adj)
        H_tensor = invariant_output.permute(0, 2, 3, 1)

        adaptive_adj = F.softmax(F.relu(torch.bmm(self.node_embeddings_1, self.node_embeddings_2)), dim=1)
        variant_output = self.st_encoder4variant.variant_encode(x, adaptive_adj)
        Z_tensor = variant_output.permute(0, 2, 3, 1)

        return H_tensor, Z_tensor

    def confounder_ext(self, Z_tensor):
        b, t, n, c = Z_tensor.shape
        Z_tilda = Z_tensor.reshape(b, n * t, c).permute(0, 2, 1)
        B_tilda = self.mlp4bank(Z_tilda).permute(0, 2, 1)

        B_new = []
        for i in range(b):
            _B_new = self.bank_gamma * self.Bank + (1 - self.bank_gamma) * B_tilda[i]
            self.Bank.data.copy_(_B_new.detach())
            B_new.append(_B_new)
        B_new = torch.stack(B_new)

        Q = Z_tensor.mean(1)
        C_tensor, att = self.att4bank(Q, B_new, B_new)
        return C_tensor, att

    def predict(self, Z_tensor, C_tensor, H):
        C_tensor = C_tensor.unsqueeze(1)
        out = C_tensor + self.tcl4c(Z_tensor)
        out = out.permute(0, 3, 2, 1)
        Y_c = self.variant_predict_conv_2(out).permute(0, 3, 2, 1)

        H = H.permute(0, 3, 2, 1)
        Y_h = self.invariant_predict_conv_2(H).permute(0, 3, 2, 1)

        C_weight = torch.relu(torch.matmul(C_tensor, self.W_weight))
        Y = C_weight * Y_c + Y_h
        return Y

    def predict_test(self, Z_tensor, H_tensor):
        H = self.tcl4h(H_tensor)
        C_tensor, att = self.confounder_ext(Z_tensor)
        Y = self.predict(Z_tensor, C_tensor, H)
        return Y

