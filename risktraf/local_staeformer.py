from __future__ import annotations

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, model_dim: int, num_heads: int = 8, mask: bool = False) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads
        self.fc_q = nn.Linear(model_dim, model_dim)
        self.fc_k = nn.Linear(model_dim, model_dim)
        self.fc_v = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]
        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        score = query @ key.transpose(-1, -2) / self.head_dim**0.5
        if self.mask:
            mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()
            score.masked_fill_(~mask, -torch.inf)
        out = torch.softmax(score, dim=-1) @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        return self.out_proj(out)


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        mask: bool = False,
    ) -> None:
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dim: int = -2) -> torch.Tensor:
        x = x.transpose(dim, -2)
        residual = x
        out = self.dropout1(self.attn(x, x, x))
        out = self.ln1(residual + out)
        residual = out
        out = self.dropout2(self.feed_forward(out))
        out = self.ln2(residual + out)
        return out.transpose(dim, -2)


class STAEformer(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        steps_per_day: int = 288,
        days_per_week: int = 7,
        input_dim: int = 3,
        output_dim: int = 1,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        spatial_embedding_dim: int = 0,
        adaptive_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_mixed_proj: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.days_per_week = days_per_week
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.use_mixed_proj = use_mixed_proj
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(days_per_week, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(torch.empty(num_nodes, spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            nn.init.xavier_uniform_(self.adaptive_embedding)
        if use_mixed_proj:
            self.output_proj = nn.Linear(in_steps * self.model_dim, out_steps * output_dim)
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, output_dim)
        self.attn_layers_t = nn.ModuleList(
            [SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.attn_layers_s = nn.ModuleList(
            [SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = self.input_proj(x[..., : self.input_dim])
        features = [x]
        if self.tod_embedding_dim > 0:
            features.append(self.tod_embedding((tod * self.steps_per_day).long()))
        if self.dow_embedding_dim > 0:
            features.append(self.dow_embedding(dow.long()))
        if self.spatial_embedding_dim > 0:
            features.append(self.node_emb.expand(batch_size, self.in_steps, *self.node_emb.shape))
        if self.adaptive_embedding_dim > 0:
            features.append(self.adaptive_embedding.expand(batch_size, *self.adaptive_embedding.shape))
        x = torch.cat(features, dim=-1)
        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        if self.use_mixed_proj:
            out = x.transpose(1, 2).reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
            out = self.output_proj(out).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
            return out.transpose(1, 2)
        out = self.temporal_proj(x.transpose(1, 3))
        return self.output_proj(out.transpose(1, 3))
