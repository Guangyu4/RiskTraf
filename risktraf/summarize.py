from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [("mae", "MAE"), ("rmse", "RMSE"), ("mape", "MAPE")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PEMS-B risk plugin results.")
    parser.add_argument("--run_dir", default="runs")
    parser.add_argument("--out", default="runs/summary.md")
    return parser.parse_args()


def load_results(run_dir: Path) -> pd.DataFrame:
    records = []
    for path in run_dir.glob("*/*/*/seed*_h*/result.json"):
        records.append(json.loads(path.read_text()))
    if not records:
        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(f"No result files found under {run_dir}")
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    df = load_results(Path(args.run_dir))
    horizons = [3, 6, 12]
    keys = ["dataset", "model"]
    rows = []
    mae_rows = []
    for (dataset, model), group in df.groupby(keys):
        base = group[group["plugin"] == "baseline"].sort_values("mae_all").head(1)
        risk = group[group["plugin"].isin(["risk", "risk_rex", "rex"])].sort_values("mae_all").head(1)
        if base.empty or risk.empty:
            continue
        b = base.iloc[0]
        r = risk.iloc[0]
        row = {"Dataset": dataset, "Model": model}
        mae_row = {"Dataset": dataset, "Model": model}
        for h in horizons:
            for metric, label in METRICS:
                bm = float(b[f"{metric}_h{h}"])
                rm = float(r[f"{metric}_h{h}"])
                row[f"H{h} {label} Base"] = bm
                row[f"H{h} {label} Risk"] = rm
                row[f"H{h} {label} Δ%"] = (bm - rm) / bm * 100
            mae_row[f"H{h} Base"] = row[f"H{h} MAE Base"]
            mae_row[f"H{h} Risk"] = row[f"H{h} MAE Risk"]
            mae_row[f"H{h} Δ%"] = row[f"H{h} MAE Δ%"]
        for metric, label in METRICS:
            bm = float(b[f"{metric}_all"])
            rm = float(r[f"{metric}_all"])
            row[f"All {label} Base"] = bm
            row[f"All {label} Risk"] = rm
            row[f"All {label} Δ%"] = (bm - rm) / bm * 100
        mae_row["All Base"] = row["All MAE Base"]
        mae_row["All Risk"] = row["All MAE Risk"]
        mae_row["All Δ%"] = row["All MAE Δ%"]
        rows.append(row)
        mae_rows.append(mae_row)
    full_table = pd.DataFrame(rows).sort_values(["Dataset", "Model"])
    mae_table = pd.DataFrame(mae_rows).sort_values(["Dataset", "Model"])
    win_rate = (mae_table["All Δ%"] > 0).mean() * 100 if len(mae_table) else 0.0
    avg_gain = mae_table["All Δ%"].mean() if len(mae_table) else 0.0
    if len(full_table):
        dataset_summary = (
            full_table.groupby("Dataset")
            .agg(
                Models=("Model", "count"),
                MAE_Wins=("All MAE Δ%", lambda values: int((values > 0).sum())),
                RMSE_Wins=("All RMSE Δ%", lambda values: int((values > 0).sum())),
                MAPE_Wins=("All MAPE Δ%", lambda values: int((values > 0).sum())),
                H3_MAE_Avg_Gain=("H3 MAE Δ%", "mean"),
                H6_MAE_Avg_Gain=("H6 MAE Δ%", "mean"),
                H12_MAE_Avg_Gain=("H12 MAE Δ%", "mean"),
                All_MAE_Avg_Gain=("All MAE Δ%", "mean"),
                All_RMSE_Avg_Gain=("All RMSE Δ%", "mean"),
                All_MAPE_Avg_Gain=("All MAPE Δ%", "mean"),
                All_MAE_Min_Gain=("All MAE Δ%", "min"),
            )
            .reset_index()
        )
    else:
        dataset_summary = pd.DataFrame()
    md = [
        "# PEMS-B Risk Extrapolation Summary",
        "",
        f"MAE win rate: {win_rate:.1f}% ({int((mae_table['All Δ%'] > 0).sum())}/{len(mae_table)})",
        f"Average MAE gain: {avg_gain:.2f}%",
        "",
        "## Dataset Aggregate",
        "",
        dataset_summary.to_markdown(index=False, floatfmt=".4f") if len(dataset_summary) else "No paired results yet.",
        "",
        "## Per-Model MAE Horizon Table",
        "",
        mae_table.to_markdown(index=False, floatfmt=".4f") if len(mae_table) else "No paired results yet.",
        "",
        "The default CSV next to this file contains MAE, RMSE, and MAPE for all reported horizons.",
        "",
    ]
    Path(args.out).write_text("\n".join(md))
    full_table.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    mae_table.to_csv(Path(args.out).with_name(Path(args.out).stem + "_mae.csv"), index=False)
    dataset_summary.to_csv(Path(args.out).with_name(Path(args.out).stem + "_dataset.csv"), index=False)
    print(Path(args.out))


if __name__ == "__main__":
    main()
