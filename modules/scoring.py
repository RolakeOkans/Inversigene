"""
modules/scoring.py

Takes the raw output from lincs.py and produces a clean, ranked drug list.

The cosine similarity is computed server-side by SigCom LINCS.
This module handles:
  - Aggregating multiple signatures for the same drug (a drug may appear
    many times across different cell lines, doses, and time points)
  - Computing a consensus score per drug
  - Producing the final ranked DataFrame that the rest of the app uses
"""

import pandas as pd
import numpy as np


def rank_drugs(lincs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-experiment drug scores into one score per drug,
    then rank from most to least promising reverser.

    Parameters
    ----------
    lincs_df : DataFrame returned by lincs.query_lincs()
               Must have columns: drug_name, reversal_score, sig_id

    Returns
    -------
    DataFrame with one row per drug, sorted by consensus_score descending.
    Columns: drug_name, consensus_score, n_experiments, top_score, sig_ids
    """

    if lincs_df.empty:
        print("Warning: Empty input to rank_drugs.")
        return pd.DataFrame()

    agg = (
        lincs_df
        .groupby("drug_name")
        .agg(
            consensus_score=("logp_avg", "median"),
            top_score=("logp_avg", "max"),
            n_experiments=("logp_avg", "count"),
            uuids=("uuid", lambda x: "; ".join(x.tolist()))
        )
        .reset_index()
        .sort_values("consensus_score", ascending=False)
        .reset_index(drop=True)
    )

    agg.insert(0, "rank", range(1, len(agg) + 1))
    print(f"Ranked {len(agg)} unique drugs.")
    return agg


def get_top_n(ranked_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return the top N drugs from a ranked DataFrame."""
    return ranked_df.head(n).copy()


def score_summary(ranked_df: pd.DataFrame) -> dict:
    """
    Print a quick summary of the scoring results.
    Useful for sanity-checking in the Jupyter notebook.
    """
    if ranked_df.empty:
        return {}

    summary = {
        "total_drugs_ranked": len(ranked_df),
        "top_drug": ranked_df.iloc[0]["drug_name"],
        "top_score": ranked_df.iloc[0]["consensus_score"],
        "score_range": (
            round(ranked_df["consensus_score"].min(), 4),
            round(ranked_df["consensus_score"].max(), 4)
        ),
        "median_score": round(ranked_df["consensus_score"].median(), 4),
    }

    print("\n── Scoring Summary ──────────────────────")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("─────────────────────────────────────────\n")

    return summary


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate what lincs.py would return
    fake_lincs_output = pd.DataFrame({
        "drug_name": ["tamoxifen", "tamoxifen", "tamoxifen",
                       "bortezomib", "bortezomib",
                       "parthenolide", "lorazepam"],
        "reversal_score": [0.82, 0.78, 0.75,
                            0.61, 0.55,
                            0.49, 0.30],
        "sig_id": [f"fake_sig_{i}" for i in range(7)]
    })

    ranked = rank_drugs(fake_lincs_output)
    score_summary(ranked)
    print(ranked.to_string())