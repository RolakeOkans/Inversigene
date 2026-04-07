"""
modules/reversal.py

Builds the data needed to visualize the signature reversal concept.

Takes the top up and down genes from the disease signature and the
top ranked drug, and returns a structure showing:
  - Disease: genes going up/down in cancer
  - Reversal: what the drug needs to do to reverse the pattern

Since LINCS does not return per-gene drug perturbation values in our
current pipeline, we use the disease log2fc values mirrored to represent
the reversal direction — this is conceptually accurate because a perfect
reverser would invert every gene's expression change.
"""

import pandas as pd


def build_reversal_data(
    sig_df: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Build a DataFrame for the reversal visualization.

    Takes the top N up and top N down genes by fold change and
    returns their disease log2fc values alongside the expected
    reversal direction (mirrored).

    Parameters
    ----------
    sig_df : full signature DataFrame from loader.py
    top_n  : how many genes to show from each direction

    Returns
    -------
    DataFrame with columns:
        gene, disease_log2fc, reversal_log2fc, direction
    """
    up_genes = (
        sig_df[sig_df["direction"] == "up"]
        .nlargest(top_n, "log2fc")[["gene_symbol", "log2fc", "direction"]]
    )
    down_genes = (
        sig_df[sig_df["direction"] == "down"]
        .nsmallest(top_n, "log2fc")[["gene_symbol", "log2fc", "direction"]]
    )

    combined = pd.concat([up_genes, down_genes]).reset_index(drop=True)
    combined.columns = ["gene", "disease_log2fc", "direction"]

    # Reversal is the mirror of the disease pattern
    combined["reversal_log2fc"] = -combined["disease_log2fc"]

    return combined