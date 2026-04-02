"""
modules/loader.py

Loads and prepares the disease gene signature from GSE45827_geo_signature.xlsx.

The file has 6,762 differentially expressed genes (p < 0.05) with columns:
  gene_symbol, log2fc, p_value, direction (up/down)

This module returns clean up and down gene lists ready to pass into lincs.py.
"""

import pandas as pd


def load_signature(
    filepath: str,
    top_n: int = 150,
    min_log2fc: float = 1.0
) -> tuple[list[str], list[str], pd.DataFrame]:
    """
    Load the gene signature file and return up/down gene lists for LINCS query.

    Parameters
    ----------
    filepath   : path to GSE45827_geo_signature.xlsx (or .csv)
    top_n      : how many top up and top down genes to use for the LINCS query.
                 Using all 6,762 genes is too noisy — the top 100-200 by fold
                 change gives a cleaner, stronger signal.
    min_log2fc : minimum absolute log2 fold change to include (default 1.0 = 2x change)

    Returns
    -------
    up_genes   : list of top upregulated gene symbols
    down_genes : list of top downregulated gene symbols
    df         : full cleaned DataFrame (useful for display and downstream steps)
    """

    # Load file
    if filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)


    # Clean up
    df = df.dropna(subset=["gene_symbol"])           # drop the 1 row with missing gene name
    df["gene_symbol"] = df["gene_symbol"].str.strip().str.upper()
    df = df.drop_duplicates(subset=["gene_symbol"])  # keep first occurrence if any duplicates

    print(f"Loaded {len(df)} genes ({df['direction'].value_counts().get('up', 0)} up, "
          f"{df['direction'].value_counts().get('down', 0)} down)")

    # Apply fold change filter
    df_filtered = df[df["log2fc"].abs() >= min_log2fc].copy()
    print(f"After |log2fc| >= {min_log2fc} filter: {len(df_filtered)} genes remain")

    # Sort by absolute fold change and take top N from each direction
    up_df = (
        df_filtered[df_filtered["direction"] == "up"]
        .sort_values("log2fc", ascending=False)
        .head(top_n)
    )
    down_df = (
        df_filtered[df_filtered["direction"] == "down"]
        .sort_values("log2fc", ascending=True)   # most negative first
        .head(top_n)
    )

    up_genes = up_df["gene_symbol"].tolist()
    down_genes = down_df["gene_symbol"].tolist()

    print(f"Using top {len(up_genes)} up-genes and top {len(down_genes)} down-genes for LINCS query.")
    return up_genes, down_genes, df


def signature_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the loaded signature — useful for sanity checking."""
    print("\n── Signature Summary ─────────────────────")
    print(f"  Total genes:      {len(df)}")
    print(f"  Upregulated:      {(df['direction'] == 'up').sum()}")
    print(f"  Downregulated:    {(df['direction'] == 'down').sum()}")
    print(f"  Max log2fc:       {df['log2fc'].max():.2f}")
    print(f"  Min log2fc:       {df['log2fc'].min():.2f}")
    print(f"  Min p-value:      {df['p_value'].min():.50f}")
    print(f"  Max p-value:      {df['p_value'].max():.10f}")
    print("──────────────────────────────────────────\n")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    up, down, df = load_signature(
        "data/GSE45827_geo_signature.csv",
        top_n=150,
        min_log2fc=1.0
    )

    signature_summary(df)

    print("Top 10 up-genes:")
    print(up[:10])

    print("\nTop 10 down-genes:")
    print(down[:10])