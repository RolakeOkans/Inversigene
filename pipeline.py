"""
pipeline.py

Runs the full Inversigene pipeline end to end:
  loader.py → lincs.py → scoring.py
"""

from modules.loader import load_signature, signature_summary
from modules.lincs import query_lincs
from modules.scoring import rank_drugs, score_summary

# Step 1: Load the disease signature
print("=== Step 1: Loading gene signature ===")
up_genes, down_genes, df = load_signature(
    "data/GSE45827_geo_signature.csv",
    top_n=150,
    min_log2fc=1.0
)
signature_summary(df)

# Step 2: Query LINCS
print("=== Step 2: Querying SigCom LINCS ===")
lincs_df = query_lincs(up_genes, down_genes, n_results=100)

# Step 3: Rank drugs
print("\n=== Step 3: Ranking drugs ===")
ranked = rank_drugs(lincs_df)
score_summary(ranked)

print("\nTop 20 drug candidates:")
print(ranked.head(20)[["rank", "drug_name", "consensus_score", "n_experiments"]].to_string())

from modules.validation import validate

# After Step 3, add:
print("\n=== Step 4: Validating top drugs ===")
ranked = validate(ranked, top_n=20)

print("\nFinal validated drug table:")
print(ranked.head(20)[["rank", "drug_name", "consensus_score", "trial_count", "in_repurposedb"]].to_string())