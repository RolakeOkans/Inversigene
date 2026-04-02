"""
modules/pathways.py

Queries the Enrichr API for pathway enrichment analysis on the
top up and down regulated genes from the disease signature.

Uses three libraries:
- KEGG_2021_Human        : canonical metabolic and signaling pathways
- GO_Biological_Process_2023 : gene ontology biological processes
- Reactome_2022          : curated biological pathways

API flow:
  1. POST genes to /addList → get userListId
  2. GET /enrich?userListId=...&backgroundType=... → get results
"""

import requests
import pandas as pd
import time

ENRICHR_BASE = "https://maayanlab.cloud/Enrichr"

LIBRARIES = [
    "KEGG_2021_Human",
    "GO_Biological_Process_2023",
    "Reactome_2022"
]


def run_enrichment(genes: list[str], description: str = "Inversigene query") -> dict:
    """
    Run Enrichr pathway enrichment on a list of gene symbols.

    Parameters
    ----------
    genes       : list of gene symbols e.g. ["TOP2A", "MKI67", "RRM2"]
    description : label for this gene list in Enrichr

    Returns
    -------
    Dict mapping library name → DataFrame with enrichment results
    """
    if not genes:
        return {}

    # Step 1: Submit gene list and get userListId
    gene_str = "\n".join(genes)
    payload = {
        "list": (None, gene_str),
        "description": (None, description)
    }

    try:
        resp = requests.post(f"{ENRICHR_BASE}/addList", files=payload, timeout=30)
        resp.raise_for_status()
        user_list_id = resp.json()["userListId"]
    except Exception as e:
        print(f"  Warning: Failed to submit gene list to Enrichr: {e}")
        return {}

    print(f"  Submitted {len(genes)} genes to Enrichr (userListId: {user_list_id})")

    # Step 2: Fetch enrichment results for each library
    results = {}
    for library in LIBRARIES:
        try:
            resp = requests.get(
                f"{ENRICHR_BASE}/enrich",
                params={"userListId": user_list_id, "backgroundType": library},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

            terms = data.get(library, [])
            if not terms:
                continue

            df = pd.DataFrame(terms, columns=[
                "rank", "term", "pvalue", "zscore",
                "combined_score", "genes", "adj_pvalue",
                "old_pvalue", "old_adj_pvalue"
            ])

            # Clean up
            df["genes"] = df["genes"].apply(
                lambda x: ";".join(x) if isinstance(x, list) else str(x)
            )
            df["term"] = df["term"].str.replace(r"_hsa\d+$", "", regex=True)  # strip KEGG codes
            df = df[["rank", "term", "pvalue", "adj_pvalue", "combined_score", "genes"]]
            df = df[df["adj_pvalue"] < 0.05].head(10)  # top 10 significant terms

            results[library] = df
            print(f"  {library}: {len(df)} significant terms")
            time.sleep(0.3)

        except Exception as e:
            print(f"  Warning: Enrichr query failed for {library}: {e}")

    return results


def get_pathway_enrichment(
    up_genes: list[str],
    down_genes: list[str],
    top_n: int = 100
) -> dict:
    """
    Run pathway enrichment separately for up and down genes.

    Parameters
    ----------
    up_genes   : upregulated gene list
    down_genes : downregulated gene list
    top_n      : how many top genes to use from each list

    Returns
    -------
    Dict with keys "up" and "down", each mapping library → DataFrame
    """
    print("Running pathway enrichment on upregulated genes...")
    up_results = run_enrichment(up_genes[:top_n], description="Upregulated genes")

    print("Running pathway enrichment on downregulated genes...")
    down_results = run_enrichment(down_genes[:top_n], description="Downregulated genes")

    return {"up": up_results, "down": down_results}


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_up = ["TOP2A", "MKI67", "RRM2", "AURKA", "CCNB1",
               "CDK1", "PCNA", "MCM2", "BUB1", "CXCL10"]
    test_down = ["ESR1", "GATA3", "TFF1", "PGR", "FOXA1",
                 "ADIPOQ", "FABP4", "PIGR", "RBP4", "SCARA5"]

    results = get_pathway_enrichment(test_up, test_down, top_n=10)

    for direction in ["up", "down"]:
        print(f"\n{'='*50}")
        print(f"{direction.upper()} GENE PATHWAYS:")
        for lib, df in results[direction].items():
            print(f"\n  {lib}:")
            if df.empty:
                print("    No significant terms")
            else:
                print(df[["term", "pvalue", "adj_pvalue", "combined_score"]].to_string())