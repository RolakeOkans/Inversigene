"""
modules/literature.py

Queries PubMed for abstracts about the top up and down genes
from the disease signature, filtered by the breast cancer MeSH term.

Uses Biopython's Entrez API — no raw HTTP requests needed.
"""

from Bio import Entrez
import pandas as pd
import time

try:
    from modules.mesh import BREAST_CANCER_MESH, BREAST_CANCER_LABEL
except ModuleNotFoundError:
    from mesh import BREAST_CANCER_MESH, BREAST_CANCER_LABEL

# Required by NCBI — set this to your email
Entrez.email = "mokanlaw@gmu.edu"


def fetch_gene_abstracts(
    gene_symbol: str,
    max_results: int = 5
) -> list[dict]:
    """
    Query PubMed for abstracts about a gene in the context of breast cancer.

    Parameters
    ----------
    gene_symbol : gene name e.g. "TOP2A"
    max_results : how many abstracts to return (default 5)

    Returns
    -------
    List of dicts with keys: pmid, title, abstract, year
    """
    query = f"{gene_symbol}[Gene Name] AND breast neoplasms[MeSH Terms]"

    try:
        # Search for PMIDs
        search_handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance"
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()

        pmids = search_results.get("IdList", [])
        if not pmids:
            return []

        # Fetch full records for those PMIDs
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=",".join(pmids),
            rettype="abstract",
            retmode="xml"
        )
        records = Entrez.read(fetch_handle)
        fetch_handle.close()

        abstracts = []
        for record in records.get("PubmedArticle", []):
            try:
                article = record["MedlineCitation"]["Article"]
                pmid = str(record["MedlineCitation"]["PMID"])
                title = str(article.get("ArticleTitle", ""))
                year = str(
                    article.get("Journal", {})
                    .get("JournalIssue", {})
                    .get("PubDate", {})
                    .get("Year", "")
                )

                # Abstract text can be structured or plain
                abstract_raw = article.get("Abstract", {}).get("AbstractText", "")
                if isinstance(abstract_raw, list):
                    abstract = " ".join(str(a) for a in abstract_raw)
                else:
                    abstract = str(abstract_raw)

                abstracts.append({
                    "gene": gene_symbol,
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "year": year
                })
            except Exception:
                continue

        return abstracts

    except Exception as e:
        print(f"  Warning: PubMed query failed for {gene_symbol}: {e}")
        return []


def fetch_literature(
    up_genes: list[str],
    down_genes: list[str],
    top_n: int = 5,
    abstracts_per_gene: int = 3
) -> dict:
    """
    Fetch PubMed abstracts for the top up and down genes.

    Parameters
    ----------
    up_genes          : full list of upregulated genes (from loader.py)
    down_genes        : full list of downregulated genes (from loader.py)
    top_n             : how many top genes from each direction to query
    abstracts_per_gene: how many abstracts to fetch per gene

    Returns
    -------
    Dict with keys "up" and "down", each containing a list of abstract dicts
    """
    top_up = up_genes[:top_n]
    top_down = down_genes[:top_n]

    results = {"up": [], "down": []}

    print(f"Querying PubMed for top {top_n} up-genes...")
    for gene in top_up:
        abstracts = fetch_gene_abstracts(gene, max_results=abstracts_per_gene)
        results["up"].extend(abstracts)
        print(f"  {gene}: {len(abstracts)} abstracts found")
        time.sleep(0.4)  # NCBI rate limit: max 3 requests/sec without API key

    print(f"Querying PubMed for top {top_n} down-genes...")
    for gene in top_down:
        abstracts = fetch_gene_abstracts(gene, max_results=abstracts_per_gene)
        results["down"].extend(abstracts)
        print(f"  {gene}: {len(abstracts)} abstracts found")
        time.sleep(0.4)

    total = len(results["up"]) + len(results["down"])
    print(f"  Done. Fetched {total} abstracts total.")
    return results


def literature_to_df(literature: dict) -> pd.DataFrame:
    """Convert the literature dict to a flat DataFrame for easy display."""
    rows = []
    for direction, abstracts in literature.items():
        for a in abstracts:
            rows.append({**a, "direction": direction})
    return pd.DataFrame(rows)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_up = ["TOP2A", "MKI67", "RRM2"]
    test_down = ["PIGR", "ADIPOQ", "FABP4"]

    lit = fetch_literature(test_up, test_down, top_n=3, abstracts_per_gene=2)

    print("\n── Sample abstracts ──────────────────────")
    for direction in ["up", "down"]:
        print(f"\n{direction.upper()} genes:")
        for entry in lit[direction]:
            print(f"  [{entry['gene']}] {entry['year']} — {entry['title'][:80]}...")