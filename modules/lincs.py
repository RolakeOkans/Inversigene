"""
modules/lincs.py

Queries the SigCom LINCS API with a disease gene signature and returns
drug perturbation signatures ranked by how well they reverse the pattern.

The API does similarity scoring server-side — we send up/down gene lists,
convert them to internal entity IDs, run the search, and parse results.
"""

import requests
import pandas as pd


METADATA_API = "https://maayanlab.cloud/sigcom-lincs/metadata-api"
DATA_API = "https://maayanlab.cloud/sigcom-lincs/data-api/api/v1"

# Library ID for LINCS L1000 Chemical Perturbations (drugs only)
CHEMICAL_LIBRARY = "l1000_cp"


def query_lincs(up_genes: list[str], down_genes: list[str], n_results: int = 50) -> pd.DataFrame:
    """
    Send a gene signature to the SigCom LINCS enrichment API and get back
    ranked drug signatures.

    Parameters
    ----------
    up_genes : list of gene symbols that are upregulated in the disease
    down_genes : list of gene symbols that are downregulated in the disease
    n_results : how many top reverser results to return (default 50)

    Returns
    -------
    DataFrame with columns: uuid, logp_avg, z_sum, p_down, rank,
                            drug_name, cell_line, pert_dose, pert_time
    """
    print(f"Querying SigCom LINCS with {len(up_genes)} up-genes and {len(down_genes)} down-genes...")

    # Step 1: Convert gene symbols to internal entity IDs
    entities = _convert_genes(up_genes, down_genes)
    if not entities.get("up_entities") or not entities.get("down_entities"):
        raise ValueError("Could not convert gene symbols to entity IDs. Check your gene names.")

    print(f"  Converted genes to entity IDs.")

    # Step 2: Run similarity search against chemical perturbations library
    payload = {
        **entities,
        "database": CHEMICAL_LIBRARY,
        "limit": n_results * 5  # fetch extra so we have enough reversers after filtering
    }

    try:
        resp = requests.post(
            f"{DATA_API}/enrich/ranktwosided",
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to query SigCom LINCS: {e}")

    raw = resp.json()

    # Step 3: Parse results
    df = _parse_results(raw, n_results=n_results)

    # Step 4: Resolve UUIDs to drug names
    if not df.empty:
        df = resolve_drug_names(df)

    print(f"  Done. Retrieved {len(df)} drug signatures.")
    return df


def _convert_genes(up_genes: list[str], down_genes: list[str]) -> dict:
    """
    Convert gene symbols to SigCom LINCS internal entity UUIDs.
    The enrichment API requires these IDs, not raw gene names.
    """
    all_genes = list(set(up_genes + down_genes))
    payload = {
        "filter": {
            "where": {
                "meta.symbol": {"inq": all_genes}
            }
        }
    }

    resp = requests.post(f"{METADATA_API}/entities/find", json=payload, timeout=30)
    resp.raise_for_status()
    results = resp.json()

    up_set = set(g.upper() for g in up_genes)
    down_set = set(g.upper() for g in down_genes)

    up_entities = []
    down_entities = []

    for entity in results:
        symbol = entity.get("meta", {}).get("symbol", "").upper()
        uid = entity.get("id")
        if symbol in up_set:
            up_entities.append(uid)
        elif symbol in down_set:
            down_entities.append(uid)

    print(f"  Matched {len(up_entities)} up-genes and {len(down_entities)} down-genes in LINCS.")
    return {"up_entities": up_entities, "down_entities": down_entities}


def _parse_results(raw: dict, n_results: int) -> pd.DataFrame:
    """
    Parse the raw API response into a clean DataFrame.

    The response shape is:
      "reversers": <integer count>   <- just a count, NOT a list
      "mimickers": <integer count>   <- same
      "results": [ list of all items, each with a "type" field ]

    Each item has: uuid, type (mimickers/reversers), logp-avg, z-sum, etc.
    We filter to type == "reversers" and sort by logp-avg descending.
    Higher logp-avg = stronger reversal = better drug candidate.
    """
    all_results = raw.get("results", [])

    rows = []
    for item in all_results:
        if item.get("type") != "reversers":
            continue

        rows.append({
            "uuid": item.get("uuid", ""),
            "logp_avg": item.get("logp-avg", 0.0),
            "z_sum": item.get("z-sum", 0.0),
            "p_down": item.get("p-down", 1.0),
            "rank": item.get("rank", 9999),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("  Warning: No reverser results found.")
        types = set(i.get("type") for i in all_results)
        print(f"  Total results returned: {len(all_results)}, types seen: {types}")
        return df

    df = df.sort_values("logp_avg", ascending=False).head(n_results).reset_index(drop=True)
    print(f"  Found {len(df)} reversers out of {len(all_results)} total results.")
    return df


def resolve_drug_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Look up human-readable drug names for the UUIDs returned by the search.

    The enrichment API returns UUIDs. This function calls the metadata API
    to resolve each UUID to a drug name, cell line, dose, and time point.

    Parameters
    ----------
    df : DataFrame with a "uuid" column (output of _parse_results)

    Returns
    -------
    Same DataFrame with added columns: drug_name, cell_line, pert_dose, pert_time
    """
    uuids = df["uuid"].tolist()

    payload = {
        "filter": {
            "where": {
                "id": {"inq": uuids}
            }
        }
    }

    try:
        resp = requests.post(f"{METADATA_API}/signatures/find", json=payload, timeout=30)
        resp.raise_for_status()
        signatures = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  Warning: Could not resolve drug names: {e}")
        df["drug_name"] = df["uuid"]
        return df

    # Build a lookup dict: uuid -> metadata
    meta_lookup = {}
    for sig in signatures:
        uid = sig.get("id", "")
        meta = sig.get("meta", {})
        meta_lookup[uid] = {
            "drug_name": (
                meta.get("pert_name")
                or meta.get("drug_name")
                or meta.get("name")
                or uid  # fallback to UUID if no name found
            ),
            "cell_line": meta.get("cell_line") or meta.get("cell_iname", ""),
            "pert_dose": meta.get("pert_dose", ""),
            "pert_time": meta.get("pert_time", ""),
        }

    # Merge into the DataFrame
    df["drug_name"] = df["uuid"].map(lambda u: meta_lookup.get(u, {}).get("drug_name", u))
    df["cell_line"] = df["uuid"].map(lambda u: meta_lookup.get(u, {}).get("cell_line", ""))
    df["pert_dose"] = df["uuid"].map(lambda u: meta_lookup.get(u, {}).get("pert_dose", ""))
    df["pert_time"] = df["uuid"].map(lambda u: meta_lookup.get(u, {}).get("pert_time", ""))

    return df


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # A few well-known ER+ breast cancer genes to test with
    # ESR1, GATA3, TFF1 are upregulated in ER+ breast cancer
    # MKI67, TOP2A, CDK1 are proliferation markers (often downregulated by treatment)
    test_up = ["ESR1", "GATA3", "TFF1", "PGR", "FOXA1"]
    test_down = ["MKI67", "TOP2A", "AURKA", "CCNB1", "CDK1"]

    df = query_lincs(test_up, test_down, n_results=20)

    print("\nTop drug candidates (reversers):")
    print(df[["drug_name", "logp_avg", "z_sum", "cell_line", "pert_dose"]].to_string())