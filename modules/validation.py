"""
modules/validation.py

Annotates the ranked drug list with real-world evidence from two sources:

1. ClinicalTrials.gov API — checks whether each drug has registered trials
   for breast cancer using the official MeSH term.

2. RepurposeDB — checks whether each drug is a documented repurposing case.
   RepurposeDB is downloaded as a flat file and looked up locally.

The output is the same ranked DataFrame with two new columns added:
  trial_count     : number of breast cancer trials found on ClinicalTrials.gov
  in_repurposedb  : True/False whether the drug appears in RepurposeDB
"""

import requests
import pandas as pd
import time
try:
    from modules.mesh import BREAST_CANCER_MESH
except ModuleNotFoundError:
    from mesh import BREAST_CANCER_MESH


CLINICALTRIALS_API = "https://clinicaltrials.gov/api/v2/studies"


# ── ClinicalTrials.gov ────────────────────────────────────────────────────────

def check_clinical_trials(drug_name: str) -> int:
    params = {
        "query.intr": drug_name,
        "query.cond": "breast cancer",
        "filter.overallStatus": "COMPLETED|RECRUITING|ACTIVE_NOT_RECRUITING",
        "countTotal": "true",
        "pageSize": 1,
        "format": "json"
    }

    try:
        resp = requests.get(CLINICALTRIALS_API, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("totalCount", 0)
    except Exception as e:
        print(f"  Warning: ClinicalTrials.gov query failed for {drug_name}: {e}")
        return 0


def annotate_trials(ranked_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Add a trial_count column to the ranked drug DataFrame.

    Only checks the top N drugs to avoid too many API calls.

    Parameters
    ----------
    ranked_df : DataFrame from scoring.rank_drugs()
    top_n     : how many top drugs to check (default 20)

    Returns
    -------
    Same DataFrame with trial_count column added.
    """
    df = ranked_df.copy()
    df["trial_count"] = 0

    drugs_to_check = df.head(top_n)["drug_name"].tolist()
    print(f"Checking ClinicalTrials.gov for top {len(drugs_to_check)} drugs...")

    for drug in drugs_to_check:
        count = check_clinical_trials(drug)
        df.loc[df["drug_name"] == drug, "trial_count"] = count
        print(f"  {drug}: {count} trials")
        time.sleep(0.3)  # be polite to the API

    return df


# ── RepurposeDB ───────────────────────────────────────────────────────────────

def load_repurposedb(filepath: str = "data/repodb.csv") -> set:
    """
    Load RepurposeDB and return a set of lowercase drug names.

    Download RepurposeDB from: http://repurposedb.dudleylab.org
    Save as data/repurposedb.csv

    If the file is not found, returns an empty set and prints a warning.
    """
    try:
        df = pd.read_csv(filepath)
        # RepurposeDB has a 'drug' or 'Drug' column — handle both
        col = next((c for c in df.columns if c.lower() == "drug_name"), None)
        if col is None:
            print(f"  Warning: No 'drug' column found in RepurposeDB. Columns: {df.columns.tolist()}")
            return set()
        return set(df[col].str.lower().str.strip())
    except FileNotFoundError:
        print(f"  Warning: RepurposeDB file not found at {filepath}.")
        print("  Download from http://repurposedb.dudleylab.org and save to data/repurposedb.csv")
        return set()


def annotate_repurposedb(ranked_df: pd.DataFrame, repurposedb_drugs: set) -> pd.DataFrame:
    """
    Add an in_repurposedb column to the ranked drug DataFrame.

    Parameters
    ----------
    ranked_df        : DataFrame from scoring.rank_drugs()
    repurposedb_drugs: set of lowercase drug names from load_repurposedb()

    Returns
    -------
    Same DataFrame with in_repurposedb column added.
    """
    df = ranked_df.copy()
    df["in_repurposedb"] = df["drug_name"].str.lower().str.strip().isin(repurposedb_drugs)
    n_found = df["in_repurposedb"].sum()
    print(f"  Found {n_found} drugs in RepurposeDB out of {len(df)} ranked drugs.")
    return df


# ── Main validation function ──────────────────────────────────────────────────

def validate(ranked_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Run full validation pipeline on the ranked drug list.

    Adds trial_count and in_repurposedb columns.
    Call this after scoring.rank_drugs().
    """
    print("\n=== Validation ===")

    # ClinicalTrials.gov
    df = annotate_trials(ranked_df, top_n=top_n)

    # RepurposeDB
    print("Checking RepurposeDB...")
    repurposedb_drugs = load_repurposedb()
    df = annotate_repurposedb(df, repurposedb_drugs)

    return df


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate a small ranked DataFrame to test with
    fake_ranked = pd.DataFrame({
        "rank": [1, 2, 3, 4, 5],
        "drug_name": ["tamoxifen", "raloxifene", "parthenolide", "mitoxantrone", "wortmannin"],
        "consensus_score": [50.0, 45.0, 40.0, 35.0, 30.0],
        "n_experiments": [3, 2, 1, 1, 1]
    })

    result = validate(fake_ranked, top_n=5)
    print("\nValidated drug table:")
    print(result[["rank", "drug_name", "consensus_score", "trial_count", "in_repurposedb"]].to_string())


