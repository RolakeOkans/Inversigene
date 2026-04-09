"""
modules/chemistry.py

Fetches drug information from two sources:

1. PubChem — chemical structure, molecular formula
2. ChEMBL REST API — target, mechanism of action, approved indication

Key fix: mechanism endpoint requires parent_molecule_chembl_id filter,
not molecule_chembl_id, to find mechanism records across salt forms.
"""

import requests

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


# ── PubChem ───────────────────────────────────────────────────────────────────

def get_drug_info(drug_name: str) -> dict:
    """
    Fetch chemical structure and formula from PubChem.
    """
    result = {
        "cid": None,
        "name": drug_name,
        "formula": None,
        "structure_url": None,
        "found": False
    }

    if drug_name.startswith("BRD-") or drug_name.startswith("BRD_"):
        return result

    try:
        cid_resp = requests.get(
            f"{PUBCHEM_BASE}/compound/name/{requests.utils.quote(drug_name)}/cids/JSON",
            timeout=10
        )
        if cid_resp.status_code != 200:
            return result

        cids = cid_resp.json().get("IdentifierList", {}).get("CID", [])
        if not cids:
            return result

        cid = cids[0]
        result["cid"] = cid
        result["found"] = True

    except Exception:
        return result

    try:
        props_resp = requests.get(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/property/MolecularFormula,IUPACName/JSON",
            timeout=10
        )
        if props_resp.status_code == 200:
            props = props_resp.json().get("PropertyTable", {}).get("Properties", [{}])[0]
            result["formula"] = props.get("MolecularFormula")
            result["name"] = props.get("IUPACName", drug_name)
    except Exception:
        pass

    result["structure_url"] = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"
        f"?image_size=300x300"
    )

    return result


# ── ChEMBL ────────────────────────────────────────────────────────────────────

def _get_chembl_id(drug_name: str) -> str | None:
    """Find ChEMBL parent ID for a drug name."""
    try:
        # Try exact preferred name
        resp = requests.get(
            f"{CHEMBL_BASE}/molecule.json",
            params={"pref_name__iexact": drug_name, "limit": 1},
            timeout=15
        )
        if resp.status_code == 200:
            mols = resp.json().get("molecules", [])
            if mols:
                return mols[0]["molecule_chembl_id"]

        # Try synonym
        resp = requests.get(
            f"{CHEMBL_BASE}/molecule.json",
            params={"molecule_synonyms__molecule_synonym__iexact": drug_name, "limit": 1},
            timeout=15
        )
        if resp.status_code == 200:
            mols = resp.json().get("molecules", [])
            if mols:
                return mols[0]["molecule_chembl_id"]

    except Exception:
        pass
    return None


def get_chembl_info(drug_name: str) -> dict:
    """
    Fetch target, mechanism of action, and indication from ChEMBL.

    Key: uses parent_molecule_chembl_id for mechanism lookup, which
    correctly finds mechanism records stored under salt forms.
    """
    result = {
        "chembl_id": None,
        "target": None,
        "mechanism": None,
        "indication": None,
        "found": False
    }

    if drug_name.startswith("BRD-") or drug_name.startswith("BRD_"):
        result["target"] = "Uncharacterized compound"
        result["mechanism"] = "Uncharacterized compound"
        result["indication"] = "Uncharacterized compound"
        return result

    chembl_id = _get_chembl_id(drug_name)
    if not chembl_id:
        return result

    result["chembl_id"] = chembl_id
    result["found"] = True

    try:
        # ── Mechanism and target ──────────────────────────────────────────
        # Must use parent_molecule_chembl_id — mechanisms are stored under
        # the parent compound ID, not always the molecule ID returned by search
        mech_resp = requests.get(
            f"{CHEMBL_BASE}/mechanism.json",
            params={
                "parent_molecule_chembl_id": chembl_id,
                "limit": 10
            },
            timeout=15
        )

        if mech_resp.status_code == 200:
            mechs = mech_resp.json().get("mechanisms", [])
            if mechs:
                # Prefer entries where disease_efficacy=1 (directly treats disease)
                best = next(
                    (m for m in mechs if m.get("disease_efficacy") == 1),
                    mechs[0]
                )
                result["mechanism"] = best.get("mechanism_of_action")
                # Get target name from target endpoint using target_chembl_id
                target_id = best.get("target_chembl_id")
                if target_id:
                    t_resp = requests.get(
                        f"{CHEMBL_BASE}/target/{target_id}.json",
                        timeout=10
                    )
                    if t_resp.status_code == 200:
                        result["target"] = t_resp.json().get("pref_name")

        # ── Indication ────────────────────────────────────────────────────
        ind_resp = requests.get(
            f"{CHEMBL_BASE}/drug_indication.json",
            params={"molecule_chembl_id": chembl_id, "limit": 20},
            timeout=15
        )

        if ind_resp.status_code == 200:
            indications = ind_resp.json().get("drug_indications", [])
            if indications:
                sorted_inds = sorted(
                    indications,
                    key=lambda x: x.get("max_phase_for_ind") or 0,
                    reverse=True
                )
                result["indication"] = sorted_inds[0].get("efo_term")

    except Exception as e:
        print(f"  ChEMBL lookup failed for {drug_name}: {e}")

    return result


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_drugs = ["tamoxifen", "mitoxantrone", "rosuvastatin", "parthenolide", "BRD-K86574132"]

    for drug in test_drugs:
        print(f"\n{'='*40}")
        print(f"Drug: {drug}")

        info = get_drug_info(drug)
        print(f"  PubChem found: {info['found']}")
        print(f"  Formula: {info['formula']}")

        chembl = get_chembl_info(drug)
        print(f"  ChEMBL found: {chembl['found']}")
        print(f"  ChEMBL ID: {chembl['chembl_id']}")
        print(f"  Target: {chembl['target']}")
        print(f"  Mechanism: {chembl['mechanism']}")
        print(f"  Indication: {chembl['indication']}")