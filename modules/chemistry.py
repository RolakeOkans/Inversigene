"""
modules/chemistry.py

Fetches drug information from the PubChem API including:
- Chemical structure image (SVG)
- Molecular formula
- Drug target / mechanism of action
- CID (PubChem compound ID)

PubChem API is free and requires no authentication.
"""

import requests


PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def get_drug_info(drug_name: str) -> dict:
    """
    Fetch drug information from PubChem by drug name.

    Parameters
    ----------
    drug_name : name of the drug e.g. "tamoxifen", "mitoxantrone"

    Returns
    -------
    Dict with keys:
        cid           : PubChem compound ID (int or None)
        name          : canonical name from PubChem
        formula       : molecular formula e.g. "C26H29NO"
        structure_url : URL to the 2D structure image (PNG)
        description   : brief description if available
        found         : bool — False if drug not found
    """
    result = {
        "cid": None,
        "name": drug_name,
        "formula": None,
        "structure_url": None,
        "description": None,
        "found": False
    }

    # Skip BRD- coded compounds — they are not in PubChem by name
    if drug_name.startswith("BRD-") or drug_name.startswith("BRD_"):
        result["description"] = "Broad Institute compound — not available in PubChem by name."
        return result

    # Step 1: Get CID from drug name
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

    # Step 2: Get molecular formula and canonical name
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

    # Step 3: Structure image URL — use PubChem's image endpoint directly
    result["structure_url"] = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"
        f"?image_size=300x300"
    )

    # Step 4: Get description from PubChem
    try:
        desc_resp = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
            f"?heading=Pharmacology+and+Biochemistry",
            timeout=10
        )
        if desc_resp.status_code == 200:
            data = desc_resp.json()
            sections = (
                data.get("Record", {})
                .get("Section", [])
            )
            for section in sections:
                if "Pharmacology" in section.get("TOCHeading", ""):
                    subsections = section.get("Section", [])
                    for sub in subsections:
                        info = sub.get("Information", [])
                        for item in info:
                            val = item.get("Value", {}).get("StringWithMarkup", [{}])
                            if val:
                                text = val[0].get("String", "")
                                if len(text) > 50:
                                    result["description"] = text[:500]
                                    break
                        if result["description"]:
                            break
    except Exception:
        pass

    return result


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for drug in ["tamoxifen", "mitoxantrone", "rosuvastatin", "parthenolide", "BRD-K86574132"]:
        info = get_drug_info(drug)
        print(f"\n{drug}:")
        print(f"  Found: {info['found']}")
        print(f"  CID: {info['cid']}")
        print(f"  Formula: {info['formula']}")
        print(f"  Structure URL: {info['structure_url']}")
        if info['description']:
            print(f"  Description: {info['description'][:100]}...")