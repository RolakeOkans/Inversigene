"""
modules/mesh.py

Single source of truth for MeSH terms used across validation and literature modules.
"""

BREAST_CANCER_MESH = "D001943"
BREAST_CANCER_LABEL = "Breast Neoplasms"

# Subtypes — useful later if you want to filter by ER+, HER2+, etc.
SUBTYPES = {
    "ER_positive": "D047069",
    "HER2_positive": "D064726",
    "triple_negative": "D064726",
}

def get_mesh_term(subtype: str = None) -> str:
    """Return the MeSH ID for breast cancer or a specific subtype."""
    if subtype and subtype in SUBTYPES:
        return SUBTYPES[subtype]
    return BREAST_CANCER_MESH