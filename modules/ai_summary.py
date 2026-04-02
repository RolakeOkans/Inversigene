"""
modules/ai_summary.py

Generates AI-powered explanations using the Anthropic API.

Two functions:
1. summarize_drugs()  — explains why top drugs ranked highly, referencing the actual genes
2. summarize_genes()  — synthesizes PubMed literature for top genes
"""

import os
import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"


def summarize_drugs(
    ranked_df: pd.DataFrame,
    top_n: int = 5,
    up_genes: list = None,
    down_genes: list = None
) -> str:
    """
    Generate plain English explanations for the top ranked drugs,
    referencing the actual genes from the disease signature.

    Parameters
    ----------
    ranked_df  : DataFrame from scoring.rank_drugs() with validation columns
    top_n      : how many top drugs to explain (default 5)
    up_genes   : list of upregulated gene symbols from the disease signature
    down_genes : list of downregulated gene symbols from the disease signature

    Returns
    -------
    String with AI-generated explanation for each top drug
    """
    top = ranked_df.head(top_n)

    # Build a readable summary of the top drugs
    drug_info = []
    for _, row in top.iterrows():
        info = f"- {row['drug_name']} (score: {row['consensus_score']:.2f}"
        if "trial_count" in row:
            info += f", breast cancer trials: {row['trial_count']}"
        if "in_repurposedb" in row:
            info += f", in repoDB: {row['in_repurposedb']}"
        info += ")"
        drug_info.append(info)

    drug_list = "\n".join(drug_info)

    # Build gene context if available
    gene_context = ""
    if up_genes and down_genes:
        gene_context = f"""
The disease signature used to find these drugs consisted of:
- Top upregulated genes (overactive in cancer): {", ".join(up_genes[:20])}
- Top downregulated genes (suppressed in cancer): {", ".join(down_genes[:20])}

These genes were matched against drug perturbation profiles in LINCS L1000.
A drug scores highly when its effect REVERSES this pattern —
suppressing the upregulated genes and activating the downregulated ones.
"""

    prompt = f"""You are a computational biologist helping interpret drug repurposing results.

The following drugs were ranked as top candidates for breast cancer treatment based on how strongly 
their gene expression signatures reverse a breast cancer gene signature from the LINCS L1000 database.
A higher score means the drug more strongly reverses the cancer expression pattern.
{gene_context}
Top ranked drugs:
{drug_list}

For each drug, provide a short paragraph (3-5 sentences) explaining:
1. What the drug is and its known mechanism of action
2. Which specific genes from the signature above it most likely targets or reverses, and why that makes biological sense
3. Any caveats or limitations to interpret this result

Be specific — reference actual genes from the list above where relevant.
If you don't know much about a drug (especially BRD- coded compounds), say so honestly."""

    message = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


def summarize_genes(literature: dict, top_n: int = 5) -> str:
    """
    Generate a synthesis of PubMed literature for the top genes.

    Parameters
    ----------
    literature : dict from literature.fetch_literature() with "up" and "down" keys
    top_n      : how many genes to synthesize per direction

    Returns
    -------
    String with AI-generated gene literature synthesis
    """
    sections = []

    for direction in ["up", "down"]:
        abstracts = literature.get(direction, [])
        if not abstracts:
            continue

        label = "upregulated" if direction == "up" else "downregulated"
        sections.append(f"\n{label.upper()} GENES:")

        genes_seen = {}
        for entry in abstracts:
            gene = entry["gene"]
            if gene not in genes_seen:
                genes_seen[gene] = []
            genes_seen[gene].append(f"  [{entry['year']}] {entry['title']}")

        for gene, titles in list(genes_seen.items())[:top_n]:
            sections.append(f"\n{gene}:")
            sections.extend(titles)

    abstract_summary = "\n".join(sections)

    prompt = f"""You are a computational biologist helping interpret gene expression results in breast cancer.

The following genes were identified as significantly differentially expressed in a breast cancer dataset.
Below are recent PubMed paper titles about each gene in the context of breast cancer.

{abstract_summary}

For each gene, write 2-3 sentences summarizing:
1. What role this gene plays in breast cancer based on the literature
2. Whether it is associated with a specific subtype (ER+, HER2+, triple negative)
3. Why its up or downregulation in this dataset makes biological sense

Be concise and scientific. Group upregulated and downregulated genes separately."""

    message = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fake_ranked = pd.DataFrame({
        "rank": [1, 2, 3],
        "drug_name": ["parthenolide", "mitoxantrone", "rosuvastatin"],
        "consensus_score": [744.4, 32.8, 32.7],
        "trial_count": [0, 6, 14],
        "in_repurposedb": [False, True, True]
    })

    test_up = ["COL11A1", "TOP2A", "RRM2", "CXCL10", "MKI67"]
    test_down = ["PIGR", "ADIPOQ", "FABP4", "SCARA5", "RBP4"]

    print("=== Drug Summary (with gene context) ===")
    drug_summary = summarize_drugs(fake_ranked, top_n=3, up_genes=test_up, down_genes=test_down)
    print(drug_summary)

    fake_literature = {
        "up": [
            {"gene": "TOP2A", "year": "2022", "title": "TOP2A expression in breast cancer subtypes"},
            {"gene": "RRM2", "year": "2023", "title": "RRM2 promotes breast cancer proliferation"},
        ],
        "down": [
            {"gene": "ADIPOQ", "year": "2021", "title": "Adiponectin and breast cancer risk"},
        ]
    }

    print("\n=== Gene Summary ===")
    gene_summary = summarize_genes(fake_literature, top_n=3)
    print(gene_summary)