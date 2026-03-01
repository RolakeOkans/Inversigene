# Inversigene

A drug repurposing tool for breast cancer. Upload a gene signature 
(list of differentially expressed genes) and Inversigene scores drugs 
from the LINCS L1000 database on how well they reverse that pattern, 
returning ranked repurposing candidates with clinical trial validation, 
literature summaries, and AI-generated explanations.

Built as part of the Data Science DC 12-week mentorship program, March–June 2026.

## Project Status
Under active development.

## Modules
- loader.py: ingests and prepares gene signatures
- lincs.py: queries SigCom LINCS API
- scoring.py: cosine similarity ranking
- validation.py: ClinicalTrials.gov and RepurposeDB annotation
- mesh.py: MeSH term management
- literature.py: PubMed abstract retrieval
- ai_summary.py: LLM-generated explanations
- visualizations.py: Plotly charts