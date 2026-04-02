"""
app.py

Inversigene — Breast Cancer Drug Repurposing Tool
Built with Streamlit. Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

from modules.loader import load_signature
from modules.lincs import query_lincs
from modules.scoring import rank_drugs
from modules.validation import validate
from modules.literature import fetch_literature, literature_to_df
from modules.ai_summary import summarize_drugs, summarize_genes

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inversigene",
    page_icon="🧬",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧬 Inversigene")
st.markdown(
    "Upload a breast cancer gene signature and get ranked drug repurposing candidates "
    "matched against the LINCS L1000 database, annotated with clinical trials and literature."
)
st.divider()

# ── Sidebar settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_n_genes = st.slider("Top genes to use for LINCS query", 50, 200, 150, step=25)
    top_n_drugs = st.slider("Top drugs to display", 10, 50, 20, step=5)
    top_n_lit = st.slider("Genes to query in PubMed", 3, 10, 5, step=1)
    abstracts_per_gene = st.slider("Abstracts per gene", 1, 5, 3, step=1)
    min_log2fc = st.number_input("Min |log2FC| filter", min_value=0.0, max_value=3.0, value=1.0, step=0.5)
    st.divider()
    st.markdown("**Expected CSV format:**")
    st.code("gene_symbol,log2fc\nBRCA1,2.3\nTP53,-1.8", language="csv")
    st.markdown("A `direction` column is optional — the app infers it from the sign of log2fc.")

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your gene signature CSV",
    type=["csv"],
    help="CSV with at least gene_symbol and log2fc columns"
)

use_demo = st.checkbox("Use built-in GSE45827 breast cancer dataset (demo)")

st.divider()

# ── Run button ────────────────────────────────────────────────────────────────
run = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

if run:
    st.session_state["results"] = None
    for key in list(st.session_state.keys()):
        if key.startswith("gene_summary_") or key.startswith("drug_summary"):
            del st.session_state[key]

    if not uploaded_file and not use_demo:
        st.error("Please upload a gene signature CSV or check 'Use built-in dataset'.")
        st.stop()

    progress = st.progress(0, text="Loading gene signature...")

    # ── Step 1: Load signature ────────────────────────────────────────────────
    try:
        if use_demo:
            filepath = "data/GSE45827_geo_signature.csv"
        else:
            content = uploaded_file.read().decode("utf-8")
            df_check = pd.read_csv(io.StringIO(content))

            if "gene_symbol" not in df_check.columns:
                st.error("CSV must have a 'gene_symbol' column.")
                st.stop()
            if "log2fc" not in df_check.columns:
                st.error("CSV must have a 'log2fc' column.")
                st.stop()

            if "direction" not in df_check.columns:
                df_check["direction"] = df_check["log2fc"].apply(
                    lambda x: "up" if x > 0 else "down"
                )

            filepath = "data/uploaded_signature.csv"
            df_check.to_csv(filepath, index=False)

        up_genes, down_genes, sig_df = load_signature(
            filepath,
            top_n=top_n_genes,
            min_log2fc=min_log2fc
        )
    except Exception as e:
        st.error(f"Failed to load gene signature: {e}")
        st.stop()

    progress.progress(15, text="Querying SigCom LINCS for drug signatures...")

    # ── Step 2: Query LINCS ───────────────────────────────────────────────────
    try:
        lincs_df = query_lincs(up_genes, down_genes, n_results=100)
    except Exception as e:
        st.error(f"LINCS query failed: {e}")
        st.stop()

    progress.progress(40, text="Scoring and ranking drugs...")

    # ── Step 3: Score and rank ────────────────────────────────────────────────
    try:
        ranked = rank_drugs(lincs_df)
    except Exception as e:
        st.error(f"Scoring failed: {e}")
        st.stop()

    progress.progress(55, text="Checking clinical trials and repurposing databases...")

    # ── Step 4: Validate ──────────────────────────────────────────────────────
    try:
        ranked = validate(ranked, top_n=top_n_drugs)
    except Exception as e:
        st.warning(f"Validation step had an issue: {e}")

    progress.progress(75, text="Searching PubMed for key gene literature...")

    # ── Step 5: Literature ────────────────────────────────────────────────────
    try:
        literature = fetch_literature(
            up_genes, down_genes,
            top_n=top_n_lit,
            abstracts_per_gene=abstracts_per_gene
        )
        lit_df = literature_to_df(literature)
    except Exception as e:
        st.warning(f"Literature step had an issue: {e}")
        literature = {"up": [], "down": []}
        lit_df = pd.DataFrame()

    progress.progress(100, text="✅ Analysis complete!")

    st.session_state["results"] = {
        "ranked": ranked,
        "lit_df": lit_df,
        "sig_df": sig_df,
        "literature": literature,
        "up_genes": up_genes,
        "down_genes": down_genes
    }

# ── Display results ───────────────────────────────────────────────────────────
if "results" in st.session_state and st.session_state["results"] is not None:
    ranked = st.session_state["results"]["ranked"]
    lit_df = st.session_state["results"]["lit_df"]
    sig_df = st.session_state["results"]["sig_df"]
    literature = st.session_state["results"]["literature"]
    up_genes = st.session_state["results"]["up_genes"]
    down_genes = st.session_state["results"]["down_genes"]

    # ═══════════════════════════════════════════════════════════════════════════
    # HEADLINE SUMMARY CARDS
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("### 📊 Analysis Summary")
    c1, c2, c3, c4 = st.columns(4)

    top_drug = ranked.iloc[0]["drug_name"] if not ranked.empty else "N/A"
    drugs_with_trials = int((ranked["trial_count"] > 0).sum()) if "trial_count" in ranked.columns else 0
    top_gene_up = sig_df[sig_df["direction"] == "up"].iloc[0]["gene_symbol"] if not sig_df.empty else "N/A"
    top_gene_down = sig_df[sig_df["direction"] == "down"].iloc[0]["gene_symbol"] if not sig_df.empty else "N/A"

    c1.metric("🥇 Top Drug Candidate", top_drug)
    c2.metric("🏥 Drugs with Clinical Trials", drugs_with_trials)
    c3.metric("⬆️ Top Upregulated Gene", top_gene_up)
    c4.metric("⬇️ Top Downregulated Gene", top_gene_down)

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1: Signature preview + Volcano plot
    # ═══════════════════════════════════════════════════════════════════════════
    st.subheader("🔬 Your Gene Signature")

    tab_preview, tab_volcano = st.tabs(["Signature preview", "Volcano plot"])

    with tab_preview:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total genes", len(sig_df))
        col2.metric("Upregulated", (sig_df["direction"] == "up").sum())
        col3.metric("Downregulated", (sig_df["direction"] == "down").sum())
        st.dataframe(sig_df.head(20), use_container_width=True, hide_index=True)

    with tab_volcano:
        st.markdown("Red = upregulated, Blue = downregulated. Dashed lines show significance thresholds.")

        vol_df = sig_df.copy()
        vol_df["log2fc"] = pd.to_numeric(vol_df["log2fc"], errors="coerce")
        vol_df["p_value"] = pd.to_numeric(vol_df["p_value"], errors="coerce")
        vol_df = vol_df.dropna(subset=["log2fc", "p_value"])
        vol_df["-log10p"] = -np.log10(vol_df["p_value"].clip(lower=1e-300))
        vol_df = vol_df[np.isfinite(vol_df["-log10p"])]

        vol_df["color"] = "neutral"
        vol_df.loc[(vol_df["log2fc"] > 1) & (vol_df["p_value"] < 0.05), "color"] = "up"
        vol_df.loc[(vol_df["log2fc"] < -1) & (vol_df["p_value"] < 0.05), "color"] = "down"

        colors = {"up": "#e74c3c", "down": "#3498db", "neutral": "#bdc3c7"}
        fig_vol = go.Figure()

        for group, color in colors.items():
            mask = vol_df["color"] == group
            subset = vol_df[mask]
            fig_vol.add_trace(go.Scatter(
                x=subset["log2fc"],
                y=subset["-log10p"],
                mode="markers",
                marker=dict(size=4, color=color, opacity=0.7),
                text=subset["gene_symbol"],
                hovertemplate="<b>%{text}</b><br>log2FC: %{x:.2f}<br>-log10p: %{y:.2f}<extra></extra>",
                name=group
            ))

        fig_vol.add_vline(x=1, line_dash="dash", line_color="gray", line_width=1)
        fig_vol.add_vline(x=-1, line_dash="dash", line_color="gray", line_width=1)
        fig_vol.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray", line_width=1)
        fig_vol.update_layout(
            height=550,
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            xaxis_title="log2 Fold Change",
            yaxis_title="-log10(p-value)",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=True, zerolinecolor="gray"),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0")
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2: Gene literature with inline AI synthesis
    # ═══════════════════════════════════════════════════════════════════════════
    st.subheader("📚 Gene Literature & AI Synthesis")
    st.markdown(
        "Expand a gene to read published abstracts and generate an AI summary inline. "
        "Upregulated genes drive the cancer pattern — downregulated genes are suppressed by it."
    )

    tab_up, tab_down = st.tabs(["⬆️ Upregulated genes", "⬇️ Downregulated genes"])

    for tab, direction in [(tab_up, "up"), (tab_down, "down")]:
        with tab:
            subset = lit_df[lit_df["direction"] == direction] if not lit_df.empty else pd.DataFrame()
            if subset.empty:
                st.info("No abstracts found.")
            else:
                for gene in subset["gene"].unique():
                    gene_abstracts = subset[subset["gene"] == gene]
                    with st.expander(f"**{gene}** — {len(gene_abstracts)} abstracts"):

                        ai_key = f"gene_summary_{gene}"
                        if st.button(f"🤖 Synthesize {gene} with AI", key=f"btn_{gene}"):
                            gene_lit = {
                                direction: [
                                    row.to_dict()
                                    for _, row in gene_abstracts.iterrows()
                                ]
                            }
                            with st.spinner(f"Generating summary for {gene}..."):
                                try:
                                    summary = summarize_genes(gene_lit, top_n=1)
                                    st.session_state[ai_key] = summary
                                except Exception as e:
                                    st.error(f"AI summary failed: {e}")

                        if ai_key in st.session_state:
                            st.markdown("**🤖 AI Synthesis:**")
                            st.markdown(st.session_state[ai_key])
                            st.divider()

                        st.markdown("**📄 Abstracts:**")
                        for _, row in gene_abstracts.iterrows():
                            st.markdown(f"**{row['year']} — {row['title']}**")
                            st.write(row["abstract"] if row["abstract"] else "Abstract not available.")
                            st.markdown(f"PMID: {row['pmid']}")
                            st.markdown("---")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: Ranked drug table + bar chart + inline AI explanations
    # ═══════════════════════════════════════════════════════════════════════════
    st.subheader("💊 Drug Repurposing Candidates")
    st.markdown(
        "Drugs ranked by how strongly they reverse your cancer gene signature. "
        "Higher score = stronger reversal. **Click any row** to get an AI explanation for that drug."
    )

    col_table, col_chart = st.columns([1.2, 1])

    with col_table:
        display_cols = ["rank", "drug_name", "consensus_score", "n_experiments", "trial_count", "in_repurposedb"]
        available_cols = [c for c in display_cols if c in ranked.columns]
        display_df = ranked.head(top_n_drugs)[available_cols].reset_index(drop=True)

        selection = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )

        csv_out = ranked.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download results CSV",
            data=csv_out,
            file_name="inversigene_results.csv",
            mime="text/csv"
        )

    with col_chart:
        top_viz = ranked.head(top_n_drugs).copy().sort_values("consensus_score")
        chart_height = max(400, top_n_drugs * 28)

        fig_bar = go.Figure(go.Bar(
            x=top_viz["consensus_score"],
            y=top_viz["drug_name"],
            orientation="h",
            marker_color="#2E86AB",
            hovertemplate="%{y}: %{x:.2f}<extra></extra>"
        ))
        fig_bar.update_layout(
            height=chart_height,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=10, t=20, b=0),
            xaxis_title="Reversal Score",
            yaxis=dict(tickfont=dict(size=11)),
            xaxis=dict(tickfont=dict(size=11))
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── AI explanation for selected drug ──────────────────────────────────────
    selected_rows = selection.selection.rows if selection.selection.rows else []

    if selected_rows:
        selected_idx = selected_rows[0]
        drug_row = ranked.iloc[selected_idx]
        drug = drug_row["drug_name"]
        drug_ai_key = f"drug_summary_{drug}"

        st.markdown(f"#### 🤖 AI Explanation — {drug}")

        if drug_ai_key not in st.session_state:
            with st.spinner(f"Generating explanation for {drug}..."):
                try:
                    single_drug_df = pd.DataFrame([drug_row])
                    explanation = summarize_drugs(
                        single_drug_df,
                        top_n=1,
                        up_genes=up_genes,
                        down_genes=down_genes
                    )
                    st.session_state[drug_ai_key] = explanation
                except Exception as e:
                    st.error(f"AI explanation failed: {e}")

        if drug_ai_key in st.session_state:
            st.markdown(st.session_state[drug_ai_key])
    else:
        st.info("👆 Click a row in the table above to get an AI explanation for that drug.")