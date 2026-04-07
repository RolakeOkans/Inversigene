"""
app.py

Inversigene — Breast Cancer Drug Repurposing Tool
Built with Streamlit. Run with: streamlit run app.py

UX hierarchy:
  Level 1 — Summary cards + drug table (always visible)
  Level 2 — Drug profile on row click (target, structure, reversal, AI)
  Level 3 — Full analysis behind "Show full analysis" button
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

from modules.loader import load_signature
from modules.lincs import query_lincs
from modules.scoring import rank_drugs
from modules.validation import validate
from modules.literature import fetch_literature, literature_to_df
from modules.ai_summary import summarize_drugs, summarize_genes
from modules.pathways import get_pathway_enrichment
from modules.reversal import build_reversal_data
from modules.chemistry import get_drug_info, get_chembl_info

FONT_COLOR = "#111111"

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
    st.markdown("A `direction` column is optional — inferred from sign of log2fc.")

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your gene signature CSV",
    type=["csv"],
    help="CSV with at least gene_symbol and log2fc columns"
)
use_demo = st.checkbox("Use built-in GSE45827 breast cancer dataset (demo)")
st.divider()

# ── Reversal chart helper ─────────────────────────────────────────────────────
def render_reversal_chart(sig_df, drug_name, top_n=8, key_prefix="default"):
    reversal_df = build_reversal_data(sig_df, top_n=top_n)
    col_d, col_arr, col_r = st.columns([5, 1, 5])

    with col_d:
        st.markdown("**Disease signature**")
        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(
            x=reversal_df["disease_log2fc"], y=reversal_df["gene"],
            orientation="h",
            marker_color=["#e74c3c" if v > 0 else "#3498db" for v in reversal_df["disease_log2fc"]],
            hovertemplate="<b>%{y}</b><br>log2FC: %{x:.2f}<extra></extra>"
        ))
        fig_d.update_layout(
            height=320, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis_title="log2 Fold Change",
            xaxis=dict(tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR),
                       automargin=True, zeroline=True, zerolinecolor="gray"),
            yaxis=dict(tickfont=dict(color=FONT_COLOR), automargin=True),
            showlegend=False
        )
        st.plotly_chart(fig_d, use_container_width=True, key=f"{key_prefix}_disease")

    with col_arr:
        st.markdown(
            "<div style='text-align:center;font-size:2rem;color:#2E86AB;padding-top:110px;'>→</div>",
            unsafe_allow_html=True
        )

    with col_r:
        st.markdown(f"**{drug_name} reversal**")
        fig_r = go.Figure()
        fig_r.add_trace(go.Bar(
            x=reversal_df["reversal_log2fc"], y=reversal_df["gene"],
            orientation="h",
            marker_color=["#3498db" if v > 0 else "#e74c3c" for v in reversal_df["reversal_log2fc"]],
            hovertemplate="<b>%{y}</b><br>Reversal: %{x:.2f}<extra></extra>"
        ))
        fig_r.update_layout(
            height=320, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis_title="Expected reversal (log2 FC)",
            xaxis=dict(tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR),
                       automargin=True, zeroline=True, zerolinecolor="gray"),
            yaxis=dict(tickfont=dict(color=FONT_COLOR), automargin=True),
            showlegend=False
        )
        st.plotly_chart(fig_r, use_container_width=True, key=f"{key_prefix}_reversal")

    st.caption(
        "Left: genes overexpressed (red) or underexpressed (blue) in cancer. "
        "Right: the drug reverses this pattern — colors flip to show the reversal."
    )

# ── Run button ────────────────────────────────────────────────────────────────
run = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

if run:
    st.session_state["results"] = None
    st.session_state["show_full_analysis"] = False
    for key in list(st.session_state.keys()):
        if key.startswith("gene_summary_") or key.startswith("drug_summary") \
                or key.startswith("chem_") or key.startswith("chembl_"):
            del st.session_state[key]

    if not uploaded_file and not use_demo:
        st.error("Please upload a gene signature CSV or check 'Use built-in dataset'.")
        st.stop()

    progress = st.progress(0, text="Loading gene signature...")

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
            filepath, top_n=top_n_genes, min_log2fc=min_log2fc
        )
    except Exception as e:
        st.error(f"Failed to load gene signature: {e}")
        st.stop()

    progress.progress(15, text="Querying SigCom LINCS for drug signatures...")
    try:
        lincs_df = query_lincs(up_genes, down_genes, n_results=100)
    except Exception as e:
        st.error(f"LINCS query failed: {e}")
        st.stop()

    progress.progress(40, text="Scoring and ranking drugs...")
    try:
        ranked = rank_drugs(lincs_df)
    except Exception as e:
        st.error(f"Scoring failed: {e}")
        st.stop()

    progress.progress(55, text="Checking clinical trials and repurposing databases...")
    try:
        ranked = validate(ranked, top_n=top_n_drugs)
    except Exception as e:
        st.warning(f"Validation step had an issue: {e}")

    progress.progress(68, text="Searching PubMed for key gene literature...")
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

    progress.progress(82, text="Running pathway enrichment via Enrichr...")
    try:
        pathways = get_pathway_enrichment(up_genes, down_genes, top_n=100)
    except Exception as e:
        st.warning(f"Pathway enrichment had an issue: {e}")
        pathways = {"up": {}, "down": {}}

    progress.progress(100, text="✅ Analysis complete!")

    st.session_state["results"] = {
        "ranked": ranked, "lit_df": lit_df, "sig_df": sig_df,
        "literature": literature, "up_genes": up_genes,
        "down_genes": down_genes, "pathways": pathways
    }

# ── Display results ───────────────────────────────────────────────────────────
if "results" in st.session_state and st.session_state["results"] is not None:
    ranked    = st.session_state["results"]["ranked"]
    lit_df    = st.session_state["results"]["lit_df"]
    sig_df    = st.session_state["results"]["sig_df"]
    literature= st.session_state["results"]["literature"]
    up_genes  = st.session_state["results"]["up_genes"]
    down_genes= st.session_state["results"]["down_genes"]
    pathways  = st.session_state["results"]["pathways"]

    # ═══════════════════════════════════════════════════════════════════════════
    # LEVEL 1 — HEADLINE SUMMARY CARDS
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
    # LEVEL 1 — DRUG CANDIDATES TABLE + BAR CHART
    # ═══════════════════════════════════════════════════════════════════════════
    st.subheader("💊 Drug Repurposing Candidates")
    st.markdown(
        "Ranked by how strongly they reverse your cancer gene signature. "
        "**Click any row** to see its full drug profile."
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
            x=top_viz["consensus_score"], y=top_viz["drug_name"],
            orientation="h", marker_color="#2E86AB",
            hovertemplate="%{y}: %{x:.2f}<extra></extra>"
        ))
        fig_bar.update_layout(
            height=chart_height, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=20, t=20, b=40), xaxis_title="Reversal Score",
            xaxis=dict(automargin=True, tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR)),
            yaxis=dict(automargin=True, tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR))
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="drug_bar_chart")

    # ═══════════════════════════════════════════════════════════════════════════
    # LEVEL 2 — DRUG PROFILE ON CLICK
    # ═══════════════════════════════════════════════════════════════════════════
    selected_rows = selection.selection.rows if selection.selection.rows else []

    if selected_rows:
        selected_idx = selected_rows[0]
        drug_row = ranked.iloc[selected_idx]
        drug = drug_row["drug_name"]
        drug_ai_key  = f"drug_summary_{drug}"
        chem_key     = f"chem_{drug}"
        chembl_key   = f"chembl_{drug}"

        st.divider()
        st.subheader(f"💊 {drug} — Drug Profile")

        # Fetch data
        if chem_key not in st.session_state:
            with st.spinner("Fetching chemical structure..."):
                st.session_state[chem_key] = get_drug_info(drug)
        if chembl_key not in st.session_state:
            with st.spinner("Looking up target and indication in ChEMBL..."):
                st.session_state[chembl_key] = get_chembl_info(drug)

        chem   = st.session_state[chem_key]
        chembl = st.session_state[chembl_key]
        trials = int(drug_row.get("trial_count", 0)) if "trial_count" in drug_row else 0
        in_repo = drug_row.get("in_repurposedb", False)

        # ── Decision support cards ────────────────────────────────────────
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown("**🎯 Target**")
            st.markdown(chembl["target"] or "Not found in ChEMBL")
        with d2:
            st.markdown("**💉 Indication**")
            st.markdown(chembl["indication"] or "Not found in ChEMBL")
        with d3:
            st.markdown("**⚙️ Mechanism**")
            st.markdown(chembl["mechanism"] or "Not found in ChEMBL")
        with d4:
            st.markdown("**🏥 Breast Cancer Trials**")
            st.markdown(str(trials))

        st.divider()

        # ── Chemical structure + links ────────────────────────────────────
        if chem["found"]:
            chem_col1, chem_col2 = st.columns([1, 2])
            with chem_col1:
                st.image(chem["structure_url"], caption=f"2D structure — {chem['formula']}", width=250)
            with chem_col2:
                st.markdown(f"**Molecular formula:** `{chem['formula']}`")
                st.markdown(
                    f"**PubChem CID:** [{chem['cid']}]"
                    f"(https://pubchem.ncbi.nlm.nih.gov/compound/{chem['cid']})"
                )
                if chembl["chembl_id"]:
                    st.markdown(
                        f"**ChEMBL ID:** [{chembl['chembl_id']}]"
                        f"(https://www.ebi.ac.uk/chembl/compound_report_card/{chembl['chembl_id']})"
                    )
                st.markdown(f"**In repoDB:** {'✅ Yes' if in_repo else '❌ No'}")
        else:
            st.info(f"Chemical structure not available for {drug} — Broad Institute screening compound.")

        st.divider()

        # ── Reversal chart ────────────────────────────────────────────────
        st.markdown("**🔄 Signature Reversal**")
        render_reversal_chart(sig_df, drug, key_prefix=f"selected_{drug}")

        st.divider()

        # ── AI explanation ────────────────────────────────────────────────
        st.markdown("**🤖 AI Explanation**")
        if drug_ai_key not in st.session_state:
            with st.spinner(f"Generating explanation for {drug}..."):
                try:
                    explanation = summarize_drugs(
                        pd.DataFrame([drug_row]), top_n=1,
                        up_genes=up_genes, down_genes=down_genes
                    )
                    st.session_state[drug_ai_key] = explanation
                except Exception as e:
                    st.error(f"AI explanation failed: {e}")

        if drug_ai_key in st.session_state:
            st.markdown(st.session_state[drug_ai_key])

    else:
        st.info("👆 Click a row in the table above to see its full drug profile.")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # LEVEL 3 — FULL ANALYSIS (behind button)
    # ═══════════════════════════════════════════════════════════════════════════
    if "show_full_analysis" not in st.session_state:
        st.session_state["show_full_analysis"] = False

    if st.button("🔬 Show full analysis", use_container_width=True):
        st.session_state["show_full_analysis"] = not st.session_state["show_full_analysis"]

    if st.session_state["show_full_analysis"]:

        st.divider()

        # ── Gene Signature ────────────────────────────────────────────────
        st.subheader("🔬 Your Gene Signature")
        tab_preview, tab_volcano = st.tabs(["Signature preview", "Volcano plot"])

        with tab_preview:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total genes", len(sig_df))
            col2.metric("Upregulated", (sig_df["direction"] == "up").sum())
            col3.metric("Downregulated", (sig_df["direction"] == "down").sum())
            st.dataframe(sig_df.head(20), use_container_width=True, hide_index=True)

        with tab_volcano:
            st.markdown("Red = upregulated · Blue = downregulated · Top genes labeled")

            vol_df = sig_df.copy()
            vol_df["log2fc"] = pd.to_numeric(vol_df["log2fc"], errors="coerce")
            vol_df["p_value"] = pd.to_numeric(vol_df["p_value"], errors="coerce")
            vol_df = vol_df.dropna(subset=["log2fc", "p_value"])
            vol_df["-log10p"] = -np.log10(vol_df["p_value"].clip(lower=1e-300))
            vol_df = vol_df[np.isfinite(vol_df["-log10p"])]

            vol_df["color"] = "neutral"
            vol_df.loc[(vol_df["log2fc"] > 1) & (vol_df["p_value"] < 0.05), "color"] = "up"
            vol_df.loc[(vol_df["log2fc"] < -1) & (vol_df["p_value"] < 0.05), "color"] = "down"

            top_up   = vol_df[vol_df["color"] == "up"].nlargest(10, "log2fc")
            top_down = vol_df[vol_df["color"] == "down"].nsmallest(10, "log2fc")
            top_genes = set(pd.concat([top_up, top_down])["gene_symbol"])

            colors = {"up": "#e74c3c", "down": "#3498db", "neutral": "#bdc3c7"}
            fig_vol = go.Figure()

            for group, color in colors.items():
                mask   = vol_df["color"] == group
                subset = vol_df[mask]
                unlabeled = subset[~subset["gene_symbol"].isin(top_genes)]
                fig_vol.add_trace(go.Scatter(
                    x=unlabeled["log2fc"], y=unlabeled["-log10p"],
                    mode="markers", marker=dict(size=4, color=color, opacity=0.7),
                    text=unlabeled["gene_symbol"],
                    hovertemplate="<b>%{text}</b><br>log2FC: %{x:.2f}<br>-log10p: %{y:.2f}<extra></extra>",
                    name=group, showlegend=False
                ))
                labeled = subset[subset["gene_symbol"].isin(top_genes)]
                if not labeled.empty:
                    fig_vol.add_trace(go.Scatter(
                        x=labeled["log2fc"], y=labeled["-log10p"],
                        mode="markers+text",
                        marker=dict(size=7, color=color, opacity=1.0),
                        text=labeled["gene_symbol"], textposition="top center",
                        textfont=dict(size=10, color=FONT_COLOR),
                        hovertemplate="<b>%{text}</b><br>log2FC: %{x:.2f}<br>-log10p: %{y:.2f}<extra></extra>",
                        name=group, showlegend=False
                    ))

            fig_vol.add_vline(x=1,  line_dash="dash", line_color="gray", line_width=1)
            fig_vol.add_vline(x=-1, line_dash="dash", line_color="gray", line_width=1)
            fig_vol.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray", line_width=1)
            fig_vol.update_layout(
                height=600, plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                xaxis_title="log2 Fold Change", yaxis_title="-log10(p-value)",
                xaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=True, zerolinecolor="gray",
                           tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR), automargin=True),
                yaxis=dict(showgrid=True, gridcolor="#f0f0f0",
                           tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR), automargin=True)
            )
            st.plotly_chart(fig_vol, use_container_width=True, key="volcano_plot")

        st.divider()

        # ── Pathway Enrichment ────────────────────────────────────────────
        st.subheader("🧬 Pathway Enrichment")
        st.markdown(
            "Dot size = gene count · Dot color = significance · "
            "Expand a pathway to see which genes drive it."
        )

        def dot_color(p):
            if p < 0.0001: return "#A32D2D"
            if p < 0.01:   return "#BA7517"
            return "#185FA5"

        def dot_size(genes_str):
            return max(8, min(30, len(genes_str.split(";")) * 4))

        tab_up_path, tab_down_path = st.tabs(["⬆️ Upregulated pathways", "⬇️ Downregulated pathways"])

        for tab, direction in [(tab_up_path, "up"), (tab_down_path, "down")]:
            with tab:
                dir_pathways = pathways.get(direction, {})
                if not dir_pathways:
                    st.info("No significant pathways found.")
                else:
                    lib_tabs = st.tabs(list(dir_pathways.keys()))
                    for lib_tab, (lib_name, df) in zip(lib_tabs, dir_pathways.items()):
                        with lib_tab:
                            if df.empty:
                                st.info("No significant terms.")
                            else:
                                df_sorted = df.sort_values("combined_score", ascending=True).reset_index(drop=True)
                                fig_path = go.Figure()
                                fig_path.add_trace(go.Scatter(
                                    x=df_sorted["combined_score"],
                                    y=df_sorted["term"],
                                    mode="markers",
                                    marker=dict(
                                        size=[dot_size(g) for g in df_sorted["genes"]],
                                        color=[dot_color(p) for p in df_sorted["adj_pvalue"]],
                                        opacity=0.85, line=dict(width=1, color="white")
                                    ),
                                    customdata=list(zip(
                                        df_sorted["adj_pvalue"],
                                        [len(g.split(";")) for g in df_sorted["genes"]],
                                        df_sorted["genes"]
                                    )),
                                    hovertemplate=(
                                        "<b>%{y}</b><br>Score: %{x:.1f}<br>"
                                        "p = %{customdata[0]:.2e}<br>"
                                        "Genes (%{customdata[1]}): %{customdata[2]}<extra></extra>"
                                    )
                                ))
                                fig_path.update_layout(
                                    height=max(300, len(df) * 50),
                                    plot_bgcolor="white", paper_bgcolor="white",
                                    margin=dict(l=10, r=20, t=10, b=40),
                                    xaxis_title="Combined score", showlegend=False,
                                    xaxis=dict(gridcolor="#f0f0f0", automargin=True,
                                               tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR)),
                                    yaxis=dict(automargin=True,
                                               tickfont=dict(color=FONT_COLOR), title_font=dict(color=FONT_COLOR))
                                )
                                st.plotly_chart(fig_path, use_container_width=True,
                                                key=f"pathway_{direction}_{lib_name}")

                                leg1, leg2, leg3, leg4 = st.columns(4)
                                leg1.markdown("🔴 p < 0.0001")
                                leg2.markdown("🟠 p < 0.01")
                                leg3.markdown("🔵 p < 0.05")
                                leg4.markdown("⚫ size = gene count")
                                st.markdown("---")

                                for _, row in df.iterrows():
                                    genes_list = row["genes"].split(";")
                                    with st.expander(
                                        f"**{row['term']}** — p = {row['adj_pvalue']:.2e} | {len(genes_list)} genes"
                                    ):
                                        st.markdown(f"**Combined score:** {row['combined_score']:.2f}")
                                        st.markdown("**Genes driving this pathway:**")
                                        st.code(" | ".join(genes_list))

        st.divider()

        # ── Gene Literature ───────────────────────────────────────────────
        st.subheader("📚 Gene Literature & AI Synthesis")
        st.markdown(
            "Expand a gene to read published abstracts and generate an AI summary inline."
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
                                gene_lit = {direction: [row.to_dict() for _, row in gene_abstracts.iterrows()]}
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