"""
Talent Pool Assessment Dashboard (multipage)

This page is intentionally separate from the existing `app.py` dashboard, so the
original dashboard isn't modified.

Run:
  streamlit run app.py
and select this page from the Streamlit sidebar.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import zlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import app as talent_app


# Reuse the project's data/logic helpers
BASE = talent_app.BASE
MODELS = talent_app.MODELS
STATS = talent_app.STATS
OUTCOMES = talent_app.OUTCOMES
PERCENT_COLS = talent_app.PERCENT_COLS
DEFAULT_SOURCE = talent_app.DEFAULT_SOURCE
CLEANED_PATH = talent_app.CLEANED_PATH

(
    col_prob,
    prob_columns_for,
    style_pct,
    at_least_one_probability_pct,
    cohort_at_least_one_table,
    merge_cohort_tags,
    filter_dataframe,
    load_cleaned_csv,
    load_recleaned,
) = (
    talent_app.col_prob,
    talent_app.prob_columns_for,
    talent_app.style_pct,
    talent_app.at_least_one_probability_pct,
    talent_app.cohort_at_least_one_table,
    talent_app.merge_cohort_tags,
    talent_app.filter_dataframe,
    talent_app.load_cleaned_csv,
    talent_app.load_recleaned,
)

load_and_clean = talent_app.load_and_clean


THEMES: dict[str, dict[str, str]] = {
    "Light (Clean)": {
        "bg": "#ffffff",
        "panel": "#f7f8fb",
        "text": "#0f172a",
        "muted": "#475569",
        "accent": "#2563eb",
    },
    "Dark (Midnight)": {
        "bg": "#0b1220",
        "panel": "#0f1b31",
        "text": "#e5e7eb",
        "muted": "#9ca3af",
        "accent": "#60a5fa",
    },
    "Corporate Blue": {
        "bg": "#f8fafc",
        "panel": "#eaf2ff",
        "text": "#0b1b3a",
        "muted": "#41506a",
        "accent": "#1d4ed8",
    },
    "Modern Slate": {
        "bg": "#f3f4f6",
        "panel": "#e5e7eb",
        "text": "#111827",
        "muted": "#4b5563",
        "accent": "#0ea5e9",
    },
    "Graphite (Minimal)": {
        "bg": "#111827",
        "panel": "#0f172a",
        "text": "#f9fafb",
        "muted": "#cbd5e1",
        "accent": "#a3e635",
    },
}


def apply_theme(theme_name: str) -> None:
    """Best-effort theming via CSS injection (Streamlit doesn't expose full theme switching)."""

    t = THEMES.get(theme_name, THEMES["Light (Clean)"])
    css = f"""
    <style>
      html, body {{
        background: {t["bg"]} !important;
        color: {t["text"]} !important;
      }}

      /* Main app container */
      .stApp {{
        background: {t["bg"]} !important;
      }}

      /* Sidebar */
      [data-testid="stSidebar"] {{
        background: {t["panel"]} !important;
      }}

      /* Text */
      .stMarkdown, .stText, .stCaption {{
        color: {t["text"]} !important;
      }}
      .stCaption, .stMarkdown span {{
        color: {t["muted"]} !important;
      }}

      /* Metric cards */
      [data-testid="stMetric"] {{
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
      }}

      /* Links */
      a {{
        color: {t["accent"]} !important;
      }}

      /* Plotly backgrounds (best-effort) */
      .js-plotly-plot .plotly .main-svg {{
        background: transparent !important;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def perf_col_for_stat(stat: str) -> str:
    """Map probability-stat names to performance columns in the dataset."""
    return "Best Score" if stat == "Best" else stat


def get_prob_cols(dff: pd.DataFrame, *, models: list[str], stat: str, outcomes: list[str]) -> list[str]:
    cols: list[str] = []
    for m in models:
        for o in outcomes:
            c = col_prob(m, stat, o)
            if c in dff.columns:
                cols.append(c)
    return cols


def _stable_u32_seed(seed: int, *parts: str) -> int:
    b = ("|".join([str(seed), *parts])).encode("utf-8", errors="ignore")
    return zlib.crc32(b) & 0xFFFFFFFF


def monte_carlo_any_success(prob_pcts: np.ndarray, *, n_sims: int, seed: int) -> float:
    """
    Monte Carlo estimate for P(at least one success), where each athlete i succeeds
    independently with probability p_i (input in 0-100 scale).
    """
    if prob_pcts.size == 0:
        return float("nan")

    p = np.asarray(prob_pcts, dtype=float) / 100.0
    p = np.clip(p, 0.0, 1.0)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)

    rng = np.random.default_rng(seed)
    # Success matrix: n_sims x n_athletes
    draws = rng.random((n_sims, p.size))
    any_success = (draws < p).any(axis=1)
    return float(any_success.mean() * 100.0)


@st.cache_data(show_spinner=False)
def compute_event_pool_monte_carlo(
    dff: pd.DataFrame,
    *,
    top_n: int,
    n_sims: int,
    seed: int,
    methods: tuple[str, ...],
    stats: tuple[str, ...],
) -> pd.DataFrame:
    """
    For each Event Name:
      - for each stat in (Mean/Median/Best), select the top N athletes by that stat score
      - for each method, Monte Carlo simulate independent medal-success events using the
        corresponding athlete-level medal probabilities.

    Output: one row per Event Name, with columns like:
      P_at_least_one_medal_%_{method}_{stat}
    """
    if dff.empty:
        return pd.DataFrame()

    out_rows: list[dict[str, object]] = []

    # Keep only what we need in memory
    has_event = "Event Name" in dff.columns
    if not has_event:
        return pd.DataFrame()

    events = sorted(dff["Event Name"].dropna().unique().tolist())
    for ev in events:
        g = dff[dff["Event Name"] == ev].copy()
        row: dict[str, object] = {"Event Name": ev}
        for stat in stats:
            pc = perf_col_for_stat(stat)
            if pc not in g.columns:
                continue
            g_stat = g.dropna(subset=[pc]).copy()
            if g_stat.empty:
                continue
            g_top = g_stat.nlargest(top_n, pc, keep="all")
            n_top = len(g_top)
            row[f"n_athletes_top_{stat}"] = int(n_top)

            for method in methods:
                prob_col = col_prob(method, stat, "Medal")
                if prob_col not in g_top.columns:
                    continue
                prob_pcts = pd.to_numeric(g_top[prob_col], errors="coerce").dropna().to_numpy(dtype=float)
                sim_seed = _stable_u32_seed(seed, str(ev), method, stat)
                p_medal_pool = monte_carlo_any_success(
                    prob_pcts,
                    n_sims=n_sims,
                    seed=sim_seed,
                )
                row[f"P_at_least_one_medal_%_{method}_{stat}"] = p_medal_pool
        out_rows.append(row)

    df_out = pd.DataFrame(out_rows)
    if "Event Name" in df_out.columns:
        df_out = df_out.sort_values("Event Name").reset_index(drop=True)
    return df_out


def main() -> None:
    st.set_page_config(
        page_title="Talent Pool Assessment Dashboard",
        page_icon="",
        layout="wide",
    )

    st.title("Talent Pool Assessment Dashboard")

    with st.sidebar:
        st.header("Display settings")

        display_models = st.multiselect(
            "Methods to display",
            options=MODELS,
            default=MODELS,
            help="Used across tabs for plots and probability tables.",
        )
        display_stat = st.selectbox(
            "Probability basis",
            options=STATS,
            index=0,
            help="Mean / Median / Best are used as the default probability basis in charts and tables.",
        )
        display_outcome = st.selectbox(
            "Outcome basis (for some charts)",
            options=OUTCOMES,
            index=1,
            help="Used for event-level probability charts (box plots, histograms) where relevant.",
        )

    # ---- Data loading + filtering sidebar (reuse original layout) ----
    with st.sidebar:
        st.header("Data")
        use_cleaned = st.radio(
            "Load",
            options=["cleaned_file", "raw_reclean"],
            format_func=lambda x: "Saved cleaned file" if x == "cleaned_file" else "Re-clean from raw CSV",
            label_visibility="collapsed",
        )
        raw_path = st.text_input("Raw CSV path", value=str(DEFAULT_SOURCE))
        cleaned_path = st.text_input("Cleaned CSV path", value=str(CLEANED_PATH))
        if st.button("Save fresh clean to disk", disabled=use_cleaned != "raw_reclean"):
            p = Path(cleaned_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            df_w = load_and_clean(Path(raw_path))
            df_w.to_csv(p, index=False)
            st.success(f"Saved {len(df_w):,} rows → {p}")
            st.cache_data.clear()

        if use_cleaned == "cleaned_file" and Path(cleaned_path).exists():
            df = load_cleaned_csv(cleaned_path)
        else:
            if use_cleaned == "cleaned_file":
                st.warning("Cleaned file missing; re-cleaning from raw.")
            df = load_recleaned(raw_path)

        st.divider()
        st.subheader("Cohort tags (optional)")
        st.caption(
            "Add **competition_scope** (e.g. National / International), **championship**, etc. via "
            "`data/cohort_tags.csv` (see `cohort_tags.example.csv` in the same folder)."
        )
        tag_upload = st.file_uploader("Upload tags CSV", type=["csv"], help="Must include Athlete ID column")
        default_tag_path = BASE / "data" / "cohort_tags.csv"
        tag_path_in = st.text_input("Tags CSV path (if not uploading)", value=str(default_tag_path))
        tags_df: pd.DataFrame | None = None
        if tag_upload is not None:
            tags_df = pd.read_csv(tag_upload)
        elif Path(tag_path_in).is_file():
            tags_df = pd.read_csv(tag_path_in)
        df = merge_cohort_tags(df, tags_df)

        st.divider()
        st.subheader("Filters")
        cats = sorted(df["Category"].dropna().unique().tolist())
        genders = sorted(df["Gender"].dropna().unique().tolist())
        all_events = sorted(df["Event Name"].dropna().unique().tolist())
        sel_cat = st.multiselect("Category", options=cats, default=None)
        sel_gen = st.multiselect("Gender", options=genders, default=None)
        sel_ev = st.multiselect("Events (optional)", options=all_events, default=None, help="Empty = all events")
        only_ok = st.checkbox("Only rows passing performance check", value=False)

        with st.expander("Age & performance cuts", expanded=False):
            age_num = pd.to_numeric(df["Age"], errors="coerce").dropna()
            if len(age_num):
                a_lo, a_hi = int(age_num.min()), int(age_num.max())
                a_lo = max(10, a_lo - 2)
                a_hi = min(55, a_hi + 2)
            else:
                a_lo, a_hi = 14, 40
            age_range = st.slider("Age range (years)", min_value=a_lo, max_value=a_hi, value=(a_lo, a_hi))
            incl_age_na = st.checkbox("Include rows with unknown age", value=True)
            apply_age = st.checkbox("Apply age filter", value=False)

            mean_num = pd.to_numeric(df["Mean"], errors="coerce").dropna()
            if len(mean_num):
                m_lo, m_hi = float(mean_num.quantile(0.01)), float(mean_num.quantile(0.99))
                if m_lo == m_hi:
                    m_lo, m_hi = float(mean_num.min()), float(mean_num.max())
            else:
                m_lo, m_hi = 0.0, 10000.0
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                mean_min_in = st.number_input("Mean min (performance)", value=m_lo, format="%.4f")
            with col_m2:
                mean_max_in = st.number_input("Mean max", value=m_hi, format="%.4f")
            incl_mean_na = st.checkbox("Include rows with missing Mean", value=True)
            apply_mean = st.checkbox("Apply Mean range filter", value=False)

        default_filter_model = display_models[0] if display_models else MODELS[0]
        with st.expander("Model probability band", expanded=False):
            f_mod = st.selectbox("Model", MODELS, index=MODELS.index(default_filter_model), key="f_mod")
            f_stat = st.selectbox("Statistic", STATS, index=STATS.index(display_stat), key="f_stat")
            f_out = st.selectbox("Outcome", OUTCOMES, index=OUTCOMES.index(display_outcome), key="f_out")
            prob_col_f = col_prob(f_mod, f_stat, f_out)
            prob_series = pd.to_numeric(df[prob_col_f], errors="coerce").dropna() if prob_col_f in df.columns else pd.Series(dtype=float)
            if len(prob_series):
                p_lo, p_hi = float(prob_series.min()), float(prob_series.max())
            else:
                p_lo, p_hi = 0.0, 100.0
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                prob_min_in = st.number_input("Min %", min_value=0.0, max_value=100.0, value=p_lo)
            with col_p2:
                prob_max_in = st.number_input("Max %", min_value=0.0, max_value=100.0, value=p_hi)
            incl_prob_na = st.checkbox("Include rows missing this probability", value=True, key="inpna")
            apply_prob = st.checkbox("Apply probability band", value=False)

        with st.expander("Text & cohort fields", expanded=False):
            ev_sub = st.text_input("Event name contains (substring)", value="")
            ath_sub = st.text_input("Athlete name contains", value="")
            scope_opts: list[str] = []
            champ_opts: list[str] = []
            if "competition_scope" in df.columns:
                scope_opts = sorted(df["competition_scope"].dropna().astype(str).str.strip().unique().tolist())
                scope_opts = [s for s in scope_opts if s and s.lower() != "nan"]
            if "championship" in df.columns:
                champ_opts = sorted(df["championship"].dropna().astype(str).str.strip().unique().tolist())
                champ_opts = [s for s in champ_opts if s and s.lower() != "nan"]
            sel_scope = st.multiselect(
                "Competition scope",
                options=scope_opts,
                default=None,
                help="National / International / … from your tags CSV. Empty selection = no filter.",
            )
            untagged_scope = st.checkbox("Include untagged rows (no competition_scope)", value=True, key="uts")
            sel_champ = st.multiselect(
                "Championship / meet tag",
                options=champ_opts,
                default=None,
                help="From tags CSV championship column.",
            )
            untagged_champ = st.checkbox("Include untagged championship", value=True, key="utc")

        dff = filter_dataframe(
            df,
            categories=sel_cat or None,
            genders=sel_gen or None,
            events=sel_ev or None,
            only_valid=only_ok,
            age_min=age_range[0] if apply_age else None,
            age_max=age_range[1] if apply_age else None,
            include_unknown_age=incl_age_na if apply_age else True,
            mean_min=mean_min_in if apply_mean else None,
            mean_max=mean_max_in if apply_mean else None,
            include_unknown_mean=incl_mean_na if apply_mean else True,
            prob_col=prob_col_f if apply_prob else None,
            prob_min=prob_min_in if apply_prob else None,
            prob_max=prob_max_in if apply_prob else None,
            include_unknown_prob=incl_prob_na if apply_prob else True,
            event_substring=ev_sub,
            athlete_substring=ath_sub,
            competition_scope=sel_scope or None,
            championship=sel_champ or None,
            include_untagged_scope=untagged_scope,
            include_untagged_championship=untagged_champ,
        )

        st.metric(
            "Filtered rows",
            f"{len(dff):,}",
            delta=f"{len(dff) - len(df):,}" if len(dff) != len(df) else None,
        )
        st.caption(f"Total in file (after tag merge): {len(df):,}")

    prob_cols_display = get_prob_cols(dff, models=display_models or MODELS, stat=display_stat, outcomes=OUTCOMES)

    tabs = st.tabs(
        [
            "Overview",
            "Models",
            "Events",
            "Athletes",
            "Leaderboards",
            "Cohort ≥1",
            "Cohort - Event Pool",
            "Data quality",
        ]
    )
    tab_ov, tab_mod, tab_ev, tab_ath, tab_lb, tab_cohort, tab_mc, tab_q = tabs

    # ---------- Overview ----------
    with tab_ov:
        if dff.empty:
            st.warning("No rows match the sidebar filters. Clear filters or widen your selection.")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Athletes", dff["Athlete ID"].nunique() if len(dff) else 0)
        with k2:
            st.metric("Events", dff["Event Name"].nunique() if len(dff) else 0)
        with k3:
            st.metric("Categories", dff["Category"].nunique() if len(dff) else 0)
        with k4:
            v = 100 * dff["performance_flag_ok"].mean() if len(dff) else 0
            st.metric("Pass performance check", f"{v:.1f}%")

        if not dff.empty:
            r1, r2 = st.columns(2)
            with r1:
                cc = dff["Category"].value_counts().reset_index()
                cc.columns = ["Category", "n"]
                st.plotly_chart(px.bar(cc, x="Category", y="n", title="Rows by category"), use_container_width=True)
            with r2:
                gg = dff["Gender"].value_counts().reset_index()
                gg.columns = ["Gender", "n"]
                st.plotly_chart(px.pie(gg, names="Gender", values="n", title="Rows by gender"), use_container_width=True)

        stat_label = "Best Score" if display_stat == "Best" else display_stat
        model_subset = display_models or MODELS

        # Box plots: Qualify
        st.subheader(f"Model comparison — {stat_label} × Qualify (all selected methods)")
        q_cols = [col_prob(m, display_stat, "Qualify") for m in model_subset if col_prob(m, display_stat, "Qualify") in dff.columns]
        if q_cols and not dff.empty:
            melt_q = dff[q_cols + ["Athlete Name", "Event Name"]].melt(
                id_vars=["Athlete Name", "Event Name"],
                value_vars=q_cols,
                var_name="Model",
                value_name="pct",
            )
            melt_q["Model"] = melt_q["Model"].str.replace(f"_{display_stat}_Qualify", "", regex=False)
            fig_box = px.box(
                melt_q,
                x="Model",
                y="pct",
                color="Model",
                points=False,
                title=f"Distribution of P(qualify | {stat_label}) by model",
            )
            fig_box.update_layout(showlegend=False, yaxis_title="Probability (%)")
            st.plotly_chart(fig_box, use_container_width=True)

            st.subheader(f"Histograms — Qualify % (per selected model) based on {stat_label}")
            fig_hist = make_subplots(rows=2, cols=3, subplot_titles=model_subset[:6])
            for i, c in enumerate(q_cols[:6]):
                row, col = i // 3 + 1, i % 3 + 1
                vals = dff[c].dropna()
                fig_hist.add_trace(
                    go.Histogram(x=vals, name=c, showlegend=False, nbinsx=35),
                    row=row,
                    col=col,
                )
            fig_hist.update_layout(height=520, title_text="")
            fig_hist.update_xaxes(title_text="%")
            st.plotly_chart(fig_hist, use_container_width=True)

        # Box plots: Medal
        st.subheader(f"Model comparison — {stat_label} × Medal (all selected methods)")
        m_cols = [col_prob(m, display_stat, "Medal") for m in model_subset if col_prob(m, display_stat, "Medal") in dff.columns]
        if m_cols and not dff.empty:
            melt_m = dff[m_cols + ["Athlete Name", "Event Name"]].melt(
                id_vars=["Athlete Name", "Event Name"],
                value_vars=m_cols,
                var_name="Model",
                value_name="pct",
            )
            melt_m["Model"] = melt_m["Model"].str.replace(f"_{display_stat}_Medal", "", regex=False)
            fig_box2 = px.box(
                melt_m,
                x="Model",
                y="pct",
                color="Model",
                points=False,
                title=f"Distribution of P(medal | {stat_label}) by model",
            )
            fig_box2.update_layout(showlegend=False, yaxis_title="Probability (%)")
            st.plotly_chart(fig_box2, use_container_width=True)

            st.subheader(f"Histograms — Medal % (per selected model) based on {stat_label}")
            fig_hist2 = make_subplots(rows=2, cols=3, subplot_titles=model_subset[:6])
            for i, c in enumerate(m_cols[:6]):
                row, col = i // 3 + 1, i % 3 + 1
                vals = dff[c].dropna()
                fig_hist2.add_trace(
                    go.Histogram(x=vals, name=c, showlegend=False, nbinsx=35),
                    row=row,
                    col=col,
                )
            fig_hist2.update_layout(height=520, title_text="")
            fig_hist2.update_xaxes(title_text="%")
            st.plotly_chart(fig_hist2, use_container_width=True)

    # ---------- Models ----------
    with tab_mod:
        c1, c2, c3 = st.columns(3)
        with c1:
            stat_pick = st.selectbox("Statistic", STATS, index=STATS.index(display_stat), key="m_stat")
        with c2:
            out_pick = st.selectbox("Outcome", OUTCOMES, index=OUTCOMES.index(display_outcome), key="m_out")
        with c3:
            sample_cap = st.slider("Max points for scatter / matrix", 1000, 8000, 4000)

        model_subset = display_models or MODELS
        cols_sel = [col_prob(m, stat_pick, out_pick) for m in model_subset if col_prob(m, stat_pick, out_pick) in dff.columns]

        if not cols_sel:
            st.warning("No probability columns for this combination.")
        else:
            sub = dff[cols_sel].dropna()
            if len(sub) > sample_cap:
                sub = sub.sample(sample_cap, random_state=42)

            st.subheader(f"Correlation matrix ({stat_pick} × {out_pick}) across selected methods")
            if len(sub) > 2:
                corr = sub.corr()
                st.plotly_chart(
                    px.imshow(
                        corr,
                        text_auto=".2f",
                        title=f"{stat_pick} × {out_pick}: correlation across selected models",
                        color_continuous_scale="RdBu",
                        zmin=-1,
                        zmax=1,
                    ),
                    use_container_width=True,
                )

            st.subheader("Pairwise: model vs model (selected methods)")
            pc1, pc2 = st.columns(2)
            with pc1:
                ma = st.selectbox("Model (X)", model_subset, index=0, key="px")
            with pc2:
                mb = st.selectbox("Model (Y)", model_subset, index=min(3, len(model_subset) - 1), key="py")
            cx, cy = col_prob(ma, stat_pick, out_pick), col_prob(mb, stat_pick, out_pick)
            if cx in dff.columns and cy in dff.columns:
                pair = dff[[cx, cy, "Athlete Name", "Event Name", "Category"]].dropna()
                if len(pair) > sample_cap:
                    pair = pair.sample(sample_cap, random_state=1)
                fig_p = px.scatter(
                    pair,
                    x=cx,
                    y=cy,
                    color="Category",
                    hover_data=["Athlete Name", "Event Name"],
                    labels={cx: f"{ma} (%)", cy: f"{mb} (%)"},
                    title=f"{stat_pick} / {out_pick}: {ma} vs {mb}",
                )
                fig_p.update_xaxes(title=f"{ma} (%)")
                fig_p.update_yaxes(title=f"{mb} (%)")
                st.plotly_chart(fig_p, use_container_width=True)

            st.subheader("Qualify vs Medal (same model & statistic)")
            mod_vm = st.selectbox("Model", model_subset, index=0, key="vmod")
            cq = col_prob(mod_vm, stat_pick, "Qualify")
            cm = col_prob(mod_vm, stat_pick, "Medal")
            if cq in dff.columns and cm in dff.columns:
                qm = dff[[cq, cm, "Athlete Name", "Event Name"]].dropna()
                if len(qm) > sample_cap:
                    qm = qm.sample(sample_cap, random_state=2)
                fig_qm = px.scatter(
                    qm,
                    x=cq,
                    y=cm,
                    hover_data=["Athlete Name", "Event Name"],
                    labels={cq: "Qualify (%)", cm: "Medal (%)"},
                    title=f"{mod_vm} — {stat_pick}: Qualify % vs Medal %",
                )
                st.plotly_chart(fig_qm, use_container_width=True)

    # ---------- Events ----------
    with tab_ev:
        ev_list = sorted(dff["Event Name"].dropna().unique().tolist()) if len(dff) else []
        if not ev_list:
            st.info("No rows after filters.")
        else:
            pick = st.selectbox("Event", ev_list, key="ev_pick")

            model_subset = display_models or MODELS
            ev_stat_pick = st.selectbox("Statistic (probabilities + performance)", STATS, index=STATS.index(display_stat), key="ev_stat")
            ev_out_pick = st.selectbox("Outcome", OUTCOMES, index=OUTCOMES.index(display_outcome), key="ev_out")
            sort_mod = st.selectbox("Sort model", model_subset, index=0, key="ev_sort_mod")
            sort_col = col_prob(sort_mod, ev_stat_pick, ev_out_pick)

            sub = dff[dff["Event Name"] == pick].copy()

            perf_col = perf_col_for_stat(ev_stat_pick)
            base_cols = ["Athlete Name", "Gender", "Age", "Category", "Mean", "Median", "Best Score", perf_col, "performance_flag_ok"]
            base_cols = [c for c in base_cols if c in sub.columns and c != perf_col] + (["Median"] if ev_stat_pick == "Median" and "Median" in sub.columns else [])  # keep simple; table selection below

            # Ensure base performance + standards are shown when available
            table_cols = []
            for c in ["Athlete Name", "Gender", "Age", "Category", "Mean", "Median", "Best Score", "Qualification Standard", "Medal Standard", "performance_flag_ok"]:
                if c in sub.columns:
                    table_cols.append(c)

            prob_show_cols = []
            for m in model_subset:
                for o in OUTCOMES:  # show both qualify + medal for the chosen stat
                    c = col_prob(m, ev_stat_pick, o)
                    if c in sub.columns:
                        prob_show_cols.append(c)

            ordered = table_cols + prob_show_cols
            if sort_col in sub.columns:
                sub = sub.sort_values(sort_col, ascending=False, na_position="last")

            st.write(f"**{len(sub)}** rows — {ev_stat_pick} probabilities as **%** (selected methods).")
            st.dataframe(style_pct(sub[ordered]), use_container_width=True, height=min(520, 120 + 28 * len(sub)))

            if perf_col in sub.columns and sort_col in sub.columns:
                fig_ev = px.scatter(
                    sub,
                    x=perf_col,
                    y=sort_col,
                    color="Gender",
                    hover_data=["Athlete Name", "Event Name"],
                    labels={sort_col: f"{sort_mod} {ev_out_pick} (%)", perf_col: perf_col},
                    title=f"{pick}: {perf_col} vs {sort_mod} P({ev_out_pick} | {ev_stat_pick})",
                )
                fig_ev.update_yaxes(title=f"{sort_mod} {ev_out_pick} (%)")
                st.plotly_chart(fig_ev, use_container_width=True)

            # Box plot: compare probability distributions across methods for the event
            st.subheader(f"Event box plots — {pick} : P({ev_out_pick} | {ev_stat_pick}) by method")
            p_cols_event = [col_prob(m, ev_stat_pick, ev_out_pick) for m in model_subset if col_prob(m, ev_stat_pick, ev_out_pick) in sub.columns]
            if p_cols_event:
                box_df = sub[["Gender", *p_cols_event]].melt(
                    id_vars=["Gender"],
                    value_vars=p_cols_event,
                    var_name="ModelProb",
                    value_name="pct",
                )
                box_df["Model"] = box_df["ModelProb"].str.replace(f"_{ev_stat_pick}_{ev_out_pick}", "", regex=False)
                fig_box = px.box(
                    box_df,
                    x="Model",
                    y="pct",
                    color="Gender",
                    points=False,
                    title=f"Distribution by method for {pick} ({ev_out_pick} | {ev_stat_pick})",
                )
                fig_box.update_layout(showlegend=True, yaxis_title="Probability (%)")
                st.plotly_chart(fig_box, use_container_width=True)

    # ---------- Athletes ----------
    with tab_ath:
        names = sorted(dff["Athlete Name"].dropna().unique().tolist()) if len(dff) else []
        name = st.selectbox("Athlete", names, key="ath_name") if names else None
        if name:
            ap = dff[dff["Athlete Name"] == name].copy()
            st.write(f"**{len(ap)}** event rows")
            base_a = [c for c in ["Event Name", "Category", "Gender", "Age", "Mean", "Median", "Best Score", "Qualification Standard", "Medal Standard", "performance_flag_ok"] if c in ap.columns]

            stat_for_heat = st.selectbox("Heatmap basis statistic", STATS, index=STATS.index(display_stat), key="ath_stat")
            out_for_heat = st.selectbox("Heatmap basis outcome", OUTCOMES, index=OUTCOMES.index(display_outcome), key="ath_out")

            prob_a = get_prob_cols(ap, models=display_models or MODELS, stat=stat_for_heat, outcomes=OUTCOMES)
            st.dataframe(style_pct(ap[base_a + prob_a]), use_container_width=True, height=420)

            # Heatmap: events × model (stat/outcome)
            p_cols_heat = [col_prob(m, stat_for_heat, out_for_heat) for m in (display_models or MODELS)]
            p_cols_heat = [c for c in p_cols_heat if c in ap.columns]
            if p_cols_heat and "Event Name" in ap.columns:
                heat = ap.set_index("Event Name")[p_cols_heat].copy()
                heat.columns = [c.replace(f"_{stat_for_heat}_{out_for_heat}", "") for c in p_cols_heat]
                st.subheader(f"{stat_for_heat} × {out_for_heat} % — models vs event")
                st.plotly_chart(
                    px.imshow(
                        heat,
                        text_auto=".0f",
                        aspect="auto",
                        title=f"{name}: P({out_for_heat} | {stat_for_heat}) by model and event",
                        labels=dict(x="Model", y="Event", color="Probability (%)"),
                    ),
                    use_container_width=True,
                )

    # ---------- Leaderboards ----------
    with tab_lb:
        ev_lb = st.selectbox("Event", sorted(dff["Event Name"].dropna().unique()), key="lb_ev") if len(dff) else None
        if ev_lb:
            stat_lb = st.selectbox("Statistic", STATS, key="lb_st")
            out_lb = st.selectbox("Outcome", OUTCOMES, key="lb_out")
            top_n = st.slider("Top N", 5, 60, 15)

            model_subset = display_models or MODELS
            rank_mod = st.selectbox("Rank by model", model_subset, index=min(1, len(model_subset) - 1), key="lb_rank")
            rank_col = col_prob(rank_mod, stat_lb, out_lb)

            pool = dff[dff["Event Name"] == ev_lb].copy()
            if rank_col not in pool.columns:
                st.error("Column missing.")
            else:
                pool = pool.dropna(subset=[rank_col])
                top = pool.nlargest(top_n, rank_col, keep="all").head(top_n)

                show = ["Athlete Name", "Gender", "Age", "Mean", "Median", "Best Score", rank_col]
                for m in model_subset:
                    c = col_prob(m, stat_lb, out_lb)
                    if c in top.columns and c not in show:
                        show.append(c)
                show = [c for c in show if c in top.columns]

                st.markdown(f"**Top {top_n}** for **{ev_lb}** — ranked by **{rank_mod}** ({stat_lb} × {out_lb})")
                st.dataframe(style_pct(top[show]), use_container_width=True)

                if len(top) >= 1:
                    row0 = top.iloc[0]
                    bar_x, bar_y = [], []
                    for m in model_subset:
                        c = col_prob(m, stat_lb, out_lb)
                        if c in row0.index and pd.notna(row0[c]):
                            bar_x.append(m)
                            bar_y.append(float(row0[c]))
                    if bar_y:
                        st.plotly_chart(
                            go.Figure(
                                data=[
                                    go.Bar(
                                        x=bar_x,
                                        y=bar_y,
                                        text=[f"{v:.1f}%" for v in bar_y],
                                        textposition="auto",
                                    )
                                ],
                                layout=dict(
                                    title=f"#1 ({row0.get('Athlete Name', '')}): all selected methods — {stat_lb} {out_lb} %",
                                    yaxis_title="%",
                                    xaxis_title="Model",
                                ),
                            ),
                            use_container_width=True,
                        )

    # ---------- Cohort: P(≥1 qualify) & P(≥1 medal) ----------
    with tab_cohort:
        st.subheader("Event cohort: at least one qualifier / medalist")
        st.markdown(
            """
            For each **event** in your filtered dataset, the cohort is the set of athletes in that event
            (**sidebar filters** still apply). Using the chosen **model** and **statistic** columns as benchmarks
            (values are interpreted as %):

            - **P(≥1 qualifies)** = **1 − Π(1 − q_i)**
            - **P(≥1 wins medal)** = **1 − Π(1 − m_i)**
            """
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            coh_model = st.selectbox("Model (benchmark)", (display_models or MODELS), index=0, key="coh_mod")
        with c2:
            coh_stat = st.selectbox("Statistic", STATS, index=STATS.index(display_stat), key="coh_st")
        with c3:
            dedupe = st.checkbox(
                "One row per athlete per event",
                value=True,
                help="If duplicates exist for the same Athlete ID + Event Name, keep the first row only.",
            )

        cq = col_prob(coh_model, coh_stat, "Qualify")
        cm = col_prob(coh_model, coh_stat, "Medal")

        if dff.empty:
            st.warning("No rows match filters.")
        elif cq not in dff.columns or cm not in dff.columns:
            st.error(f"Missing columns: `{cq}` and/or `{cm}`. Reload cleaned data.")
        else:
            cohort_df = cohort_at_least_one_table(dff, cq, cm, dedupe_athlete=dedupe)
            if cohort_df.empty:
                st.info("No events in cohort.")
            else:
                cohort_df = cohort_df.sort_values("P_at_least_one_qualifies_%", ascending=False, na_position="last").reset_index(drop=True)
                st.caption(f"Benchmark columns: **{cq}** and **{cm}** (values interpreted as %).")
                show_c = cohort_df[["Event Name", "n_athletes", "P_at_least_one_qualifies_%", "P_at_least_one_medal_%"]].copy()
                sty_c = show_c.style.format(
                    {"P_at_least_one_qualifies_%": "{:.2f}%", "P_at_least_one_medal_%": "{:.2f}%"},
                )
                st.dataframe(sty_c, use_container_width=True)

        st.divider()
        st.subheader("All methods: P(≥1 medal) together (Mean/Median/Best)")
        coh_medal_stats = st.multiselect("Medal probability stats", STATS, default=STATS, key="med_stats")
        model_subset = display_models or MODELS

        if dff.empty or not coh_medal_stats or not model_subset:
            st.info("Select stats / methods to display.")
        else:
            # Compute cohort medal probabilities for each event, across all selected models and stats.
            w = dff.copy()
            if dedupe and "Athlete ID" in w.columns and "Event Name" in w.columns:
                w = w.drop_duplicates(subset=["Event Name", "Athlete ID"], keep="first")

            rows: list[dict[str, object]] = []
            for ev, g in w.groupby("Event Name", sort=True):
                n = int(g["Athlete ID"].nunique()) if "Athlete ID" in g.columns else len(g)
                row: dict[str, object] = {"Event Name": ev, "n_athletes": n}
                for stat in coh_medal_stats:
                    for method in model_subset:
                        c = col_prob(method, stat, "Medal")
                        if c in g.columns:
                            row[f"P_at_least_one_medal_%_{method}_{stat}"] = at_least_one_probability_pct(g[c])
                rows.append(row)

            all_medals = pd.DataFrame(rows)
            if not all_medals.empty:
                # Order columns by stat then method
                desired_cols = ["Event Name", "n_athletes"]
                for stat in coh_medal_stats:
                    for method in model_subset:
                        key = f"P_at_least_one_medal_%_{method}_{stat}"
                        if key in all_medals.columns:
                            desired_cols.append(key)
                desired_cols = [c for c in desired_cols if c in all_medals.columns]
                all_medals = all_medals[desired_cols].sort_values(
                    desired_cols[-1] if len(desired_cols) > 2 else "n_athletes", ascending=False, na_position="last"
                )

                st.dataframe(
                    all_medals.style.format({c: "{:.2f}%" for c in all_medals.columns if c.startswith("P_at_least_one_medal_%_")}),
                    use_container_width=True,
                    height=520,
                )

                # Heatmap for a chosen stat
                heat_stat = st.selectbox("Heatmap stat", coh_medal_stats, index=0, key="heat_stat_med")
                heat_cols = [col for col in all_medals.columns if col.endswith(f"_{heat_stat}") and col.startswith("P_at_least_one_medal_%_")]
                if heat_cols:
                    heat_df = all_medals.set_index("Event Name")[heat_cols].copy()
                    heat_df.columns = [c.replace("P_at_least_one_medal_%_", "").replace(f"_{heat_stat}", "") for c in heat_cols]
                    st.subheader(f"Heatmap — P(≥1 medal) by method ({heat_stat})")

                    # Reduce density: show either a single event or top-N events (per method),
                    # depending on what the user selects.
                    heat_event_pick = st.selectbox(
                        "Heatmap event (optional)",
                        options=["(Use Top N selection)"] + sorted(all_medals["Event Name"].dropna().unique().tolist()),
                        index=0,
                        key="heat_event_pick_coh",
                    )
                    top_n_events = st.slider(
                        "Top N events per method (used when event is not selected)",
                        min_value=5,
                        max_value=50,
                        value=15,
                        step=5,
                        key="top_n_events_coh",
                    )

                    if heat_event_pick != "(Use Top N selection)":
                        heat_df = heat_df.loc[[heat_event_pick]]
                    else:
                        chosen: set[str] = set()
                        for method in heat_df.columns:
                            s = heat_df[method].dropna().sort_values(ascending=False)
                            chosen.update(s.head(top_n_events).index.tolist())
                        if chosen:
                            chosen_list = list(chosen)
                            heat_df = heat_df.loc[chosen_list]
                            # Sort rows by average across methods for readability
                            heat_df = heat_df.loc[
                                heat_df.mean(axis=1).sort_values(ascending=False).index.tolist()
                            ]

                    st.plotly_chart(
                        px.imshow(
                            heat_df,
                            aspect="auto",
                            text_auto=".0f",
                            title=f"Event × Method medal probability ({heat_stat})",
                            labels=dict(x="Method", y="Event", color="Probability (%)"),
                        ),
                        use_container_width=True,
                    )

    # ---------- Event Pool Monte Carlo ----------
    with tab_mc:
        st.subheader("Event pool-level P(≥1 medal) via Monte Carlo")
        st.caption("Uses athlete-level medal probabilities. For each event + stat, we take the top N athletes by that stat score, then simulate medal success independently.")

        if dff.empty:
            st.info("No rows after filters.")
        else:
            model_subset = MODELS
            mc_top_n = st.slider("Top N athletes per event (by stat score)", 10, 200, 100, step=5)
            mc_n_sims = st.slider("Monte Carlo simulations", 100, 2000, 500, step=50)
            mc_seed = st.number_input("Random seed", min_value=0, max_value=2_000_000_000, value=1234, step=1)
            mc_stats = st.multiselect("Pool stat(s) to compute", STATS, default=STATS)
            mc_methods = st.multiselect(
                "Methods to run (pool Monte Carlo)",
                options=MODELS,
                default=MODELS,
                help="Select any subset of the available methods; results are computed for every chosen method.",
            )
            if not mc_stats or not mc_methods:
                st.warning("Select at least one stat and one method.")
            else:
                mc_df = compute_event_pool_monte_carlo(
                    dff,
                    top_n=mc_top_n,
                    n_sims=mc_n_sims,
                    seed=int(mc_seed),
                    methods=tuple(mc_methods),
                    stats=tuple(mc_stats),
                )

                if mc_df.empty:
                    st.info("No Monte Carlo results (missing required columns).")
                else:
                    # Default table view
                    cols = ["Event Name"]
                    for stat in mc_stats:
                        for method in mc_methods:
                            c = f"P_at_least_one_medal_%_{method}_{stat}"
                            if c in mc_df.columns:
                                cols.append(c)
                    mc_view = mc_df[cols].copy()

                    st.dataframe(
                        mc_view.style.format({c: "{:.2f}%" for c in mc_view.columns if c.startswith("P_at_least_one_medal_%_")}),
                        use_container_width=True,
                        height=540,
                    )

                    st.divider()
                    heat_stat_mc = st.selectbox("Heatmap stat", mc_stats, index=0, key="mc_heat_stat")
                    mc_cols_stat = [f"P_at_least_one_medal_%_{m}_{heat_stat_mc}" for m in mc_methods]
                    mc_cols_stat = [c for c in mc_cols_stat if c in mc_df.columns]

                    if mc_cols_stat:
                        heat_mc = mc_df.set_index("Event Name")[mc_cols_stat].copy()
                        heat_mc.columns = [c.replace("P_at_least_one_medal_%_", "").replace(f"_{heat_stat_mc}", "") for c in mc_cols_stat]

                        heat_event_pick = st.selectbox(
                            "Heatmap event (optional)",
                            options=["(Use Top N selection)"] + sorted(mc_df["Event Name"].dropna().unique().tolist()),
                            index=0,
                            key="heat_event_pick_mc",
                        )
                        top_n_events_mc = st.slider(
                            "Top N events per method (used when event is not selected)",
                            min_value=5,
                            max_value=50,
                            value=15,
                            step=5,
                            key="top_n_events_mc",
                        )

                        if heat_event_pick != "(Use Top N selection)":
                            heat_mc = heat_mc.loc[[heat_event_pick]]
                        else:
                            chosen_mc: set[str] = set()
                            for method in heat_mc.columns:
                                s = heat_mc[method].dropna().sort_values(ascending=False)
                                chosen_mc.update(s.head(top_n_events_mc).index.tolist())
                            if chosen_mc:
                                chosen_list = list(chosen_mc)
                                heat_mc = heat_mc.loc[chosen_list]
                                heat_mc = heat_mc.loc[
                                    heat_mc.mean(axis=1).sort_values(ascending=False).index.tolist()
                                ]

                        st.plotly_chart(
                            px.imshow(
                                heat_mc,
                                aspect="auto",
                                text_auto=".0f",
                                title=f"Event × Method pool medal probability ({heat_stat_mc})",
                                labels=dict(x="Method", y="Event", color="Probability (%)"),
                            ),
                            use_container_width=True,
                        )

    # ---------- Quality ----------
    with tab_q:
        st.subheader("Cleaning flags")
        bad = dff[~dff["performance_flag_ok"]].copy() if len(dff) else pd.DataFrame()
        st.write("Range checks on **Mean**; combined events skipped; pole vault recategorized **Throw → Jump**.")
        if bad.empty:
            st.success("No flagged rows in current filter.")
        else:
            ex = (
                bad["cleaning_issues"]
                .str.split(";", expand=True)
                .stack()
                .replace("", pd.NA)
                .dropna()
            )
            if len(ex):
                ic = ex.value_counts().reset_index()
                ic.columns = ["issue", "count"]
                st.plotly_chart(px.bar(ic, x="issue", y="count"), use_container_width=True)
            st.dataframe(
                bad[["Athlete Name", "Event Name", "Category", "Mean", "cleaning_issues"]].head(300),
                use_container_width=True,
            )

    # ---------- Export ----------
    st.divider()
    with st.expander("Download filtered data"):
        buf = StringIO()
        dff.to_csv(buf, index=False)
        st.download_button(
            "Download filtered table as CSV",
            data=buf.getvalue(),
            file_name="talent_pool_filtered_assessment.csv",
            mime="text/csv",
        )
        st.caption("Includes all model columns; probabilities are numeric 0–100 in the file (same as source).")


main()

