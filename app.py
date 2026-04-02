"""
Talent pool dashboard: all models (LR, Bayes, RF, GB, MC), probabilities as %,
filters, comparisons, athletes, and exports.

Run: streamlit run app.py
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from clean_talent_pool import DEFAULT_SOURCE, PERCENT_COLS, load_and_clean

BASE = Path(__file__).resolve().parent
CLEANED_PATH = BASE / "data" / "talent_pool_cleaned.csv"

MODELS: list[str] = ["LR", "Bayes", "RF", "GB", "MC"]
STATS: list[str] = ["Mean", "Median", "Best"]
OUTCOMES: list[str] = ["Qualify", "Medal"]


def col_prob(model: str, stat: str, outcome: str) -> str:
    return f"{model}_{stat}_{outcome}"


def prob_columns_for(stat: str, outcome: str) -> list[str]:
    return [col_prob(m, stat, outcome) for m in MODELS]


def style_pct(df: pd.DataFrame):
    """Format model probability columns as percentages (values are 0–100)."""
    fmt: dict[str, str] = {}
    for c in PERCENT_COLS:
        if c in df.columns:
            fmt[c] = "{:.1f}%"
    sty = df.style.format(fmt, na_rep="—")
    return sty


def at_least_one_probability_pct(series: pd.Series) -> float:
    """
    P(at least one success) = 1 - prod_i (1 - p_i), with p_i in [0, 1].
    Input series values are percentages in 0–100 (benchmark / model outputs).
    Uses log-space for numerical stability with many athletes.
    """
    p = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    p = p[np.isfinite(p)]
    p = np.clip(p / 100.0, 0.0, 1.0)
    if p.size == 0:
        return float("nan")
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    log_prod_fail = float(np.sum(np.log(1.0 - p)))
    if log_prod_fail < -700.0:
        return 100.0
    return float((1.0 - np.exp(log_prod_fail)) * 100.0)


def cohort_at_least_one_table(
    work: pd.DataFrame,
    col_qualify: str,
    col_medal: str,
    dedupe_athlete: bool,
) -> pd.DataFrame:
    """
    One row per Event Name: n athletes in cohort, P(≥1 qualify), P(≥1 medal).
    """
    w = work.copy()
    if dedupe_athlete and "Athlete ID" in w.columns and "Event Name" in w.columns:
        w = w.drop_duplicates(subset=["Event Name", "Athlete ID"], keep="first")
    if "Event Name" not in w.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for ev, g in w.groupby("Event Name", sort=True):
        n = int(g["Athlete ID"].nunique()) if "Athlete ID" in g.columns else len(g)
        pq = at_least_one_probability_pct(g[col_qualify]) if col_qualify in g.columns else float("nan")
        pm = at_least_one_probability_pct(g[col_medal]) if col_medal in g.columns else float("nan")
        rows.append(
            {
                "Event Name": ev,
                "n_athletes": n,
                "P_at_least_one_qualifies_%": pq,
                "P_at_least_one_medal_%": pm,
            }
        )
    return pd.DataFrame(rows)


def merge_cohort_tags(df: pd.DataFrame, tags: pd.DataFrame | None) -> pd.DataFrame:
    """Left-merge tag columns onto df by Athlete ID. Duplicate Athlete IDs in tags: last row wins."""
    if tags is None or tags.empty or "Athlete ID" not in tags.columns:
        return df
    t = tags.drop_duplicates(subset=["Athlete ID"], keep="last").copy()
    extra = [c for c in t.columns if c != "Athlete ID"]
    if not extra:
        return df
    return df.merge(t[["Athlete ID"] + extra], on="Athlete ID", how="left")


def filter_dataframe(
    df: pd.DataFrame,
    *,
    categories: list[str] | None = None,
    genders: list[str] | None = None,
    events: list[str] | None = None,
    only_valid: bool = False,
    age_min: float | None = None,
    age_max: float | None = None,
    include_unknown_age: bool = True,
    mean_min: float | None = None,
    mean_max: float | None = None,
    include_unknown_mean: bool = True,
    prob_col: str | None = None,
    prob_min: float | None = None,
    prob_max: float | None = None,
    include_unknown_prob: bool = True,
    event_substring: str = "",
    athlete_substring: str = "",
    competition_scope: list[str] | None = None,
    championship: list[str] | None = None,
    include_untagged_scope: bool = True,
    include_untagged_championship: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    if categories:
        out = out[out["Category"].isin(categories)]
    if genders:
        out = out[out["Gender"].isin(genders)]
    if events:
        out = out[out["Event Name"].isin(events)]
    if only_valid and "performance_flag_ok" in out.columns:
        out = out[out["performance_flag_ok"]]

    if age_min is not None and age_max is not None and "Age" in out.columns:
        a = pd.to_numeric(out["Age"], errors="coerce")
        mask = (a >= age_min) & (a <= age_max)
        if include_unknown_age:
            mask = mask | a.isna()
        out = out[mask]

    if (mean_min is not None or mean_max is not None) and "Mean" in out.columns:
        m = pd.to_numeric(out["Mean"], errors="coerce")
        mask = pd.Series(True, index=out.index)
        if mean_min is not None:
            mask &= m >= mean_min
        if mean_max is not None:
            mask &= m <= mean_max
        if include_unknown_mean:
            mask = mask | m.isna()
        out = out[mask]

    if prob_col and prob_col in out.columns and (prob_min is not None or prob_max is not None):
        p = pd.to_numeric(out[prob_col], errors="coerce")
        mask = pd.Series(True, index=out.index)
        if prob_min is not None:
            mask &= p >= prob_min
        if prob_max is not None:
            mask &= p <= prob_max
        if include_unknown_prob:
            mask = mask | p.isna()
        out = out[mask]

    es = (event_substring or "").strip()
    if es and "Event Name" in out.columns:
        out = out[out["Event Name"].astype(str).str.contains(es, case=False, na=False)]

    ans = (athlete_substring or "").strip()
    if ans and "Athlete Name" in out.columns:
        out = out[out["Athlete Name"].astype(str).str.contains(ans, case=False, na=False)]

    if competition_scope and "competition_scope" in out.columns:
        cs = out["competition_scope"].astype(str).str.strip()
        untagged = cs.isna() | (cs == "") | (cs.str.lower() == "nan")
        m = out["competition_scope"].isin(competition_scope)
        if include_untagged_scope:
            m = m | untagged
        out = out[m]

    if championship and "championship" in out.columns:
        ch = out["championship"].astype(str).str.strip()
        untagged = ch.isna() | (ch == "") | (ch.str.lower() == "nan")
        m = out["championship"].isin(championship)
        if include_untagged_championship:
            m = m | untagged
        out = out[m]

    return out


@st.cache_data
def load_cleaned_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_recleaned(path: str) -> pd.DataFrame:
    return load_and_clean(Path(path))


def main():
    st.set_page_config(
        page_title="Talent pool dashboard",
        page_icon="",
        layout="wide",
    )
    st.title("Talent pool analytics")
    st.caption(
        "All methods: **LR**, **Bayes**, **RF**, **GB**, **MC** — each with Mean / Median / Best × Qualify & Medal (all shown as **%**)."
    )

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
        if st.button("Save fresh clean to disk"):
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

        with st.expander("Model probability band", expanded=False):
            f_mod = st.selectbox("Model", MODELS, index=1, key="f_mod")
            f_stat = st.selectbox("Statistic", STATS, index=0, key="f_stat")
            f_out = st.selectbox("Outcome", OUTCOMES, index=0, key="f_out")
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

        st.metric("Filtered rows", f"{len(dff):,}", delta=f"{len(dff) - len(df):,}" if len(dff) != len(df) else None)
        st.caption(f"Total in file (after tag merge): {len(df):,}")

    tab_ov, tab_mod, tab_ev, tab_ath, tab_lb, tab_cohort, tab_q = st.tabs(
        [
            "Overview",
            "Models",
            "Events",
            "Athletes",
            "Leaderboards",
            "Cohort ≥1",
            "Data quality",
        ]
    )

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
                st.plotly_chart(
                    px.bar(cc, x="Category", y="n", title="Rows by category"),
                    use_container_width=True,
                )
            with r2:
                gg = dff["Gender"].value_counts().reset_index()
                gg.columns = ["Gender", "n"]
                st.plotly_chart(
                    px.pie(gg, names="Gender", values="n", title="Rows by gender"),
                    use_container_width=True,
                )

        st.subheader("Model comparison — Mean × Qualify (all five methods)")
        q_mean = [c for c in prob_columns_for("Mean", "Qualify") if c in dff.columns]
        if q_mean and not dff.empty:
            melt_q = dff[q_mean + ["Athlete Name", "Event Name"]].melt(
                id_vars=["Athlete Name", "Event Name"],
                value_vars=q_mean,
                var_name="Model",
                value_name="pct",
            )
            melt_q["Model"] = melt_q["Model"].str.replace("_Mean_Qualify", "")
            fig_box = px.box(
                melt_q,
                x="Model",
                y="pct",
                color="Model",
                points=False,
                title="Distribution of P(qualify | mean) by model",
            )
            fig_box.update_layout(showlegend=False, yaxis_title="Probability (%)")
            st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Model comparison — Mean × Medal")
        m_mean = [c for c in prob_columns_for("Mean", "Medal") if c in dff.columns]
        if m_mean and not dff.empty:
            melt_m = dff[m_mean + ["Athlete Name", "Event Name"]].melt(
                id_vars=["Athlete Name", "Event Name"],
                value_vars=m_mean,
                var_name="Model",
                value_name="pct",
            )
            melt_m["Model"] = melt_m["Model"].str.replace("_Mean_Medal", "")
            fig_bm = px.box(
                melt_m,
                x="Model",
                y="pct",
                color="Model",
                points=False,
                title="Distribution of P(medal | mean) by model",
            )
            fig_bm.update_layout(showlegend=False, yaxis_title="Probability (%)")
            st.plotly_chart(fig_bm, use_container_width=True)

        # Small multiples: histogram per model
        if q_mean and not dff.empty:
            st.subheader("Histograms — Mean qualify % (per model)")
            fig_hist = make_subplots(rows=2, cols=3, subplot_titles=q_mean[:6])
            for i, c in enumerate(q_mean[:6]):
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

    # ---------- Models ----------
    with tab_mod:
        c1, c2, c3 = st.columns(3)
        with c1:
            stat_pick = st.selectbox("Statistic", STATS, index=0, key="m_stat")
        with c2:
            out_pick = st.selectbox("Outcome", OUTCOMES, index=0, key="m_out")
        with c3:
            sample_cap = st.slider("Max points for scatter / matrix", 1000, 8000, 4000)

        cols_sel = [c for c in prob_columns_for(stat_pick, out_pick) if c in dff.columns]
        if not cols_sel:
            st.warning("No columns for this combination.")
        else:
            sub = dff[cols_sel].dropna()
            if len(sub) > sample_cap:
                sub = sub.sample(sample_cap, random_state=42)

            st.subheader("Correlation matrix (selected statistic & outcome)")
            if len(sub) > 2:
                corr = sub.corr()
                st.plotly_chart(
                    px.imshow(
                        corr,
                        text_auto=".2f",
                        title=f"{stat_pick} × {out_pick}: correlation across models",
                        color_continuous_scale="RdBu",
                        zmin=-1,
                        zmax=1,
                    ),
                    use_container_width=True,
                )

            st.subheader("Pairwise: model vs model")
            pc1, pc2 = st.columns(2)
            with pc1:
                ma = st.selectbox("Model (X)", MODELS, index=1, key="px")
            with pc2:
                mb = st.selectbox("Model (Y)", MODELS, index=3, key="py")
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
            mod_vm = st.selectbox("Model", MODELS, index=1, key="vmod")
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
        ev_list = sorted(dff["Event Name"].dropna().unique().tolist())
        if not ev_list:
            st.info("No rows after filters.")
        else:
            pick = st.selectbox("Event", ev_list, key="ev_pick")
            sort_mod = st.selectbox("Sort table by", MODELS, index=1, key="ev_sort")
            sort_col = col_prob(sort_mod, "Mean", "Qualify")
            sub = dff[dff["Event Name"] == pick].copy()
            base_cols = [
                "Athlete Name",
                "Gender",
                "Age",
                "Category",
                "Mean",
                "Median",
                "Best Score",
                "Qualification Standard",
                "Medal Standard",
                "performance_flag_ok",
            ]
            base_cols = [c for c in base_cols if c in sub.columns]
            prob_cols = [c for c in PERCENT_COLS if c in sub.columns]
            ordered = base_cols + prob_cols
            if sort_col in sub.columns:
                sub = sub.sort_values(sort_col, ascending=False, na_position="last")
            st.write(f"**{len(sub)}** rows — all model probabilities as **%**")
            st.dataframe(
                style_pct(sub[ordered]),
                use_container_width=True,
                height=min(520, 120 + 36 * len(sub)),
            )

            if "Mean" in sub.columns and sort_col in sub.columns:
                fig_ev = px.scatter(
                    sub,
                    x="Mean",
                    y=sort_col,
                    color="Gender",
                    hover_data=["Athlete Name"],
                    labels={sort_col: f"{sort_mod} Mean Qualify (%)"},
                    title=f"{pick}: performance vs {sort_mod} P(qualify | mean)",
                )
                fig_ev.update_yaxes(title=f"{sort_mod} qualify %")
                st.plotly_chart(fig_ev, use_container_width=True)

    # ---------- Athletes ----------
    with tab_ath:
        names = sorted(dff["Athlete Name"].dropna().unique().tolist())
        name = st.selectbox("Athlete", names, key="ath_name")
        if name:
            ap = dff[dff["Athlete Name"] == name].copy()
            st.write(f"**{len(ap)}** event rows")
            base_a = [
                "Event Name",
                "Category",
                "Gender",
                "Age",
                "Mean",
                "Best Score",
                "Qualification Standard",
                "Medal Standard",
                "performance_flag_ok",
            ]
            base_a = [c for c in base_a if c in ap.columns]
            prob_a = [c for c in PERCENT_COLS if c in ap.columns]
            st.dataframe(style_pct(ap[base_a + prob_a]), use_container_width=True, height=420)

            # Heatmap: events × model (Mean Qualify)
            hq = [col_prob(m, "Mean", "Qualify") for m in MODELS if col_prob(m, "Mean", "Qualify") in ap.columns]
            if hq and "Event Name" in ap.columns:
                heat = ap.set_index("Event Name")[hq]
                heat.columns = [c.replace("_Mean_Qualify", "") for c in hq]
                st.subheader("Mean × Qualify % — models vs event")
                st.plotly_chart(
                    px.imshow(
                        heat,
                        text_auto=".0f",
                        aspect="auto",
                        title=f"{name}: P(qualify|mean) by model and event",
                        labels=dict(x="Model", y="Event", color="%"),
                    ),
                    use_container_width=True,
                )

    # ---------- Leaderboards ----------
    with tab_lb:
        ev_lb = st.selectbox("Event", sorted(dff["Event Name"].dropna().unique()), key="lb_ev")
        stat_lb = st.selectbox("Statistic", STATS, key="lb_st")
        out_lb = st.selectbox("Outcome", OUTCOMES, key="lb_out")
        top_n = st.slider("Top N", 5, 40, 15)
        rank_mod = st.selectbox("Rank by model", MODELS, index=1, key="lb_rank")
        rank_col = col_prob(rank_mod, stat_lb, out_lb)

        pool = dff[dff["Event Name"] == ev_lb].copy()
        if rank_col not in pool.columns:
            st.error("Column missing.")
        else:
            pool = pool.dropna(subset=[rank_col])
            top = pool.nlargest(top_n, rank_col, keep="all").head(top_n)
            show = ["Athlete Name", "Gender", "Age", "Mean", "Best Score", rank_col]
            for m in MODELS:
                c = col_prob(m, stat_lb, out_lb)
                if c in top.columns and c not in show:
                    show.append(c)
            show = [c for c in show if c in top.columns]
            st.markdown(f"**Top {top_n}** for **{ev_lb}** — ranked by **{rank_mod}** ({stat_lb} × {out_lb})")
            st.dataframe(
                style_pct(top[show]),
                use_container_width=True,
            )

            # Bar: all models for #1 row
            if len(top) >= 1:
                row0 = top.iloc[0]
                bar_x = []
                bar_y = []
                for m in MODELS:
                    c = col_prob(m, stat_lb, out_lb)
                    if c in row0.index and pd.notna(row0[c]):
                        bar_x.append(m)
                        bar_y.append(float(row0[c]))
                if bar_y:
                    st.plotly_chart(
                        go.Figure(
                            data=[go.Bar(x=bar_x, y=bar_y, text=[f"{v:.1f}%" for v in bar_y], textposition="auto")],
                            layout=dict(
                                title=f"#1 ({row0.get('Athlete Name', '')}): all models — {stat_lb} {out_lb} %",
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
            (same percentages as elsewhere, 0–100):

            - **P(≥1 qualifies)** = **1 − Π<sub>i</sub> (1 − q<sub>i</sub>)**
            - **P(≥1 wins medal)** = **1 − Π<sub>i</sub> (1 − m<sub>i</sub>)**

            where **q<sub>i</sub>**, **m<sub>i</sub>** are each athlete’s qualify / medal probabilities on the **0–1** scale.
            Rows with a **missing** value in that column are **omitted** from that product.

            Assumes **independence** between athletes (complement of “everyone fails”).
            """
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            coh_model = st.selectbox("Model (benchmark)", MODELS, index=1, key="coh_mod")
        with c2:
            coh_stat = st.selectbox("Statistic", STATS, index=0, key="coh_st")
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
                cohort_df = cohort_df.sort_values(
                    "P_at_least_one_qualifies_%", ascending=False, na_position="last"
                ).reset_index(drop=True)

                st.caption(f"Benchmark columns: **{cq}**, **{cm}** (values interpreted as %).")
                if "Athlete ID" in dff.columns:
                    n_dup = len(dff) - len(dff.drop_duplicates(subset=["Event Name", "Athlete ID"], keep="first"))
                    if n_dup > 0 and dedupe:
                        st.caption(f"Duplicate Athlete ID × Event rows removed before product: **{n_dup}**.")
                    elif n_dup > 0 and not dedupe:
                        st.warning(
                            f"**{n_dup}** duplicate Athlete ID × Event rows: each row is included in the product "
                            "(may not match “one probability per athlete”). Turn on **One row per athlete per event** to dedupe."
                        )

                show_c = cohort_df[
                    [
                        "Event Name",
                        "n_athletes",
                        "P_at_least_one_qualifies_%",
                        "P_at_least_one_medal_%",
                    ]
                ]
                sty_c = show_c.style.format(
                    {
                        "P_at_least_one_qualifies_%": "{:.2f}%",
                        "P_at_least_one_medal_%": "{:.2f}%",
                    },
                    na_rep="—",
                )
                st.dataframe(sty_c, use_container_width=True, height=min(560, 140 + 28 * len(show_c)))

                b1, b2 = st.columns(2)
                with b1:
                    fig_c1 = px.bar(
                        cohort_df.head(40),
                        x="Event Name",
                        y="P_at_least_one_qualifies_%",
                        title="P(≥1 athlete qualifies) by event (top 40 by value)",
                    )
                    fig_c1.update_layout(xaxis_tickangle=-45, yaxis_title="%")
                    st.plotly_chart(fig_c1, use_container_width=True)
                with b2:
                    fig_c2 = px.bar(
                        cohort_df.sort_values("P_at_least_one_medal_%", ascending=False).head(40),
                        x="Event Name",
                        y="P_at_least_one_medal_%",
                        title="P(≥1 athlete medals) by event (top 40)",
                    )
                    fig_c2.update_layout(xaxis_tickangle=-45, yaxis_title="%")
                    st.plotly_chart(fig_c2, use_container_width=True)

                sc = StringIO()
                cohort_df.to_csv(sc, index=False)
                st.download_button(
                    "Download cohort table (CSV)",
                    data=sc.getvalue(),
                    file_name="cohort_at_least_one.csv",
                    mime="text/csv",
                    key="dl_cohort",
                )

    # ---------- Quality ----------
    with tab_q:
        st.subheader("Cleaning flags")
        bad = dff[~dff["performance_flag_ok"]].copy() if len(dff) else pd.DataFrame()
        st.write(
            "Range checks on **Mean**; combined events skipped; pole vault recategorized **Throw → Jump**."
        )
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
            st.dataframe(bad[["Athlete Name", "Event Name", "Category", "Mean", "cleaning_issues"]].head(300), use_container_width=True)

    # ---------- Export ----------
    st.divider()
    with st.expander("Download filtered data"):
        buf = StringIO()
        dff.to_csv(buf, index=False)
        st.download_button(
            "Download filtered table as CSV",
            data=buf.getvalue(),
            file_name="talent_pool_filtered.csv",
            mime="text/csv",
        )
        st.caption("Includes all model columns; probabilities are numeric 0–100 in the file (same as source).")


if __name__ == "__main__":
    main()
