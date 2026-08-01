"""
Growth Decision Brief — UAE Food
A plan-vs-actual console that explains the revenue gap and recommends the actions.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from growth_engine import (
    pack_to_json,
    FACTORS, FACTOR_LABELS, build_evidence_pack, cohort_matrix, comparison_table,
    funnel, load_data, metrics, month_options, overview_table, revenue_bridge,
    weekly_breakdown, weeks_in_range,
)
from brief import MODELS, SYSTEM_PROMPT, generate_brief

INK, PAPER, SURFACE = "#0E2B2B", "#FAF8F4", "#FFFFFF"
LINE, MUTED = "#DDE2DA", "#5F7370"
JADE, AMBER, SAND = "#1C7C6B", "#B4531B", "#E9E3D6"

st.set_page_config(page_title="Growth Decision Brief — UAE Food",
                   page_icon="◆", layout="wide")

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
  .stApp {{ background: {PAPER}; }}
  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {INK}; }}
  h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif !important;
                    letter-spacing: -0.02em; color: {INK}; }}
  .masthead {{ border-bottom: 2px solid {INK}; padding-bottom: .6rem; margin-bottom: 1.4rem; }}
  .eyebrow {{ font-family: 'IBM Plex Mono', monospace; font-size: .68rem;
              letter-spacing: .16em; text-transform: uppercase; color: {MUTED}; }}
  .masthead h1 {{ font-size: 1.9rem; margin: .15rem 0 0 0; }}
  .card {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 3px;
           padding: .85rem 1rem; height: 100%; }}
  .card .label {{ font-family: 'IBM Plex Mono', monospace; font-size: .64rem;
                  letter-spacing: .12em; text-transform: uppercase; color: {MUTED}; }}
  .card .value {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem;
                  font-weight: 600; line-height: 1.25; margin-top: .2rem; }}
  .card .delta {{ font-family: 'IBM Plex Mono', monospace; font-size: .76rem;
                  font-weight: 500; margin-top: .15rem; }}
  .up {{ color: {JADE}; }} .down {{ color: {AMBER}; }}
  .note {{ font-size: .8rem; color: {MUTED}; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid {LINE};
      flex-wrap: wrap; }}
  .stTabs [data-baseweb="tab"] {{ font-family: 'IBM Plex Mono', monospace;
      font-size: .72rem; letter-spacing: .09em; text-transform: uppercase;
      padding: 0 0 .5rem 0; }}
  section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {LINE}; }}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", size=12, color=INK),
    margin=dict(l=10, r=10, t=40, b=10), hoverlabel=dict(font_family="IBM Plex Mono"),
)


def aed(x, digits=2):
    a = abs(x)
    if a >= 1e6:
        return f"AED {x/1e6:,.{digits}f}M"
    if a >= 1e3:
        return f"AED {x/1e3:,.0f}k"
    return f"AED {x:,.0f}"


def kpi(label, value, delta=None, good=None):
    cls = "" if good is None else ("up" if good else "down")
    d = f'<div class="delta {cls}">{delta}</div>' if delta else ""
    return (f'<div class="card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>{d}</div>')


BASE_DIR = Path(__file__).resolve().parent
WEEKLY_CSV = BASE_DIR / "data" / "uae_food_weekly.csv"
COHORT_CSV = BASE_DIR / "data" / "uae_food_cohorts.csv"


@st.cache_data
def get_data():
    """Load the dataset, rebuilding it from the seeded generator if it is absent."""
    if not (WEEKLY_CSV.exists() and COHORT_CSV.exists()):
        cwd = os.getcwd()
        try:
            os.chdir(BASE_DIR)
            runpy.run_path(str(BASE_DIR / "generate_data.py"), run_name="__generated__")
        finally:
            os.chdir(cwd)
    return load_data(str(WEEKLY_CSV), str(COHORT_CSV))


def get_token() -> str:
    try:
        if "HF_TOKEN" in st.secrets:
            return str(st.secrets["HF_TOKEN"])
    except Exception:
        pass
    return os.environ.get("HF_TOKEN", "")


weekly, cohorts = get_data()
MONTHS = month_options(weekly)

st.markdown('<div class="masthead"><div class="eyebrow">Careem Food · UAE · '
            'synthetic data</div><h1>Growth Decision Brief</h1></div>',
            unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown('<div class="eyebrow">Market</div>', unsafe_allow_html=True)
    market = st.selectbox("City", ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "All UAE"],
                          label_visibility="collapsed")

    st.markdown('<div class="eyebrow">Chosen period</div>', unsafe_allow_html=True)
    cur_range = st.select_slider(
        "Chosen period", options=MONTHS,
        value=(MONTHS[-2], MONTHS[-1]), label_visibility="collapsed")

    st.markdown('<div class="eyebrow">Comparison period</div>', unsafe_allow_html=True)
    prior_default = (MONTHS[max(len(MONTHS) - 4, 0)], MONTHS[max(len(MONTHS) - 3, 0)])
    prior_range = st.select_slider(
        "Comparison period", options=MONTHS,
        value=prior_default, label_visibility="collapsed")

    st.divider()
    st.markdown('<div class="eyebrow">Recommendation engine</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Open-weights model", list(MODELS))
    st.caption("Inference key detected." if get_token()
               else "No inference key set — the rule-based writer runs instead.")
    st.divider()
    st.download_button("Download the dataset (CSV)",
                       weekly.to_csv(index=False), "uae_food_weekly.csv", "text/csv")

cur_weeks = weeks_in_range(weekly, *cur_range)
prior_weeks = weeks_in_range(weekly, *prior_range)
CUR_LABEL = cur_range[0] if cur_range[0] == cur_range[1] else f"{cur_range[0]} – {cur_range[1]}"
PRIOR_LABEL = prior_range[0] if prior_range[0] == prior_range[1] else f"{prior_range[0]} – {prior_range[1]}"

scope = weekly if market == "All UAE" else weekly[weekly.city == market]
cur = scope[scope.week_start.isin(cur_weeks)]
prior = scope[scope.week_start.isin(prior_weeks)]

b = revenue_bridge(cur)
a = b["agg"]

tabs = st.tabs(["Start here", "Overview", "Pulse", "Why", "Funnel & cohorts",
                "Decision brief", "How it works"])

# --------------------------------------------------------------------------- #
# 1 — Start here
# --------------------------------------------------------------------------- #
with tabs[0]:
    st.markdown("""
#### What this tool does

It answers the two questions a growth review actually turns on: **are we on plan, why
not, and what do we do about it?** Pick a market and two time windows on the left, then
work left to right across the tabs — each one narrows the question.

#### The controls on the left

| Control | What it does |
|---|---|
| **City** | Sets the market for every tab except Overview, which always shows all cities |
| **Chosen period** | The window under review. Drag either end to select a month range |
| **Comparison period** | The benchmark window. Every "vs comparison" figure in Pulse and Funnel & cohorts measures against this |
| **Open-weights model** | Which model writes the recommendations on the Decision brief tab |

Both period sliders read their options from the dataset, so they always cover exactly
the months available.

#### The tabs

| Tab | Question it answers |
|---|---|
| **Overview** | How does every city compare this period, and where is the money? |
| **Pulse** | For the chosen city, what happened week by week, and how does it compare to the benchmark window? |
| **Why** | Which factor caused the revenue gap, sized in AED? |
| **Funnel & cohorts** | Where does traffic leak, and is the quality of new users holding? |
| **Decision brief** | What should we actually do, with owners and guardrails? |
| **How it works** | The method, and the editable prompt behind the recommendations |

#### How to read a session

Start on **Overview** to find the city that is furthest from plan. Switch the sidebar to
that city and open **Why** to see which of the five factors caused it. Confirm the story
on **Funnel & cohorts** — a traffic lift alongside a conversion drop means demand is
arriving and leaking. Then generate the recommendations.
""")

# --------------------------------------------------------------------------- #
# 2 — Overview
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.markdown(f'<div class="eyebrow">All cities · {CUR_LABEL}</div>',
                unsafe_allow_html=True)
    ov = overview_table(weekly, cur_weeks)
    st.dataframe(
        ov.style.format({
            "Sessions (avg/wk)": "{:,.0f}", "MAU (avg/wk)": "{:,.0f}",
            "Orders": "{:,.0f}", "OPU": "{:.3f}", "AOV (AED)": "{:,.1f}",
            "Take rate": "{:.1%}", "Net revenue (AED)": "{:,.0f}",
            "Plan (AED)": "{:,.0f}", "Attainment": "{:.1%}", "Gap (AED)": "{:+,.0f}"}),
        hide_index=True, width="stretch")
    st.markdown('<div class="note">The final row aggregates all four cities. '
                '“Largest drag” is the factor contributing most negatively to that '
                'city’s revenue gap.</div>', unsafe_allow_html=True)

    st.write("")
    cities = ov[ov.City != "All UAE"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cities.City, y=cities["Net revenue (AED)"] / 1e6,
                         name="Actual", marker_color=INK))
    fig.add_trace(go.Bar(x=cities.City, y=cities["Plan (AED)"] / 1e6,
                         name="Plan", marker_color=SAND))
    fig.update_layout(**PLOT_LAYOUT, height=330, barmode="group",
                      title="Net revenue vs plan by city, AED millions",
                      legend=dict(orientation="h", y=1.14, x=0))
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# 3 — Pulse
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.markdown(f'<div class="eyebrow">{market} · {CUR_LABEL}</div>',
                unsafe_allow_html=True)
    cols = st.columns(6)
    cards = [
        ("Net revenue", aed(a["rev_a"]),
         f"{b['attainment']*100:.1f}% of plan · {aed(b['total_gap'])}", b["total_gap"] >= 0),
        ("Sessions (avg/wk)", f"{a['sessions_a']:,.0f}",
         f"{(a['sessions_a']/a['sessions_p']-1)*100:+.1f}% vs plan",
         a["sessions_a"] >= a["sessions_p"]),
        ("Traffic → user rate", f"{a['user_rate_a']*100:.1f}%",
         f"{(a['user_rate_a']-a['user_rate_p'])*100:+.2f}pp vs plan",
         a["user_rate_a"] >= a["user_rate_p"]),
        ("OPU (orders/user/wk)", f"{a['opu_a']:.3f}",
         f"{(a['opu_a']/a['opu_p']-1)*100:+.1f}% vs plan", a["opu_a"] >= a["opu_p"]),
        ("AOV", f"{a['aov_a']:,.1f}",
         f"{(a['aov_a']/a['aov_p']-1)*100:+.1f}% vs plan", a["aov_a"] >= a["aov_p"]),
        ("Take rate", f"{a['take_a']*100:.1f}%",
         f"{(a['take_a']-a['take_p'])*100:+.2f}pp vs plan", a["take_a"] >= a["take_p"]),
    ]
    for col, (l, v, d, g) in zip(cols, cards):
        col.markdown(kpi(l, v, d, g), unsafe_allow_html=True)

    st.write("")
    trend = (cur.groupby("week_start")[["net_revenue_actual_aed", "net_revenue_plan_aed"]]
             .sum().reset_index())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend.week_start, y=trend.net_revenue_plan_aed / 1e6,
                             name="Plan", line=dict(color=MUTED, width=1.6, dash="dot")))
    fig.add_trace(go.Scatter(x=trend.week_start, y=trend.net_revenue_actual_aed / 1e6,
                             name="Actual", line=dict(color=INK, width=2.6),
                             mode="lines+markers"))
    fig.update_layout(**PLOT_LAYOUT, height=320,
                      title=f"Weekly net revenue, AED millions · {market}",
                      legend=dict(orientation="h", y=1.14, x=0))
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    fig.update_xaxes(gridcolor=LINE)
    st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="eyebrow">Weekly breakdown</div>', unsafe_allow_html=True)
    wb = weekly_breakdown(cur)
    st.dataframe(
        wb.style.format({
            "Sessions": "{:,.0f}", "MAU": "{:,.0f}", "Orders": "{:,.0f}",
            "OPU": "{:.3f}", "AOV (AED)": "{:,.1f}", "Take rate": "{:.1%}",
            "Net revenue (AED)": "{:,.0f}", "Plan (AED)": "{:,.0f}",
            "Attainment": "{:.1%}"}),
        hide_index=True, width="stretch")

    st.write("")
    st.markdown('<div class="eyebrow">Chosen vs comparison period</div>',
                unsafe_allow_html=True)
    ct = comparison_table(cur, prior, CUR_LABEL, PRIOR_LABEL)
    if ct.empty:
        st.info("Select a comparison period with data to see this table.")
    else:
        st.dataframe(ct, hide_index=True, width="stretch")
        st.markdown('<div class="note">Rate metrics change in percentage points; '
                    'volume and value metrics change in percent.</div>',
                    unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# 4 — Why
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.markdown(f"#### Where the {aed(b['total_gap'])} gap comes from")
    st.markdown('<div class="note">Net revenue = sessions × traffic-to-user rate × OPU '
                '× AOV × take rate. Each bar is that factor’s contribution to the plan '
                'variance, chain-linked so the five bars reconcile exactly to the total '
                'gap.</div>', unsafe_allow_html=True)
    st.write("")

    c = b["contributions"]
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * 5 + ["total"],
        x=["Plan"] + [FACTOR_LABELS[k] for k in FACTORS] + ["Actual"],
        y=[a["rev_p"] / 1e6] + [c[k] / 1e6 for k in FACTORS] + [None],
        text=[aed(a["rev_p"])] + [aed(c[k]) for k in FACTORS] + [aed(a["rev_a"])],
        textposition="outside",
        connector=dict(line=dict(color=LINE, width=1)),
        increasing=dict(marker=dict(color=JADE)),
        decreasing=dict(marker=dict(color=AMBER)),
        totals=dict(marker=dict(color=INK)),
    ))
    fig.update_layout(**PLOT_LAYOUT, height=440, showlegend=False,
                      yaxis_title="AED millions")
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="eyebrow">Reads as</div>', unsafe_allow_html=True)
        drag, lift = b["largest_drag"], b["largest_lift"]
        st.markdown(
            f"**{FACTOR_LABELS[drag]}** is the largest drag at **{aed(c[drag])}**. "
            f"**{FACTOR_LABELS[lift]}** is the largest lift at **{aed(c[lift])}**. "
            f"A gap concentrated in traffic is a top-of-funnel problem; one in the "
            f"traffic-to-user rate or OPU means demand is arriving and leaking; one in "
            f"take rate is unit economics. They need different owners and different "
            f"budgets.")
    with right:
        st.markdown('<div class="eyebrow">Factor detail</div>', unsafe_allow_html=True)
        detail = pd.DataFrame([
            [FACTOR_LABELS["sessions"], a["sessions_a"], a["sessions_p"], c["sessions"]],
            [FACTOR_LABELS["user_rate"], a["user_rate_a"], a["user_rate_p"], c["user_rate"]],
            [FACTOR_LABELS["opu"], a["opu_a"], a["opu_p"], c["opu"]],
            [FACTOR_LABELS["aov"], a["aov_a"], a["aov_p"], c["aov"]],
            [FACTOR_LABELS["take"], a["take_a"], a["take_p"], c["take"]],
        ], columns=["Factor", "Actual", "Plan", "AED impact"])
        detail["vs plan"] = detail.Actual / detail.Plan - 1
        st.dataframe(detail.style.format(
            {"Actual": "{:,.3f}", "Plan": "{:,.3f}", "AED impact": "{:+,.0f}",
             "vs plan": "{:+.1%}"}), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# 5 — Funnel & cohorts
# --------------------------------------------------------------------------- #
with tabs[4]:
    left, right = st.columns([1, 1])
    with left:
        st.markdown(f'<div class="eyebrow">Conversion funnel · {CUR_LABEL} vs '
                    f'{PRIOR_LABEL}</div>', unsafe_allow_html=True)
        f = funnel(cur, prior)
        show = pd.DataFrame({
            "Step": f["Step"],
            "Volume": f["Volume"].map("{:,.0f}".format),
            "Step conversion": f["Step conversion"].map(
                lambda v: "—" if pd.isna(v) else f"{v:.1%}"),
        })
        if "Change" in f.columns:
            show["Change"] = [
                "—" if pd.isna(v) else f"{v:+.1f} {u}"
                for v, u in zip(f["Change"], f["Change unit"])]
        st.dataframe(show, hide_index=True, width="stretch")
        st.markdown('<div class="note">Step conversion is measured against the previous '
                    'stage, so a fall in one row localises the leak. Total traffic has no '
                    'prior step; its change is volume growth vs the comparison '
                    'period.</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="eyebrow">Retention by cohort</div>',
                    unsafe_allow_html=True)
        cm = cohort_matrix(cohorts, market) * 100
        fig = go.Figure(go.Heatmap(
            z=cm.values, x=[f"M{col}" for col in cm.columns], y=cm.index,
            colorscale=[[0, PAPER], [0.5, "#8FBDAF"], [1, INK]],
            text=cm.round(1).values, texttemplate="%{text}%",
            showscale=False, hovertemplate="%{y} · %{x}: %{z:.1f}%<extra></extra>"))
        fig.update_layout(**PLOT_LAYOUT, height=310)
        fig.update_xaxes(side="top", showgrid=False)
        fig.update_yaxes(autorange="reversed", title="Cohort")
        st.plotly_chart(fig, width="stretch")
        st.markdown('<div class="note">Read down a column: if later cohorts retain worse '
                    'at the same age, acquisition quality is falling. Cohorts span the '
                    'full dataset, not the chosen period.</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown(f'<div class="eyebrow">Acquisition cost and service quality · vs '
                f'{PRIOR_LABEL}</div>', unsafe_allow_html=True)
    m, mp = metrics(cur), (metrics(prior) if len(prior) else {})
    cols = st.columns(5)
    items = [
        ("Promo per new user", f"AED {m['cpa']:,.1f}", "cpa", False),
        ("Promo per repeat user/wk", f"AED {m['promo_per_repeat']:,.2f}",
         "promo_per_repeat", False),
        ("Promo as % of GMV", f"{m['promo_pct_gmv']*100:.1f}%", "promo_pct_gmv", False),
        ("Avg delivery time", f"{m['eta']:,.1f} min", "eta", False),
        ("Cancellation rate", f"{m['cancel']*100:.2f}%", "cancel", False),
    ]
    for col, (label, value, key, higher_is_better) in zip(cols, items):
        if mp:
            change = (m[key] / mp[key] - 1) * 100
            delta = f"{change:+.1f}% vs comparison"
            good = change >= 0 if higher_is_better else change <= 0
        else:
            delta, good = None, None
        col.markdown(kpi(label, value, delta, good), unsafe_allow_html=True)

    if mp:
        st.write("")
        keys = [("cpa", "Promo per new user"), ("promo_per_repeat", "Promo per repeat user/wk"),
                ("eta", "Avg delivery time (min)")]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[k[1] for k in keys], y=[mp[k[0]] for k in keys],
                             name=PRIOR_LABEL, marker_color=SAND))
        fig.add_trace(go.Bar(x=[k[1] for k in keys], y=[m[k[0]] for k in keys],
                             name=CUR_LABEL, marker_color=INK))
        fig.update_layout(**PLOT_LAYOUT, height=300, barmode="group",
                          title="Cost and service quality vs comparison period",
                          legend=dict(orientation="h", y=1.16, x=0))
        fig.update_yaxes(gridcolor=LINE, zeroline=False)
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# 6 — Decision brief
# --------------------------------------------------------------------------- #
with tabs[5]:
    pack = build_evidence_pack(weekly, cohorts, market, cur_weeks, prior_weeks,
                               CUR_LABEL, PRIOR_LABEL)
    c1, c2 = st.columns([1, 3])
    run = c1.button("Generate Recommendations", type="primary", width="stretch")
    c2.markdown('<div class="note">The model receives only pre-computed figures, never '
                'raw rows. It ranks, explains and recommends; it never calculates.</div>',
                unsafe_allow_html=True)

    key = f"{market}|{CUR_LABEL}|{PRIOR_LABEL}|{model_name}"
    if run or st.session_state.get("brief_key") == key:
        if run or "brief_text" not in st.session_state:
            with st.spinner("Reading the evidence pack…"):
                text, src = generate_brief(
                    pack, MODELS[model_name], token=get_token(),
                    system_prompt=st.session_state.get("system_prompt", SYSTEM_PROMPT))
            st.session_state.update(brief_key=key, brief_text=text, brief_src=src)
        with st.container(border=True):
            st.markdown(st.session_state.brief_text)
        st.caption(f"Written by: {st.session_state.brief_src}")
        st.download_button("Download the brief (Markdown)", st.session_state.brief_text,
                           f"decision_brief_{market.lower().replace(' ', '_')}.md",
                           "text/markdown")
    else:
        st.info("Set the market and periods in the sidebar, then generate the "
                "recommendations.")

# --------------------------------------------------------------------------- #
# 7 — How it works
# --------------------------------------------------------------------------- #
with tabs[6]:
    st.markdown("""
#### The design rule

**The model never does arithmetic.** A pandas layer decomposes the plan variance,
checks the funnel, reads the cohorts, and packages the result as a closed evidence pack.
Only then does an open-weights model read that pack and write the recommendations. That
split makes every number auditable and removes the failure mode that makes LLM reporting
untrustworthy — confident invented figures.

#### The chain

1. **Decompose** — chain-linked revenue bridge across sessions, traffic-to-user rate,
   OPU, AOV and take rate; the five contributions reconcile exactly to the total gap.
2. **Corroborate** — funnel step conversion versus the comparison period, cohort
   retention by acquisition month, promo per new and per repeat user, delivery time and
   cancellations.
3. **Narrate** — the model ranks causes by AED impact and writes three takeaways plus a
   table of actions, each with an owner, an expected impact and a guardrail metric.
4. **Falsify** — the brief ends with what would overturn the diagnosis and the next
   dataset to pull. An analyst who cannot be wrong is not being useful.

#### Guardrails

Seasonality sits in both plan and actuals, so Ramadan does not get blamed for a
structural gap. The prompt forbids figures outside the evidence pack. If inference is
unavailable the same brief is written by a rule-based writer from the same pack, so the
tool never shows a blank page.

#### The prompt

This is the instruction the model receives alongside the evidence pack. Edit it and
regenerate on the Decision brief tab to see how the output changes — loosening constraint
1 is the fastest way to watch a model start inventing numbers.
""")
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = SYSTEM_PROMPT
    edited = st.text_area("System prompt", value=st.session_state.system_prompt,
                          height=420, label_visibility="collapsed")
    c1, c2 = st.columns([1, 5])
    if c1.button("Save prompt"):
        st.session_state.system_prompt = edited
        st.session_state.pop("brief_key", None)
        st.success("Saved. Regenerate on the Decision brief tab.")
    if c2.button("Reset to default"):
        st.session_state.system_prompt = SYSTEM_PROMPT
        st.session_state.pop("brief_key", None)
        st.rerun()

    with st.expander("The evidence pack the model sees"):
        st.code(pack_to_json(build_evidence_pack(
            weekly, cohorts, market, cur_weeks, prior_weeks, CUR_LABEL, PRIOR_LABEL)),
            language="json")

    st.markdown("""
#### Data

Fully synthetic, generated by `generate_data.py` from a fixed seed — 26 weeks, four UAE
cities, weekly grain, plus a cohort table. No confidential information is used.
""")
