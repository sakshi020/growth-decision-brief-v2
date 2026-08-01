"""
Deterministic analytics layer.

Design rule for this project: the model never does arithmetic. Every number in the
brief is computed here in pandas, then handed to the LLM as a closed "evidence pack".
The LLM's only job is to prioritise, explain and recommend. That keeps the numbers
auditable and removes the main failure mode of LLM-written reporting.

Revenue identity used throughout — five factors, starting at the top of the funnel:

    Net revenue = Sessions x User rate x OPU x AOV x Take rate

    where  User rate = MAU / sessions      (traffic that becomes an active user)
           OPU       = orders / MAU        (order frequency per active user, per week)
           AOV       = GMV / orders        (average order value)
           Take rate = net revenue / GMV   (revenue capture after discounts)

Sessions x User rate collapses to MAU, so the identity holds exactly while exposing
traffic as its own lever. The plan-vs-actual gap is split across the five factors with
a chain-linked (sequential) decomposition, which reconciles exactly to the total gap.
"""

from __future__ import annotations

import json
import pandas as pd

BASE_FUNNEL = 0.72 * 0.55 * 0.80 * 0.86

FACTORS = ["sessions", "user_rate", "opu", "aov", "take"]
FACTOR_LABELS = {
    "sessions": "Traffic (sessions)",
    "user_rate": "Traffic to active-user rate",
    "opu": "OPU (order frequency)",
    "aov": "AOV (basket size)",
    "take": "Take rate (revenue capture)",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_data(weekly_path: str, cohort_path: str):
    weekly = pd.read_csv(weekly_path, parse_dates=["week_start"])
    if "sessions_plan" not in weekly.columns:
        # Backwards compatibility with datasets generated before the traffic factor.
        weekly["sessions_plan"] = weekly["orders_plan"] / BASE_FUNNEL
    cohorts = pd.read_csv(cohort_path)
    return weekly.sort_values(["city", "week_start"]), cohorts


def month_label(ts) -> str:
    return pd.Timestamp(ts).strftime("%b %Y")


def month_options(weekly: pd.DataFrame) -> list[str]:
    months = sorted(weekly.week_start.dt.to_period("M").unique())
    return [m.to_timestamp().strftime("%b %Y") for m in months]


def weeks_in_range(weekly: pd.DataFrame, start_label: str, end_label: str) -> list:
    opts = month_options(weekly)
    lo, hi = sorted([opts.index(start_label), opts.index(end_label)])
    chosen = set(opts[lo:hi + 1])
    mask = weekly.week_start.dt.strftime("%b %Y").isin(chosen)
    return sorted(weekly.loc[mask, "week_start"].unique())


# --------------------------------------------------------------------------- #
# Core metric block — one dict per slice, reused everywhere
# --------------------------------------------------------------------------- #
def metrics(df: pd.DataFrame) -> dict:
    """Every headline metric for a slice. MAU and sessions are stocks/flows measured
    per week, so they are averaged across weeks; everything else is summed."""
    n = df["week_start"].nunique()
    if n == 0:
        return {}
    per_week = df.groupby("week_start")
    sess_a = per_week["sessions"].sum().mean()
    sess_p = per_week["sessions_plan"].sum().mean()
    mau_a = per_week["mau_actual"].sum().mean()
    mau_p = per_week["mau_plan"].sum().mean()
    new_a = per_week["new_users"].sum().mean()
    o_a, o_p = df.orders_actual.sum(), df.orders_plan.sum()
    g_a, g_p = df.gmv_actual_aed.sum(), df.gmv_plan_aed.sum()
    r_a, r_p = df.net_revenue_actual_aed.sum(), df.net_revenue_plan_aed.sum()
    repeat = max(mau_a - new_a, 1.0)

    return dict(
        weeks=n,
        sessions_a=sess_a, sessions_p=sess_p,
        user_rate_a=mau_a / sess_a, user_rate_p=mau_p / sess_p,
        mau_a=mau_a, mau_p=mau_p,
        opu_a=o_a / mau_a / n, opu_p=o_p / mau_p / n,
        aov_a=g_a / o_a, aov_p=g_p / o_p,
        take_a=r_a / g_a, take_p=r_p / g_p,
        orders_a=o_a, orders_p=o_p,
        gmv_a=g_a, gmv_p=g_p,
        rev_a=r_a, rev_p=r_p,
        attainment=r_a / r_p,
        new_users=df.new_users.sum(),
        repeat_users=repeat,
        cpa=df.acquisition_promo_aed.sum() / max(df.new_users.sum(), 1),
        promo_per_repeat=df.retention_promo_aed.sum() / (repeat * n),
        promo_pct_gmv=(df.acquisition_promo_aed.sum() + df.retention_promo_aed.sum()) / g_a,
        eta=df.avg_delivery_time_min.mean(),
        cancel=df.cancellation_rate.mean(),
        conv_overall=o_a / df.sessions.sum(),
    )


# --------------------------------------------------------------------------- #
# Revenue bridge — five factors
# --------------------------------------------------------------------------- #
def revenue_bridge(df: pd.DataFrame) -> dict:
    """Chain-linked decomposition of the actual-vs-plan revenue gap across the five
    factors. Contributions sum exactly to (actual - plan)."""
    a = metrics(df)
    w = a["weeks"]
    p = [a["sessions_p"], a["user_rate_p"], a["opu_p"], a["aov_p"], a["take_p"]]
    q = [a["sessions_a"], a["user_rate_a"], a["opu_a"], a["aov_a"], a["take_a"]]

    contributions = {}
    for i, name in enumerate(FACTORS):
        term = w
        for j in range(len(FACTORS)):
            if j < i:
                term *= q[j]
            elif j == i:
                term *= (q[j] - p[j])
            else:
                term *= p[j]
        contributions[name] = term

    total = a["rev_a"] - a["rev_p"]
    drift = total - sum(contributions.values())
    biggest = max(contributions, key=lambda k: abs(contributions[k]))
    contributions[biggest] += drift

    return dict(
        agg=a,
        contributions=contributions,
        total_gap=total,
        attainment=a["attainment"],
        largest_drag=min(contributions, key=lambda k: contributions[k]),
        largest_lift=max(contributions, key=lambda k: contributions[k]),
    )


# --------------------------------------------------------------------------- #
# Funnel
# --------------------------------------------------------------------------- #
FUNNEL_STEPS = [
    ("Total traffic", "sessions", None),
    ("Menu views", "menu_views", "sessions"),
    ("Add to cart", "add_to_cart", "menu_views"),
    ("Checkout started", "checkout_started", "add_to_cart"),
    ("Orders", "orders_actual", "checkout_started"),
]


def funnel(df: pd.DataFrame, compare: pd.DataFrame | None = None) -> pd.DataFrame:
    def rates(d):
        return {label: (d[col].sum() / d[prev].sum()) if prev else None
                for label, col, prev in FUNNEL_STEPS}

    cur = rates(df)
    prior = rates(compare) if compare is not None and len(compare) else None
    rows = []
    for label, col, prev in FUNNEL_STEPS:
        row = {"Step": label, "Volume": df[col].sum(), "Step conversion": cur[label]}
        if prior:
            if prev is None:
                row["Change"] = (df[col].sum() / compare[col].sum() - 1) * 100
                row["Change unit"] = "%"
            else:
                row["Change"] = (cur[label] - prior[label]) * 100
                row["Change unit"] = "pp"
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Cohorts
# --------------------------------------------------------------------------- #
def cohort_matrix(cohorts: pd.DataFrame, city: str | None = None) -> pd.DataFrame:
    d = cohorts if city in (None, "All UAE") else cohorts[cohorts.city == city]
    return (d.pivot_table(index="cohort_month", columns="months_since_first_order",
                          values="retained_pct", aggfunc="mean").sort_index())


# --------------------------------------------------------------------------- #
# Tables: overview, weekly breakdown, period comparison
# --------------------------------------------------------------------------- #
METRIC_ROWS = [
    ("Sessions (avg/week)", "sessions_a", "{:,.0f}", "pct"),
    ("Traffic to user rate", "user_rate_a", "{:.1%}", "pp_rate"),
    ("MAU (avg/week)", "mau_a", "{:,.0f}", "pct"),
    ("Orders", "orders_a", "{:,.0f}", "pct"),
    ("OPU (orders/user/wk)", "opu_a", "{:.3f}", "pct"),
    ("AOV (AED)", "aov_a", "{:,.1f}", "pct"),
    ("Take rate", "take_a", "{:.1%}", "pp_rate"),
    ("GMV (AED)", "gmv_a", "{:,.0f}", "pct"),
    ("Net revenue (AED)", "rev_a", "{:,.0f}", "pct"),
    ("Plan attainment", "attainment", "{:.1%}", "pp_rate"),
    ("Session to order conversion", "conv_overall", "{:.2%}", "pp_rate"),
    ("New users", "new_users", "{:,.0f}", "pct"),
    ("Promo per new user (AED)", "cpa", "{:,.1f}", "pct"),
    ("Promo per repeat user/wk (AED)", "promo_per_repeat", "{:,.2f}", "pct"),
    ("Promo as % of GMV", "promo_pct_gmv", "{:.1%}", "pp_rate"),
    ("Avg delivery time (min)", "eta", "{:,.1f}", "pct"),
    ("Cancellation rate", "cancel", "{:.2%}", "pp_rate"),
]


def overview_table(weekly: pd.DataFrame, weeks: list) -> pd.DataFrame:
    d = weekly[weekly.week_start.isin(weeks)]
    rows = []
    for city, g in d.groupby("city"):
        m = metrics(g)
        b = revenue_bridge(g)
        rows.append({
            "City": city,
            "Sessions (avg/wk)": m["sessions_a"],
            "MAU (avg/wk)": m["mau_a"],
            "Orders": m["orders_a"],
            "OPU": m["opu_a"],
            "AOV (AED)": m["aov_a"],
            "Take rate": m["take_a"],
            "Net revenue (AED)": m["rev_a"],
            "Plan (AED)": m["rev_p"],
            "Attainment": m["attainment"],
            "Gap (AED)": m["rev_a"] - m["rev_p"],
            "Largest drag": FACTOR_LABELS[b["largest_drag"]],
        })
    total = metrics(d)
    tb = revenue_bridge(d)
    rows.append({
        "City": "All UAE", "Sessions (avg/wk)": total["sessions_a"],
        "MAU (avg/wk)": total["mau_a"], "Orders": total["orders_a"],
        "OPU": total["opu_a"], "AOV (AED)": total["aov_a"], "Take rate": total["take_a"],
        "Net revenue (AED)": total["rev_a"], "Plan (AED)": total["rev_p"],
        "Attainment": total["attainment"], "Gap (AED)": total["rev_a"] - total["rev_p"],
        "Largest drag": FACTOR_LABELS[tb["largest_drag"]],
    })
    return pd.DataFrame(rows)


def weekly_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wk, g in df.groupby("week_start"):
        m = metrics(g)
        rows.append({
            "Week": pd.Timestamp(wk).strftime("%d %b %Y"),
            "Sessions": m["sessions_a"], "MAU": m["mau_a"], "Orders": m["orders_a"],
            "OPU": m["opu_a"], "AOV (AED)": m["aov_a"], "Take rate": m["take_a"],
            "Net revenue (AED)": m["rev_a"], "Plan (AED)": m["rev_p"],
            "Attainment": m["attainment"],
        })
    return pd.DataFrame(rows)


def comparison_table(cur: pd.DataFrame, prior: pd.DataFrame,
                     cur_label: str, prior_label: str) -> pd.DataFrame:
    a, b = metrics(cur), metrics(prior)
    if not a or not b:
        return pd.DataFrame()
    rows = []
    for name, key, fmt, kind in METRIC_ROWS:
        va, vb = a[key], b[key]
        if kind == "pp_rate":
            change = f"{(va - vb) * 100:+.2f} pp"
        else:
            change = f"{(va / vb - 1) * 100:+.1f}%" if vb else "n/a"
        rows.append({"Metric": name, cur_label: fmt.format(va),
                     prior_label: fmt.format(vb), "Change": change})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Evidence pack -> the only thing the model ever sees
# --------------------------------------------------------------------------- #
def build_evidence_pack(weekly, cohorts, city, weeks, prior_weeks,
                        period_label="", prior_label="") -> dict:
    scope = weekly if city == "All UAE" else weekly[weekly.city == city]
    cur = scope[scope.week_start.isin(weeks)]
    prior = scope[scope.week_start.isin(prior_weeks)] if prior_weeks is not None else scope.head(0)

    b = revenue_bridge(cur)
    a = b["agg"]
    p = metrics(prior) if len(prior) else {}

    fn = funnel(cur, prior)
    funnel_out = []
    for _, r in fn.iterrows():
        entry = {"step": r["Step"]}
        if pd.notna(r["Step conversion"]):
            entry["conversion_pct"] = round(r["Step conversion"] * 100, 1)
        else:
            entry["volume_per_period"] = int(r["Volume"])
        if "Change" in fn.columns and pd.notna(r["Change"]):
            entry[f"change_vs_comparison_{r['Change unit']}"] = round(r["Change"], 1)
        funnel_out.append(entry)

    cm = cohort_matrix(cohorts, city)
    m1 = cm[1].dropna() if 1 in cm.columns else pd.Series(dtype=float)

    def pct(x):
        return round(x * 100, 1)

    def vs(key, as_pp=False):
        if not p:
            return None
        return (round((a[key] - p[key]) * 100, 2) if as_pp
                else round((a[key] / p[key] - 1) * 100, 1))

    return {
        "market": city,
        "period": period_label,
        "comparison_period": prior_label or "none selected",
        "currency": "AED",
        "revenue": {
            "actual": round(a["rev_a"]), "plan": round(a["rev_p"]),
            "gap": round(b["total_gap"]), "attainment_pct": pct(b["attainment"]),
        },
        "revenue_gap_bridge_aed": {FACTOR_LABELS[k]: round(v)
                                   for k, v in b["contributions"].items()},
        "drivers": {
            "sessions_per_week": {"actual": round(a["sessions_a"]),
                                  "plan": round(a["sessions_p"]),
                                  "vs_plan_pct": pct(a["sessions_a"] / a["sessions_p"] - 1),
                                  "vs_comparison_pct": vs("sessions_a")},
            "traffic_to_user_rate_pct": {"actual": pct(a["user_rate_a"]),
                                         "plan": pct(a["user_rate_p"]),
                                         "vs_plan_pp": round((a["user_rate_a"] - a["user_rate_p"]) * 100, 2)},
            "MAU": {"actual": round(a["mau_a"]), "plan": round(a["mau_p"]),
                    "vs_plan_pct": pct(a["mau_a"] / a["mau_p"] - 1),
                    "vs_comparison_pct": vs("mau_a")},
            "OPU_weekly": {"actual": round(a["opu_a"], 3), "plan": round(a["opu_p"], 3),
                           "vs_plan_pct": pct(a["opu_a"] / a["opu_p"] - 1),
                           "vs_comparison_pct": vs("opu_a")},
            "AOV": {"actual": round(a["aov_a"], 2), "plan": round(a["aov_p"], 2),
                    "vs_plan_pct": pct(a["aov_a"] / a["aov_p"] - 1)},
            "take_rate_pct": {"actual": pct(a["take_a"]), "plan": pct(a["take_p"]),
                              "vs_plan_pp": round((a["take_a"] - a["take_p"]) * 100, 2)},
            "orders": {"actual": round(a["orders_a"]), "plan": round(a["orders_p"]),
                       "vs_plan_pct": pct(a["orders_a"] / a["orders_p"] - 1)},
        },
        "funnel_conversion": funnel_out,
        "acquisition": {
            "new_users": round(a["new_users"]),
            "promo_cost_per_new_user_aed": round(a["cpa"], 1),
            "promo_cost_per_new_user_comparison": round(p["cpa"], 1) if p else None,
            "promo_per_repeat_user_per_week_aed": round(a["promo_per_repeat"], 2),
            "promo_per_repeat_user_comparison": round(p["promo_per_repeat"], 2) if p else None,
            "total_promo_pct_of_gmv": pct(a["promo_pct_gmv"]),
            "total_promo_pct_of_gmv_comparison": pct(p["promo_pct_gmv"]) if p else None,
        },
        "retention_cohorts_month1_pct": {k: pct(v) for k, v in m1.items()},
        "operations": {
            "avg_delivery_time_min": round(a["eta"], 1),
            "avg_delivery_time_min_comparison": round(p["eta"], 1) if p else None,
            "cancellation_rate_pct": pct(a["cancel"]),
            "cancellation_rate_pct_comparison": pct(p["cancel"]) if p else None,
        },
        "city_league_table": [
            {"city": r["City"], "attainment_pct": pct(r["Attainment"]),
             "gap_aed": round(r["Gap (AED)"]), "largest_drag": r["Largest drag"]}
            for _, r in overview_table(weekly, weeks).iterrows()
        ],
    }


def pack_to_json(pack: dict) -> str:
    return json.dumps(pack, indent=2, default=str)
