"""
Generates the synthetic dataset used by the Growth Decision Brief app.

Everything here is invented. No real or confidential marketplace data is used.
The generator is seeded, so the CSVs are reproducible from this file alone.

The data encodes a deliberate, realistic story so the analysis engine has
something true to find:

  - Dubai is AHEAD of plan on MAU but BEHIND on revenue.
  - The gap is driven by order frequency (OPU) and take rate, not audience size.
  - Acquisition promo per new user is climbing while new-cohort M1 retention falls
    -> we are buying users who do not come back.
  - Checkout -> order conversion and delivery times deteriorate in the same window,
    which is the operational explanation for the frequency drop.
  - Ramadan seasonality is applied to BOTH plan and actuals, so the residual gap
    is structural rather than seasonal.
  - Abu Dhabi runs ahead of plan; Sharjah is in line; Ajman is soft on audience.

Outputs (weekly grain, 4 UAE cities, 26 weeks of 2025):
  data/uae_food_weekly.csv
  data/uae_food_cohorts.csv
"""

import os

import numpy as np
import pandas as pd

SEED = 7
WEEKS = 26
START = pd.Timestamp("2025-01-06")  # Monday

rng = np.random.default_rng(SEED)

# Baseline end-to-end session -> order conversion used to set the traffic plan.
BASE_FUNNEL = 0.72 * 0.55 * 0.80 * 0.86

os.makedirs("data", exist_ok=True)

CITIES = {
    #             mau     opu_w  aov  take   mau_wow_plan  mau_wow_act
    "Dubai":     dict(mau=420_000, opu=0.80, aov=56.0, take=0.222, g_plan=0.0090, g_act=0.0116),
    "Abu Dhabi": dict(mau=168_000, opu=0.74, aov=54.0, take=0.225, g_plan=0.0085, g_act=0.0098),
    "Sharjah":   dict(mau=92_000,  opu=0.70, aov=47.0, take=0.228, g_plan=0.0080, g_act=0.0079),
    "Ajman":     dict(mau=31_000,  opu=0.64, aov=44.0, take=0.230, g_plan=0.0075, g_act=0.0048),
}


def seasonality(week_index: int) -> tuple[float, float]:
    """Ramadan (approx. weeks 8-12 of 2025) lifts orders and basket size; Eid week dips.
    Applied identically to plan and actual so it cancels out of the variance."""
    order_mult, aov_mult = 1.0, 1.0
    if 8 <= week_index <= 11:
        order_mult, aov_mult = 1.14, 1.11
    elif week_index == 12:  # Eid week
        order_mult, aov_mult = 0.91, 1.05
    elif week_index == 13:
        order_mult, aov_mult = 0.97, 1.00
    # mild summer softening from late June
    if week_index >= 24:
        order_mult *= 0.98
    return order_mult, aov_mult


def ramp(week_index: int, start_week: int, per_week: float, cap: float) -> float:
    """Linear deterioration/improvement that kicks in at start_week and caps out."""
    if week_index < start_week:
        return 0.0
    return min((week_index - start_week) * per_week, cap)


rows = []
for city, p in CITIES.items():
    is_dxb = city == "Dubai"
    is_auh = city == "Abu Dhabi"

    for w in range(WEEKS):
        week_start = START + pd.Timedelta(weeks=w)
        o_mult, a_mult = seasonality(w)
        noise = lambda s: float(rng.normal(1.0, s))

        # ---------- PLAN ----------
        mau_plan = p["mau"] * (1 + p["g_plan"]) ** w
        opu_plan = p["opu"] * (1 + 0.0015) ** w
        aov_plan = p["aov"] * (1 + 0.0010) ** w * a_mult
        take_plan = p["take"]

        orders_plan = mau_plan * opu_plan * o_mult
        gmv_plan = orders_plan * aov_plan
        rev_plan = gmv_plan * take_plan
        sessions_plan = orders_plan / BASE_FUNNEL

        # ---------- ACTUAL ----------
        mau_act = p["mau"] * (1 + p["g_act"]) ** w * noise(0.004)

        # Dubai: frequency erodes from week 13 onward (structural, post-Ramadan persistent)
        opu_drag = ramp(w, 13, 0.0075, 0.085) if is_dxb else (-ramp(w, 10, 0.0020, 0.028) if is_auh else 0.0)
        opu_act = p["opu"] * (1 + 0.0015) ** w * (1 - opu_drag) * noise(0.010)

        # Dubai: deeper discounting shows up as slightly lower net basket + lower take rate
        aov_act = aov_plan * (1 - (0.012 if is_dxb else 0.0)) * noise(0.006)
        take_drag = ramp(w, 12, 0.0011, 0.015) if is_dxb else 0.0
        take_act = take_plan - take_drag + (0.002 if is_auh else 0.0)

        orders_act = mau_act * opu_act * o_mult
        gmv_act = orders_act * aov_act
        rev_act = gmv_act * take_act

        # ---------- FUNNEL ----------
        # sessions -> menu views -> add to cart -> checkout started -> order
        r_menu = 0.72 * noise(0.006)
        r_cart = (0.55 - (ramp(w, 14, 0.0020, 0.030) if is_dxb else 0.0)) * noise(0.008)
        r_chk = 0.80 * noise(0.006)
        r_ord = (0.86 - (ramp(w, 14, 0.0060, 0.072) if is_dxb else 0.0)) * noise(0.005)

        sessions = orders_act / (r_menu * r_cart * r_chk * r_ord)
        menu_views = sessions * r_menu
        add_to_cart = menu_views * r_cart
        checkout_started = add_to_cart * r_chk

        # ---------- ACQUISITION & PROMO ----------
        new_rate = 0.081 + (ramp(w, 10, 0.0011, 0.020) if is_dxb else 0.0)
        new_users = mau_act * new_rate * noise(0.02)
        cpa = (18.0 + ramp(w, 10, 0.85, 13.0)) if is_dxb else (16.0 + w * 0.10)
        acq_promo = new_users * cpa
        ret_promo_pct = (0.031 + (ramp(w, 12, 0.0013, 0.017) if is_dxb else 0.0))
        ret_promo = gmv_act * ret_promo_pct

        # ---------- OPS ----------
        eta = (28.0 + (ramp(w, 14, 0.75, 9.0) if is_dxb else 0.0)) * noise(0.02)
        cancel = (0.021 + (ramp(w, 14, 0.00135, 0.016) if is_dxb else 0.0)) * noise(0.03)

        rows.append(dict(
            week_start=week_start.date(),
            city=city,
            mau_actual=round(mau_act),
            mau_plan=round(mau_plan),
            orders_actual=round(orders_act),
            orders_plan=round(orders_plan),
            gmv_actual_aed=round(gmv_act, 2),
            gmv_plan_aed=round(gmv_plan, 2),
            net_revenue_actual_aed=round(rev_act, 2),
            net_revenue_plan_aed=round(rev_plan, 2),
            sessions=round(sessions),
            sessions_plan=round(sessions_plan),
            menu_views=round(menu_views),
            add_to_cart=round(add_to_cart),
            checkout_started=round(checkout_started),
            new_users=round(new_users),
            acquisition_promo_aed=round(acq_promo, 2),
            retention_promo_aed=round(ret_promo, 2),
            avg_delivery_time_min=round(eta, 1),
            cancellation_rate=round(cancel, 4),
        ))

weekly = pd.DataFrame(rows)
weekly.to_csv("data/uae_food_weekly.csv", index=False)

# ---------- COHORT RETENTION ----------
# % of a month's new users who order again in month M+n.
cohorts = []
base_curve = {1: 0.41, 2: 0.31, 3: 0.26, 4: 0.23, 5: 0.21}
city_offset = {"Dubai": 0.00, "Abu Dhabi": 0.015, "Sharjah": -0.005, "Ajman": -0.02}
# Dubai's newer cohorts are materially weaker; other cities are stable.
decay = {"Dubai": 0.024, "Abu Dhabi": 0.002, "Sharjah": 0.004, "Ajman": 0.005}

for city in CITIES:
    for ci, cohort in enumerate(pd.date_range("2025-01-01", periods=6, freq="MS")):
        for m, base in base_curve.items():
            if ci + m > 5:
                continue
            val = (base + city_offset[city] - decay[city] * ci) * float(rng.normal(1.0, 0.012))
            cohorts.append(dict(
                cohort_month=cohort.strftime("%Y-%m"),
                city=city,
                months_since_first_order=m,
                retained_pct=round(max(val, 0.05), 4),
            ))

pd.DataFrame(cohorts).to_csv("data/uae_food_cohorts.csv", index=False)

print(weekly.groupby("city")[["net_revenue_actual_aed", "net_revenue_plan_aed"]].sum().round(0))
print("\nrows:", len(weekly), "| cohort rows:", len(cohorts))
