"""
The narrative layer: turns the evidence pack into an exec-ready decision brief.

Two things matter here.

1. The prompt is the product. It is version-controlled, shown in the UI, and
   written as a contract: role, hard constraints, output schema, refusal rule.
2. The app degrades gracefully. If no inference token is configured or the
   provider is unavailable, a deterministic rule-based writer produces the same
   brief structure from the same evidence pack, so the tool is never a blank screen.
"""

from __future__ import annotations

import os
from growth_engine import pack_to_json

MODELS = {
    "Llama 3.3 70B Instruct (Meta, open weights)": "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen 2.5 72B Instruct (Alibaba, open weights)": "Qwen/Qwen2.5-72B-Instruct",
    "Mistral Small 24B Instruct (open weights)": "mistralai/Mistral-Small-24B-Instruct-2501",
    "Llama 3.1 8B Instruct (fast, open weights)": "meta-llama/Llama-3.1-8B-Instruct",
}

# --------------------------------------------------------------------------- #
# THE PROMPT  (this is also the deliverable for challenge 3)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You are a senior growth analyst on a food-delivery marketplace, writing the weekly \
decision brief that the General Manager reads before the business review. You are \
judged on whether the reader can make a decision from your brief without opening a \
dashboard.

HARD CONSTRAINTS
1. Use ONLY the numbers in the EVIDENCE block. Never estimate, extrapolate, or \
introduce a figure that is not there. If something needed for a claim is missing, \
write "not in the data" rather than guessing.
2. Do not perform arithmetic beyond restating a supplied number. Percentages, gaps \
and contributions are pre-computed for you.
3. The revenue bridge is the source of truth for WHY the gap exists. Rank causes by \
their AED contribution, not by how dramatic the percentage looks. The bridge starts at \
the top of the funnel: traffic, then the rate at which traffic becomes an active user, \
then order frequency, basket size and revenue capture. A traffic lift alongside a \
conversion or frequency drag means demand is arriving and leaking, which is a different \
problem from demand not arriving at all.
4. Separate correlation from causation. Where two metrics move together, say they \
are consistent with each other and name the test that would confirm it.
5. Seasonality is already reflected in both plan and actuals, so do not explain a \
variance as seasonal.
6. Every recommended action needs a named owner function, an expected direction of \
impact, and a guardrail metric that tells us if the action is doing damage.
7. Currency is AED. Be specific and quantitative. No filler, no praise, no hedging \
language like "it seems" or "potentially".

OUTPUT FORMAT — markdown, under 350 words, exactly these sections:

**Headline** — one sentence: are we on plan, and what is the single reason.

**Three takeaways** — three bullets. Each is one sentence: the metric, its size in \
AED or percentage points, and the driver behind it. Lead with the largest AED impact.

**Recommended actions** — a markdown table with columns: Action | Owner | Expected \
impact | Guardrail | Time to signal. Three to four rows, ordered by expected impact. \
Actions must be specific enough to start on Monday.

**What would change this read** — two bullets: the evidence that would overturn the \
diagnosis, and the one dataset you would pull next.
"""

USER_PROMPT_TEMPLATE = """\
Write the decision brief for {market}, covering {period}.

EVIDENCE
```json
{evidence}
```

Notes on reading the evidence:
- `revenue_gap_bridge_aed` decomposes the full actual-vs-plan revenue gap into five \
factors that sum to the total gap. Negative values are drags, positive values are lifts.
- `OPU_weekly` is orders per monthly-active user per week: order frequency.
- `retention_cohorts_month1_pct` is the share of each month's new users who ordered \
again the following month. Read it as a trend across cohorts.
- `city_league_table` is for context on whether this is a local or market-wide issue.
"""


def build_messages(pack: dict, system_prompt: str | None = None) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            market=pack["market"], period=pack["period"], evidence=pack_to_json(pack))},
    ]


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def generate_brief(pack: dict, model_id: str, token: str | None = None,
                   system_prompt: str | None = None,
                   temperature: float = 0.2) -> tuple[str, str]:
    """Returns (markdown_brief, source_label)."""
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        return fallback_brief(pack), "rule-based fallback (no HF_TOKEN configured)"

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(api_key=token)
        resp = client.chat.completions.create(
            model=model_id,
            messages=build_messages(pack, system_prompt),
            max_tokens=900,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip(), model_id
    except Exception as exc:  # noqa: BLE001
        return (fallback_brief(pack)
                + f"\n\n---\n*Model call did not complete ({type(exc).__name__}: {exc}). "
                  f"Showing the rule-based brief instead.*"), "rule-based fallback"


# --------------------------------------------------------------------------- #
# Deterministic fallback — same structure, no model required
# --------------------------------------------------------------------------- #
def _aed(x: float) -> str:
    a = abs(x)
    if a >= 1_000_000:
        return f"AED {x/1_000_000:,.2f}M"
    if a >= 1_000:
        return f"AED {x/1_000:,.0f}k"
    return f"AED {x:,.0f}"


def fallback_brief(pack: dict) -> str:
    r = pack["revenue"]
    bridge = sorted(pack["revenue_gap_bridge_aed"].items(), key=lambda kv: kv[1])
    drags = [(k, v) for k, v in bridge if v < 0]
    lifts = [(k, v) for k, v in bridge if v > 0]
    d = pack["drivers"]
    acq = pack["acquisition"]
    ops = pack["operations"]
    coh = pack["retention_cohorts_month1_pct"]
    steps = [s for s in pack["funnel_conversion"] if "conversion_pct" in s]
    worst = sorted(steps, key=lambda s: s.get("change_vs_comparison_pp", 0))[0] if steps else None

    coh_line = ""
    if len(coh) >= 2:
        ks = list(coh)
        coh_line = (f"Month-1 retention moved from {coh[ks[0]]}% for the {ks[0]} cohort "
                    f"to {coh[ks[-1]]}% for the {ks[-1]} cohort")

    status = "ahead of plan" if r["gap"] >= 0 else "behind plan"
    top = drags[0] if drags else lifts[-1]
    lift = lifts[-1] if lifts else None

    lines = [
        f"**Headline** \u2014 {pack['market']} closed {pack['period']} at "
        f"{r['attainment_pct']}% of the net revenue plan ({_aed(r['gap'])} {status}), "
        f"and the largest single factor is {top[0]} at {_aed(top[1])}.",
        "",
        "**Three takeaways**",
        f"- {top[0]} contributed {_aed(top[1])} to the gap, the biggest term in the "
        f"revenue bridge.",
    ]

    if lift:
        lines.append(
            f"- Demand is arriving: {lift[0]} added {_aed(lift[1])}, with sessions "
            f"{d['sessions_per_week']['vs_plan_pct']}% vs plan while order frequency ran "
            f"{d['OPU_weekly']['vs_plan_pct']}% vs plan \u2014 the shortfall is conversion "
            f"and repeat behaviour, not audience reach.")
    else:
        lines.append(
            f"- Sessions ran {d['sessions_per_week']['vs_plan_pct']}% vs plan and MAU "
            f"{d['MAU']['vs_plan_pct']}%, so the top of the funnel is part of the problem.")

    if coh_line:
        lines.append(f"- {coh_line}, while promo per new user moved from "
                     f"AED {acq['promo_cost_per_new_user_comparison']} to "
                     f"AED {acq['promo_cost_per_new_user_aed']}: acquisition is getting "
                     f"more expensive and less durable at the same time.")
    else:
        second = drags[1] if len(drags) > 1 else bridge[1]
        lines.append(f"- {second[0]} added {_aed(second[1])} to the gap.")

    leak = (f"{worst['step']} ({worst.get('change_vs_comparison_pp', 0):+.1f}pp vs "
            f"comparison)") if worst else "the weakest funnel step"

    lines += [
        "",
        "**Recommended actions**",
        "",
        "| Action | Owner | Expected impact | Guardrail | Time to signal |",
        "|---|---|---|---|---|",
        f"| Rebalance promo from acquisition to reactivation of 30-60 day lapsed users "
        f"| Growth + CRM | Recover frequency without adding spend | Promo % of GMV "
        f"(currently {acq['total_promo_pct_of_gmv']}%) | 2-3 weeks |",
        f"| Fix the weakest funnel step: {leak} | Product + Eng | Convert traffic that "
        f"is already arriving | Cancellation rate ({ops['cancellation_rate_pct']}%) "
        f"| 1-2 weeks |",
        f"| Supply and dispatch review in the worst ETA zones (avg "
        f"{ops['avg_delivery_time_min']} min vs {ops['avg_delivery_time_min_comparison']} "
        f"in the comparison window) | Operations | Restore conversion and repeat rate "
        f"| Cost per delivery | 3-4 weeks |",
        f"| Reset take rate on the deepest-discount merchant tier "
        f"({d['take_rate_pct']['vs_plan_pp']}pp vs plan) | Commercial | Recover revenue "
        f"capture | Order volume on affected merchants | 4 weeks |",
        "",
        "**What would change this read**",
        "- If the frequency decline is concentrated in one promo-acquired segment, this is "
        "an acquisition-quality problem rather than a base-user problem, and the promo "
        "rebalance is the wrong lever.",
        "- Next pull: order-level data split by acquisition channel and cohort, joined to "
        "delivery-zone ETA, to test whether frequency loss tracks service quality.",
        "",
        "*Generated by the rule-based writer from the same evidence pack the model receives.*",
    ]
    return "\n".join(lines)
