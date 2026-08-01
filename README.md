---
title: Growth Decision Brief — UAE Food
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Turns weekly plan-vs-actual data into a diagnosis and an action list.
---

# Growth Decision Brief — UAE Food

A growth manager's week is mostly one question: **we are off plan — why, and what do we
do on Monday?** Dashboards answer the first half slowly and the second half not at all.
This tool closes that loop.

Point it at weekly marketplace data and it decomposes the revenue gap, corroborates the
diagnosis against the funnel and the cohorts, and writes an exec-ready brief: three
takeaways, a table of actions with owners and guardrails, and an explicit statement of
what would prove the diagnosis wrong.

## The design rule

**The model never does arithmetic.**

A pandas layer (`growth_engine.py`) does all the maths and hands the model a closed
*evidence pack* of pre-computed figures. An open-weights model (`brief.py`) reads that
pack and writes the narrative. Nothing else.

That split is the whole point. It makes every number in the output auditable, and it
removes the failure mode that makes LLM-written reporting untrustworthy: confident
invented figures. The prompt is written as a contract — hard constraints, an output
schema, and a refusal rule (`"not in the data"`) for anything the pack does not contain.

## How the diagnosis works

Net revenue is decomposed through the identity that actually drives a marketplace:

```
Net revenue  =  MAU  ×  OPU  ×  AOV  ×  Take rate
```

The plan variance is split across those four factors with a chain-linked decomposition,
so the four contributions reconcile **exactly** to the total gap. That ranks causes by
AED impact rather than by which percentage looks most dramatic — a gap sitting in OPU is
a demand-quality problem, a gap in MAU is a top-of-funnel problem, and they need
different owners and different budgets.

The bridge is then corroborated against step-by-step funnel conversion versus the prior
period, new-user retention by acquisition cohort, promo cost per new user, average
delivery time and cancellation rate.

## Guardrails

- Seasonality (Ramadan, Eid, summer) is applied to **both** plan and actuals, so it
  cancels out of the variance and cannot be used as an excuse for a structural gap. The
  prompt explicitly forbids explaining variance as seasonal.
- The prompt forbids any figure not present in the evidence pack.
- Every recommended action must carry an owner function, an expected impact and a
  guardrail metric that says when the action is doing damage.
- The brief must end with what would overturn the diagnosis. An analyst who cannot be
  wrong is not being useful.
- If inference is unavailable, a rule-based writer produces the same brief structure
  from the same evidence pack. The tool never shows a blank page.

## The data

Fully synthetic and reproducible — `python generate_data.py` regenerates it from a fixed
seed. 26 weeks of 2025, weekly grain, four UAE cities (Dubai, Abu Dhabi, Sharjah,
Ajman), plus a cohort retention table. 


## Files

| File | What it does |
|---|---|
| `app.py` | Streamlit console: pulse, revenue bridge, funnel & cohorts, brief |
| `growth_engine.py` | All arithmetic: bridge, funnel, cohorts, evidence pack |
| `brief.py` | The prompt, the inference call, and the rule-based fallback |
| `generate_data.py` | Seeded synthetic dataset generator |
| `Dockerfile` | Container definition for any container host |

