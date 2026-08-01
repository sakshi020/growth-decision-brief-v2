---
title: Growth Decision Brief — UAE Food
emoji: ◆
colorFrom: green
colorTo: gray
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
Ajman), plus a cohort retention table. **No confidential information is used.**

The generator encodes a deliberate story so the engine has something real to find: Dubai
runs *ahead* of plan on audience but *behind* on revenue, because order frequency and
take rate are eroding while acquisition promo per new user climbs and each new cohort
retains worse than the last — the shape of a market buying users it cannot keep.

## Run it

```bash
pip install -r requirements.txt
python generate_data.py          # writes data/*.csv
streamlit run app.py
```

## Deploy it (Streamlit Community Cloud — free)

Hugging Face moved Gradio and Docker Spaces behind a paid plan in mid-2026, so
Community Cloud is the free route. Public apps are unlimited; you need a GitHub account.

1. Push this folder to a **public** GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **Create app** → **Deploy a public app from GitHub** → pick the repo, branch `main`,
   main file path `app.py`.
4. Under **Advanced settings → Secrets**, paste:

   ```toml
   HF_TOKEN = "hf_your_token_here"
   ```

   Use a Hugging Face access token with *Make calls to Inference Providers* permission.
   Inference is free to call from anywhere — only Spaces *hosting* became paid.
5. Deploy. The app builds in two or three minutes at
   `https://<your-app-name>.streamlit.app`.

Without the secret the app still runs, on the rule-based writer. Free apps sleep after
about 12 hours of no traffic and wake on the next visit, so open the link once before
you share it.

### Other hosts

A `Dockerfile` is included for anywhere that takes a container — Render, Railway, Fly,
a VPS, or a Hugging Face Docker Space if you have PRO. It serves on port 8501 and reads
`HF_TOKEN` from the environment.

## Files

| File | What it does |
|---|---|
| `app.py` | Streamlit console: pulse, revenue bridge, funnel & cohorts, brief |
| `growth_engine.py` | All arithmetic: bridge, funnel, cohorts, evidence pack |
| `brief.py` | The prompt, the inference call, and the rule-based fallback |
| `generate_data.py` | Seeded synthetic dataset generator |
| `Dockerfile` | Container definition for any container host |

## 100-word summary

> Growth managers spend their week asking one question: we are off plan, why, and what do
> we do Monday? This tool answers both. A pandas engine decomposes the revenue gap through
> MAU × OPU × AOV × take rate using a chain-linked bridge that reconciles exactly, then
> corroborates it against funnel conversion, cohort retention, promo cost per new user and
> delivery times. Only then does an open-weights model read that closed evidence pack and
> write three takeaways plus actions with owners and guardrails. The rule is deliberate:
> the model never does arithmetic, so every figure is auditable. Data is synthetic and
> reproducible.
