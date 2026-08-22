# DECISIONS.md — Engineering & Product Explanation (1 Page Max)

## 1. Why this strategy over the obvious alternative rejected?

**Rejected Alternative**: Building a standard React/Next.js SPA with Tailwind UI boilerplate components and fictional mock SaaS data.

**Chosen Strategy**: Zero-dependency Vanilla HTML5 + CSS3 + 3D Canvas + Web API paired with an actual empirical dataset of **39,017 cleaned U.S. NTSB aviation incident records** and a trained **XGBoost machine-learning classifier** (Multiclass Accuracy: `58.5%`, F1-Score: `0.5707`).

**Product & Sequential SaaS Checkout**:
- **Strictly Sequential Checkout Flow**: Engineered a clean, 1-state-at-a-time checkout modal (`Checkout Form` ➔ `Processing Payment...` ➔ `Payment Successful` ➔ `AeroIntel Intelligence Unlocked`). Form, processing, and success screens are strictly separated to prevent visual clutter.
- **Multi-Method Indian SaaS Checkout**: Supports compact selectable payment tabs (`⚡ UPI`, `💳 Card`, `🏦 Net Banking`, `👛 Wallet`) tailored for ₹499/month subscription pricing, defaulting to UPI ID input.
- **1 Free Analysis Limit**: Every new user receives exactly 1 free scenario analysis showing high-level severity (`FATAL`, `SERIOUS`, `MINOR`, `NONE`), while factor intelligence remains locked (`Want to know why?`). Attempting a 2nd analysis triggers a persistent subscription upgrade lock.
- **Defensible Machine Learning Rigor**: A 58.5% accuracy on a 4-class imbalanced classification problem represents a **2.34x improvement over the 25.0% random baseline**, demonstrating honest empirical ML.
- **Non-Causation Framing**: Strictly formatted all explanations as *"Features influencing model assessment"* and *"Model feature importance"*, guarding against invalid causal claims.

---

## 2. One trade-off under the time limit & 1-week plan

**Time-Limit Trade-off**: Under the time constraint, I prioritized shipping an ultra-polished frontend interface, interactive sequential checkout modal, 1-free-analysis persistent state, live CSV table search, and dark-mode glassmorphism over deploying a live Python/FastAPI microservice container on Render/Railway. The ML inference engine was compiled into an interactive client-side JS scoring prototype using exact weights derived from the trained XGBoost model.

**What I'd do with a real week**:
1. **Production ML Microservice**: Containerize a FastAPI server serving `best_model_xgboost.pkl` with automated OpenAPI endpoints and CORS security.
2. **Real-time SHAP Explainability**: Render interactive SHAP force-plots displaying exact positive/negative feature contributions per incident scenario.
3. **Automated ETL Pipeline**: Build automated NTSB monthly ingestion workers with model drift monitoring and retraining triggers.

---

## 3. AI Usage & Personal Verification

AI tools were used as a pair-programming multiplier:
- **Where AI was used**: Generating 3D projection math matrix boilerplate, parsing `av.ipynb` notebook JSON cells for evaluation tables, and prototyping CSS glassmorphism token combinations.
- **What I personally verified & changed**:
  1. **Scoping & Domain Logic**: Explicitly restricted model parameters to pre-flight variables (`WeatherCondition`, `AmateurBuilt`, `PurposeOfFlight`, `Engines`), eliminating post-incident damage variables to prevent data leakage.
  2. **Strict Honesty Audit**: Verified that all accuracy numbers reflect actual notebook evaluations (58.5% accuracy / 0.5707 F1-score) without artificial inflation. Replaced stock photos with NTSB case study graphics.
  3. **Sequential SaaS Checkout**: Hand-crafted the 520px compact modal card, multi-method tab switcher (UPI/Card/NetBanking/Wallet), sequential processing states, and post-payment intelligence reveal.
  4. **Responsive Layout Polish**: Hand-crafted executive multi-column footer grids and mobile viewport controls (390px to 1440px).

---

*Bonus Easter Egg: Type `alok`, `aero`, `ntsb`, or `qatar` on your keyboard (or click the AeroIntel brand logo 5 times) to play an authentic 2-tone aircraft cabin chime and unlock 5-Star First Class Telemetry Mode!*
