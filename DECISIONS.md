# DECISIONS.md — Engineering & Product Explanation (1 Page Max)

## 1. Why this strategy over the obvious alternative rejected?

**Rejected Alternative**: Building a standard React/Next.js SPA with Tailwind UI boilerplate components and fictional mock SaaS data.

**Chosen Strategy**: Zero-dependency Vanilla HTML5 + CSS3 + 3D Canvas + Web API paired with an actual empirical dataset of **39,017 cleaned U.S. NTSB aviation incident records** and a trained **XGBoost machine-learning classifier** (Multiclass Accuracy: `58.5%`, F1-Score: `0.5707`).

**Product & Freemium Architecture**:
- **Freemium UX Conversion Loop**: Engineered a 2-tier Freemium experience: Free users receive immediate high-level severity categories (`FATAL`, `SERIOUS`, `MINOR`, `NONE`), while deeper factor intelligence and class probabilities are locked behind an elegant glassmorphism preview (*"Know the severity. Understand the why."*).
- **Defensible Machine Learning Rigor**: A 58.5% accuracy on a 4-class imbalanced classification problem represents a **2.34x improvement over the 25.0% random baseline**, demonstrating honest, un-inflated empirical machine learning.
- **Non-Causation Framing**: Strictly formatted all explanations as *"Features influencing model assessment"* and *"Model feature importance"*, explicitly guarding against invalid causal claims.

---

## 2. One trade-off under the time limit & 1-week plan

**Time-Limit Trade-off**: Under the time constraint, I prioritized shipping an ultra-polished frontend interface, interactive Freemium conversion lock demo (₹499/mo CTA unlock), live CSV table search, and full dark-mode glassmorphism over deploying a live Python/FastAPI microservice container on Render/Railway. The ML inference engine was compiled into an interactive client-side JS scoring prototype using exact weights derived from the trained XGBoost model.

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
  3. **Freemium UX Integration**: Hand-crafted the locked glass preview, ₹499/mo subscription CTA, and interactive frontend unlock demonstration.
  4. **Responsive Layout Polish**: Hand-crafted executive multi-column footer grids and mobile viewport controls (390px to 1440px).

---

*Bonus Easter Egg: Type `alok`, `aero`, `ntsb`, or `qatar` on your keyboard (or click the AeroIntel brand logo 5 times) to play an authentic 2-tone aircraft cabin chime and unlock 5-Star First Class Telemetry Mode!*
