# DECISIONS.md — Engineering & Product Decisions

## 1. Why this approach?

I chose to evolve an existing NTSB aviation data-science project into a product-focused intelligence interface (`AeroIntel`) rather than creating an unrelated fictional SaaS product.

By grounding the product in an actual dataset of **39,017 cleaned U.S. NTSB aviation incidents** and a trained **XGBoost machine-learning classification model** (F1-score: 0.5707), the application maintains deep technical authenticity while meeting the challenge's product craft and UX requirements.

### Qatar Airways 5-Star Luxury Theme & Product Refinement:
1. **Qatar Airways Visual Identity**: Modeled after Qatar Airways' world-famous 5-star aviation visual design system, featuring deep Qatar Burgundy/Maroon background tones (`#5C0632`, `#3B0420`, `#0B030A`), luxurious Champagne Gold accents (`#D4AF37`), crisp cream/white typography (`#FDFBF7`), and booking-card inspired glassmorphism panels.
2. **Pre-Incident Operational Focus**: Aircraft damage was removed from scenario inputs because damage occurs post-incident. Scenario evaluation focuses strictly on pre-flight parameters (`WeatherCondition`, `AmateurBuilt`, `PurposeOfFlight`, `AirCraftCategory`, `NumberOfEngines`, `Season`, `Year`).
3. **Explicit Dataset & Code Pipeline Preview**: Added an interactive section allowing visitors to preview sample NTSB records (`aviation_cleaned.csv`), filter records with live search, download CSV datasets (`aviation_cleaned.csv` and `aviation.csv`), and view step-by-step Python code highlights from `av.ipynb`.

---

## 2. One trade-off under the time limit

Given the **3–4 hour time constraint**, I prioritized building a highly polished, responsive visual frontend interface, interactive scenario analyzer micro-interaction, live dataset explorer, and detailed data visualizations over deploying a production Python ML model server (e.g. Flask/FastAPI backend API) and authentication pipeline.

The frontend includes a self-contained JS inference engine (`analyzeIncidentScenario`) implementing the exact scoring weights, feature importances, and class probability distributions derived from the trained XGBoost model.

### With a full dedicated development week, I would implement:
1. **API & Microservice Architecture**: Containerized FastAPI service wrapping `best_model_xgboost.pkl`, `scaler.pkl`, and `label_encoders.pkl` with automated OpenAPI specs.
2. **Model Explainability & SHAP**: Real-time SHAP force-plots displaying individual feature contributions per incident scenario.
3. **Uncertainty & Calibration**: Temperature scaling and confidence interval bounds on predicted class probabilities.
4. **Enhanced Data Pipeline**: Automated ETL fetching new NTSB monthly exports with active model monitoring and drift detection.

---

## 3. AI Usage

AI tools were utilized transparently as pair-programming assistants throughout the challenge:

- **Data Inspection & Feature Extraction**: Assisted in parsing `av.ipynb` JSON cells to extract model evaluation tables, class target distributions, and feature importance rankings accurately.
- **Qatar Airways Design Tokens**: Assisted in configuring Qatar Burgundy variables (`#5C0632`, `#3B0420`), Champagne Gold accents (`#D4AF37`), specular border finishes, and gold ambient aura glows.
- **Interactive Component Architecture**: Assisted in drafting dynamic JS handlers for sidebar tabs, dataset table searching, and code step switching.

All code, data metrics, UX decisions, visual hierarchies, and operational disclaimers were personally reviewed, engineered, and validated.
