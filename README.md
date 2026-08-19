# AeroIntel — Aviation Incident Intelligence

> Understand what makes aviation incidents severe using historical NTSB data and machine learning analysis.

Built for the **ACdyon Technologies Frontend Challenge (Part 2 — The Premium Home Page)**.

---

## ✈️ Overview

**AeroIntel** is a premium aviation intelligence platform homepage built around an actual NTSB aviation data-science capstone project. It allows users to explore historical U.S. aviation incident patterns and perform interactive, ML-assisted pre-incident scenario severity assessments based on real historical incident characteristics.

---

## ✨ Features & Qatar Airways Luxury Theme

- **Qatar Airways 5-Star Visual Design System**: Modeled after Qatar Airways' world-famous homepage aesthetic, featuring rich Qatar Burgundy/Maroon canvas backgrounds (`#5C0632`, `#3B0420`), luxurious Champagne Gold accents (`#D4AF37`), crisp cream/white typography (`#FDFBF7`), and booking-engine style glassmorphism cards.
- **Data & Code Pipeline Preview**: Dedicated section providing:
  - **Dataset Table Preview**: Preview sample NTSB records from `aviation_cleaned.csv`, search records in real time, and download `aviation_cleaned.csv` (8.5 MB) & raw `aviation.csv` (22.8 MB).
  - **Code Pipeline Highlights**: Interactive step switcher displaying Python / Pandas / XGBoost notebook code highlights from `av.ipynb` with download link for `av.ipynb`.
- **Dynamic AeroIntel Console Tabs**: Interactive sidebar switching between *Scenario Inspector*, *NTSB Dataset Explorer*, *XGBoost Model Metrics*, and *Parameters & Encoding*. Live telemetry updates in real-time when running scenario analysis.
- **Pre-Incident Scenario Analyzer**: Interactive scenario analyzer focused strictly on pre-incident parameters (`WeatherCondition`, `PurposeOfFlight`, `AmateurBuilt`, `AirCraftCategory`, `NumberOfEngines`, `Season`, `Year`).
- **Historical Intelligence Cards**: Empirical visualizations covering incident trajectory, weather risk multipliers (3.4x IMC fatality rate), homebuilt fleet risk, and flight purpose profiles.
- **Honest Methodology & Disclaimers**: Full transparency on NTSB target distributions, model performance (Logistic Regression vs Random Forest vs XGBoost F1 `0.5707`), and operational limitations.
- **100% Responsive Design**: Tested at 390px mobile viewport and 1440px desktop viewport with zero horizontal overflow.

---

## 🛠️ Tech Stack

- **HTML5**: Semantic document structure with accessible ARIA landmarks.
- **CSS3**: Custom Qatar Airways 5-Star Luxury design system (`Inter`, system-ui font stack, Flexbox/Grid layouts, CSS backdrop filters, glassmorphic design tokens).
- **ES6 JavaScript**: Dynamic client-side inference engine, interactive console & explorer tabs, live dataset table search, code step switcher.
- **Python**: Built-in HTTP server (`python -m http.server`) for local development and review.

---

## 📊 Dataset & Model Summary

- **Raw Dataset**: 46,377 NTSB incident records ([`aviation.csv`](file:///c:/PaNDa/ACD/aviation.csv)).
- **Cleaned Dataset**: 39,017 quality U.S. aviation records ([`aviation_cleaned.csv`](file:///c:/PaNDa/ACD/aviation_cleaned.csv)).
- **Target Variable**: `HighestInjuryLevel` (`None`: 21,921 | `Fatal`: 6,859 | `Minor`: 5,981 | `Serious`: 4,648).
- **Winning ML Model**: **XGBoost Classifier** (Accuracy: `58.5%`, F1-Score: `0.5707`).
- **Jupyter Notebook**: [`av.ipynb`](file:///c:/PaNDa/ACD/av.ipynb).

---

## ⚠️ Important Disclaimer

> **Operational Limitation Notice:**
> AeroIntel is an exploratory product concept. It evaluates historical NTSB incident characteristics for analytical and educational purposes only. It does NOT predict specific future accidents, calculate flight crash probabilities, or serve as an operational flight-safety decision tool.

---

## 🚀 Local Development Setup

To run the AeroIntel homepage locally:

```bash
# Clone repository
git clone https://github.com/your-username/aerointel.git
cd aerointel

# Start Python HTTP server
python -m http.server 8000
```

Open `http://localhost:8000` in your web browser.
