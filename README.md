---
title: Norwich EPC Predictive Analytics Dashboard
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Norwich EPC Predictive Analytics Dashboard

Interactive screening prototype supporting the NBS-7096B R001 reassessment. The app uses the saved CatBoost model and privacy-reduced analytical outputs; raw addresses, UPRNs and LMK keys are not included in this deployment package.

The tool estimates the probability that a Norwich property falls below EPC Band C and provides SHAP-based model explanations plus historical recommendation patterns. It is a screening prototype, not a certified EPC assessment or personalised engineering recommendation.
