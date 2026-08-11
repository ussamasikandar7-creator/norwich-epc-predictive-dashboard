import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostClassifier, Pool


# ============================================================
# FILE PATHS
# ============================================================

BASE = Path(__file__).resolve().parent


# ============================================================
# LOAD DATA AND SAVED MODEL OUTPUTS
# ============================================================

df = pd.read_csv(
    BASE / "analysis_data.csv",
    parse_dates=["LODGEMENT_DATE"]
)

metrics = pd.read_json(BASE / "model_metrics.json")

shap_df = pd.read_csv(BASE / "shap_importance.csv")

rec_df = pd.read_csv(BASE / "recommendation_summary.csv")

th = pd.read_csv(BASE / "threshold_sensitivity.csv")


with open(BASE / "model_metadata.json") as f:
    meta = json.load(f)


with open(BASE / "recommendation_lookup.json") as f:
    rec_lookup = json.load(f)


model = CatBoostClassifier()

model.load_model(
    str(BASE / "epc_catboost_model.cbm")
)


# ============================================================
# FRIENDLY FEATURE NAMES
# ============================================================

pretty = {
    "WALL_CATEGORY": "Wall insulation",
    "ROOF_CATEGORY": "Roof insulation",
    "PROPERTY_TYPE": "Property type",
    "AGE_BAND_CLEAN": "Construction age",
    "BUILT_FORM": "Built form",
    "HEATING_CATEGORY": "Heating system",
    "LOW_ENERGY_LIGHTING": "Low-energy lighting",
    "FLOOR_HEIGHT": "Floor height",
    "WINDOW_CATEGORY": "Glazing/windows",
    "MAIN_FUEL_CAT": "Main fuel",
    "TOTAL_FLOOR_AREA": "Floor area",
    "NUMBER_HABITABLE_ROOMS": "Habitable rooms",
    "REPORT_TYPE": "Assessment type",
    "PHOTO_SUPPLY": "PV roof-area share",
    "MECHANICAL_VENTILATION": "Ventilation",
    "SOLAR_WATER_HEATING_FLAG": "Solar water heating",
    "GLAZED_AREA": "Glazed area"
}


# ============================================================
# OVERVIEW CHARTS
# ============================================================

ratings = (
    df["CURRENT_ENERGY_RATING"]
    .value_counts()
    .reindex(list("ABCDEFG"), fill_value=0)
    .reset_index()
)

ratings.columns = ["Rating", "Properties"]


fig_ratings = px.bar(
    ratings,
    x="Rating",
    y="Properties",
    title="Current EPC rating distribution"
)


age_order = [
    "before 1900",
    "1900-1929",
    "1930-1949",
    "1950-1966",
    "1967-1975",
    "1976-1982",
    "1983-1990",
    "1991-1995",
    "1996-2002",
    "2003-2006",
    "2007-2011",
    "2012 onwards"
]


age = (
    df.groupby("AGE_BAND_CLEAN")["below_c"]
    .mean()
    .mul(100)
    .reindex(age_order)
    .dropna()
    .reset_index()
)

age.columns = [
    "Construction age",
    "Below C %"
]


fig_age = px.bar(
    age,
    x="Below C %",
    y="Construction age",
    orientation="h",
    title="Below-C share by construction age"
)

fig_age.update_yaxes(
    categoryorder="array",
    categoryarray=age_order[::-1]
)


# ============================================================
# MODEL PERFORMANCE CHARTS
# ============================================================

melt = metrics.melt(
    id_vars="Model",
    value_vars=[
        "Accuracy",
        "F1",
        "ROC_AUC"
    ],
    var_name="Metric",
    value_name="Score"
)


fig_models = px.bar(
    melt,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    title="Predictive model comparison",
    range_y=[0.65, 1.0]
)


fig_threshold = go.Figure()


for c in [
    "Precision",
    "Recall",
    "F1"
]:
    fig_threshold.add_trace(
        go.Scatter(
            x=th["Threshold"],
            y=th[c],
            mode="lines+markers",
            name=c
        )
    )


fig_threshold.update_layout(
    title="CatBoost decision-threshold trade-off",
    xaxis_title="Probability threshold",
    yaxis_title="Score",
    yaxis_range=[0.55, 1.0]
)


# ============================================================
# SHAP EXPLAINABILITY CHART
# ============================================================

sh = shap_df.head(12).copy()

sh["Feature"] = sh["Feature"].map(
    lambda x: pretty.get(x, x)
)

sh = sh.sort_values(
    "MeanAbsSHAP"
)


fig_shap = px.bar(
    sh,
    x="MeanAbsSHAP",
    y="Feature",
    orientation="h",
    title="Global SHAP feature importance"
)


# ============================================================
# RETROFIT RECOMMENDATION CHART
# ============================================================

rec_plot = (
    rec_df
    .head(10)
    .sort_values("Count")
)


fig_rec = px.bar(
    rec_plot,
    x="Count",
    y="Recommendation",
    orientation="h",
    title="Most common recommendations among below-C properties"
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(
    property_type,
    built_form,
    age_band,
    floor_area,
    hab_rooms,
    fuel,
    wall,
    roof,
    window,
    heating,
    low_light,
    threshold,
    glazed_area,
    ventilation,
    report_type,
    solar,
    pv,
    floor_height
):

    row = {
        "PROPERTY_TYPE": property_type,
        "BUILT_FORM": built_form,
        "AGE_BAND_CLEAN": age_band,
        "TOTAL_FLOOR_AREA": float(floor_area),
        "NUMBER_HABITABLE_ROOMS": float(hab_rooms),
        "MAIN_FUEL_CAT": fuel,
        "GLAZED_AREA": glazed_area,
        "WALL_CATEGORY": wall,
        "ROOF_CATEGORY": roof,
        "WINDOW_CATEGORY": window,
        "HEATING_CATEGORY": heating,
        "LOW_ENERGY_LIGHTING": float(low_light),
        "MECHANICAL_VENTILATION": ventilation,
        "REPORT_TYPE": report_type,
        "PHOTO_SUPPLY": float(pv),
        "SOLAR_WATER_HEATING_FLAG": solar,
        "FLOOR_HEIGHT": float(floor_height)
    }


    x = pd.DataFrame(
        [row]
    )[meta["feature_cols"]]


    # Fill numerical missing values
    for c in meta["num_cols"]:

        x[c] = pd.to_numeric(
            x[c],
            errors="coerce"
        )

        x[c] = x[c].fillna(
            meta["medians"][c]
        )


    # Predicted probability of being below EPC C
    prob = float(
        model.predict_proba(x)[:, 1][0]
    )


    # User-friendly risk level
    if prob >= 0.70:

        risk = "HIGH"

    elif prob >= 0.40:

        risk = "MEDIUM"

    else:

        risk = "LOW"


    # Business screening threshold
    if prob >= float(threshold):

        flag = "FLAG FOR REVIEW"

    else:

        flag = "PASS SCREEN"


    # ========================================================
    # LOCAL SHAP EXPLANATION
    # ========================================================

    pool = Pool(
        x,
        cat_features=[
            x.columns.get_loc(c)
            for c in meta["cat_cols"]
        ]
    )


    sv = model.get_feature_importance(
        pool,
        type="ShapValues"
    )[0, :-1]


    ex = pd.DataFrame(
        {
            "Feature": [
                pretty.get(c, c)
                for c in meta["feature_cols"]
            ],
            "SHAP contribution": sv
        }
    )


    ex = ex.reindex(
        ex[
            "SHAP contribution"
        ]
        .abs()
        .sort_values(
            ascending=False
        )
        .index
    )


    ex = (
        ex
        .head(6)
        .round(3)
    )


    # ========================================================
    # COMPARABLE RETROFIT RECOMMENDATIONS
    # ========================================================

    key = (
        f"{property_type}"
        f"|||{age_band}"
        f"|||{wall}"
    )


    rr = rec_lookup.get(
        key,
        rec_lookup["__GLOBAL__"]
    )[:5]


    rec_text = "\n".join(
        [
            f"• {r['text']} — observed {r['count']:,} times"
            for r in rr
        ]
    )


    # ========================================================
    # OUTPUT SUMMARY
    # ========================================================

    summary = f"""
### Probability below EPC C: **{prob * 100:.1f}%**

### Risk band: **{risk}**

### Screening result: **{flag}**

*Screening prototype only; not a certified EPC assessment.*
"""


    return (
        summary,
        ex,
        rec_text
    )


# ============================================================
# DASHBOARD CSS
# ============================================================

css = """
.gradio-container {
    max-width: 1500px !important;
    margin: auto !important;
}

#hero {
    padding: 8px 0 2px 0;
}
"""


# ============================================================
# GRADIO DASHBOARD
# ============================================================

with gr.Blocks(
    title="Norwich EPC Predictive Analytics Dashboard",
    css=css
) as demo:


    gr.Markdown(
        "# Norwich EPC Predictive Analytics Dashboard",
        elem_id="hero"
    )


    gr.Markdown(
        "**2022–2024 reassessment dataset | "
        "Latest certificate per property | "
        "Screening tool, not a certified EPC assessment**"
    )


    with gr.Tabs():


        # ====================================================
        # TAB 1 — OVERVIEW
        # ====================================================

        with gr.Tab("Overview"):


            with gr.Row():

                gr.Markdown(
                    f"""
### Properties analysed
## {len(df):,}
"""
                )


                gr.Markdown(
                    f"""
### Below EPC C
## {df['below_c'].mean() * 100:.1f}%
"""
                )


                gr.Markdown(
                    """
### Median current energy-efficiency score
## 70
"""
                )


                gr.Markdown(
                    """
### Study window
## 2022–2024
"""
                )


            with gr.Row():

                gr.Plot(
                    fig_ratings
                )

                gr.Plot(
                    fig_age
                )


            gr.Markdown(
                """
### Descriptive insight

Older properties and weak building fabric show the
highest below-C rates.

Postcode and tenure are retained for descriptive
segmentation but excluded from the predictive model
to reduce proxy dependence.
"""
            )


        # ====================================================
        # TAB 2 — MODEL PERFORMANCE
        # ====================================================

        with gr.Tab(
            "Model Performance"
        ):


            gr.Dataframe(
                metrics[
                    [
                        "Model",
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1",
                        "ROC_AUC",
                        "PR_AUC"
                    ]
                ].round(3),
                interactive=False
            )


            with gr.Row():

                gr.Plot(
                    fig_models
                )

                gr.Plot(
                    fig_threshold
                )


            gr.Markdown(
                f"""
**Temporal robustness:** training on 2022–2023 and
testing on 2024 produced ROC-AUC
**{meta['temporal_metrics']['ROC_AUC']:.3f}**,
lower than the random holdout result.

This supports monitoring and retraining rather than
assuming a static model.
"""
            )


        # ====================================================
        # TAB 3 — EXPLAINABLE AI
        # ====================================================

        with gr.Tab(
            "Explainable AI"
        ):


            with gr.Row():

                gr.Plot(
                    fig_shap
                )

                gr.Plot(
                    fig_rec
                )


            gr.Markdown(
                """
**SHAP interpretation:** larger absolute values mean
the feature changes the model prediction more strongly
on average.

These explanations describe model behaviour and
should not be interpreted as causal effects.
"""
            )


        # ====================================================
        # TAB 4 — RISK PREDICTOR
        # ====================================================

        with gr.Tab(
            "Risk Predictor"
        ):


            with gr.Row():


                property_type = gr.Dropdown(
                    meta["categories"]["PROPERTY_TYPE"],
                    value="House",
                    label="Property type"
                )


                built_form = gr.Dropdown(
                    meta["categories"]["BUILT_FORM"],
                    value=meta[
                        "categories"
                    ]["BUILT_FORM"][0],
                    label="Built form"
                )


                age_band = gr.Dropdown(
                    meta["categories"]["AGE_BAND_CLEAN"],
                    value="1900-1929",
                    label="Construction age"
                )


                floor_area = gr.Number(
                    value=85,
                    label="Total floor area (m²)"
                )


            with gr.Row():


                wall = gr.Dropdown(
                    meta["categories"]["WALL_CATEGORY"],
                    value="No insulation",
                    label="Wall insulation"
                )


                roof = gr.Dropdown(
                    meta["categories"]["ROOF_CATEGORY"],
                    value="100-199 mm",
                    label="Roof insulation"
                )


                window = gr.Dropdown(
                    meta["categories"]["WINDOW_CATEGORY"],
                    value="Full double glazing",
                    label="Glazing/windows"
                )


                heating = gr.Dropdown(
                    meta["categories"]["HEATING_CATEGORY"],
                    value="Gas boiler",
                    label="Heating system"
                )


            with gr.Row():


                hab_rooms = gr.Number(
                    value=4,
                    label="Habitable rooms"
                )


                fuel = gr.Dropdown(
                    meta["categories"]["MAIN_FUEL_CAT"],
                    value="Mains gas",
                    label="Main fuel"
                )


                low_light = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=80,
                    step=1,
                    label="Low-energy lighting (%)"
                )


                threshold = gr.Slider(
                    minimum=0.3,
                    maximum=0.7,
                    value=0.5,
                    step=0.05,
                    label="Screening threshold"
                )


            # =================================================
            # ADVANCED INPUTS
            # =================================================

            with gr.Accordion(
                "Advanced assumptions",
                open=False
            ):


                with gr.Row():


                    glazed_area = gr.Dropdown(
                        meta["categories"]["GLAZED_AREA"],
                        value=(
                            "Normal"
                            if "Normal"
                            in meta[
                                "categories"
                            ]["GLAZED_AREA"]
                            else meta[
                                "categories"
                            ]["GLAZED_AREA"][0]
                        ),
                        label="Glazed area"
                    )


                    ventilation = gr.Dropdown(
                        meta[
                            "categories"
                        ]["MECHANICAL_VENTILATION"],
                        value=(
                            "natural"
                            if "natural"
                            in meta[
                                "categories"
                            ][
                                "MECHANICAL_VENTILATION"
                            ]
                            else meta[
                                "categories"
                            ][
                                "MECHANICAL_VENTILATION"
                            ][0]
                        ),
                        label="Ventilation"
                    )


                    report_type = gr.Dropdown(
                        meta[
                            "categories"
                        ]["REPORT_TYPE"],
                        value="100",
                        label="Assessment type"
                    )


                    solar = gr.Dropdown(
                        meta[
                            "categories"
                        ][
                            "SOLAR_WATER_HEATING_FLAG"
                        ],
                        value=(
                            "N"
                            if "N"
                            in meta[
                                "categories"
                            ][
                                "SOLAR_WATER_HEATING_FLAG"
                            ]
                            else meta[
                                "categories"
                            ][
                                "SOLAR_WATER_HEATING_FLAG"
                            ][0]
                        ),
                        label="Solar water heating"
                    )


                    pv = gr.Number(
                        value=meta[
                            "medians"
                        ]["PHOTO_SUPPLY"],
                        label="PV roof-area share (%)"
                    )


                    floor_height = gr.Number(
                        value=meta[
                            "medians"
                        ]["FLOOR_HEIGHT"],
                        label="Floor height (m)"
                    )


            # =================================================
            # PREDICT BUTTON
            # =================================================

            btn = gr.Button(
                "Generate risk assessment",
                variant="primary"
            )


            with gr.Row():


                summary = gr.Markdown()


                explanation = gr.Dataframe(
                    headers=[
                        "Feature",
                        "SHAP contribution"
                    ],
                    interactive=False,
                    label="Main drivers"
                )


            recs = gr.Textbox(
                label=(
                    "Common recommendations in "
                    "comparable below-C records"
                ),
                lines=6
            )


            btn.click(
                fn=predict,
                inputs=[
                    property_type,
                    built_form,
                    age_band,
                    floor_area,
                    hab_rooms,
                    fuel,
                    wall,
                    roof,
                    window,
                    heating,
                    low_light,
                    threshold,
                    glazed_area,
                    ventilation,
                    report_type,
                    solar,
                    pv,
                    floor_height
                ],
                outputs=[
                    summary,
                    explanation,
                    recs
                ]
            )


# ============================================================
# RENDER / LOCAL DEPLOYMENT
# ============================================================

if __name__ == "__main__":

    # Render automatically provides the PORT variable.
    # When running locally, the app defaults to port 7860.
    port = int(
        os.environ.get(
            "PORT",
            7860
        )
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
