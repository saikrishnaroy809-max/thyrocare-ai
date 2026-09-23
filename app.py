import gradio as gr
import pandas as pd
import joblib
import os

# ============================================================
# THYROCARE AI - ADVANCED DASHBOARD
# ============================================================

MODEL_FILE = "thyroid_model.pkl"
SCALER_FILE = "thyroid_scaler.pkl"
FEATURES_FILE = "thyroid_features.pkl"
DATASET_FILE = "cleaned_dataset_Thyroid1.csv"


# ============================================================
# LOAD MODEL
# ============================================================

loaded_model = joblib.load(MODEL_FILE)
loaded_scaler = joblib.load(SCALER_FILE)
loaded_features = joblib.load(FEATURES_FILE)

X = pd.read_csv(DATASET_FILE)

print("✅ ThyroCare AI model loaded")
print("✅ Dataset loaded")
print(f"✅ Number of features: {len(loaded_features)}")


# ============================================================
# PREDICTION
# ============================================================

def final_prediction(patient_data):

    patient_df = pd.DataFrame(
        [patient_data],
        columns=loaded_features
    )

    patient_scaled = loaded_scaler.transform(patient_df)

    prediction = loaded_model.predict(patient_scaled)[0]

    probabilities = loaded_model.predict_proba(
        patient_scaled
    )[0]

    confidence = probabilities[int(prediction)] * 100

    if prediction == 1:
        result = "Thyroid Disease Detected"
    else:
        result = "No Thyroid Disease Detected"

    return result, confidence, probabilities


# ============================================================
# COUNTERFACTUAL XAI
# ============================================================

def counterfactual_explanation(patient_data):

    patient_df = pd.DataFrame(
        [patient_data],
        columns=loaded_features
    )

    patient_scaled = loaded_scaler.transform(patient_df)

    original_prediction = loaded_model.predict(
        patient_scaled
    )[0]

    counterfactuals = []

    for feature in loaded_features:

        original_value = patient_df.iloc[0][feature]

        possible_values = X[feature].dropna().unique()

        for new_value in possible_values:

            modified_patient = patient_df.copy()

            modified_patient.at[0, feature] = new_value

            modified_scaled = loaded_scaler.transform(
                modified_patient
            )

            new_prediction = loaded_model.predict(
                modified_scaled
            )[0]

            if new_prediction != original_prediction:

                counterfactuals.append({
                    "Feature": feature,
                    "Original Value": original_value,
                    "New Value": new_value,
                    "Original Prediction": original_prediction,
                    "New Prediction": new_prediction
                })

                break

    return original_prediction, counterfactuals


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def thyroid_app(*values):

    try:

        patient_data = list(values)

        result, confidence, probabilities = final_prediction(
            patient_data
        )

        original_prediction, counterfactuals = (
            counterfactual_explanation(patient_data)
        )

        no_disease = probabilities[0] * 100
        disease = probabilities[1] * 100

        # ----------------------------------------------------
        # RESULT CARD
        # ----------------------------------------------------

        if original_prediction == 1:

            result_html = """
            <div class="result-card danger">
                <div class="result-icon">⚠️</div>
                <div>
                    <div class="result-label">MODEL PREDICTION</div>
                    <div class="result-title">
                        Thyroid Disease Detected
                    </div>
                    <div class="result-subtitle">
                        The model predicts the thyroid-disease class.
                    </div>
                </div>
            </div>
            """

        else:

            result_html = """
            <div class="result-card success">
                <div class="result-icon">✓</div>
                <div>
                    <div class="result-label">MODEL PREDICTION</div>
                    <div class="result-title">
                        No Thyroid Disease Detected
                    </div>
                    <div class="result-subtitle">
                        The model predicts the no-thyroid-disease class.
                    </div>
                </div>
            </div>
            """

        # ----------------------------------------------------
        # PROBABILITY DASHBOARD
        # ----------------------------------------------------

        probability_html = f"""
        <div class="probability-container">

            <div class="probability-header">
                <span>Prediction Analysis</span>
                <span class="confidence-badge">
                    {confidence:.2f}% confidence
                </span>
            </div>

            <div class="prob-row">
                <div class="prob-title">
                    <span>No Thyroid Disease</span>
                    <strong>{no_disease:.2f}%</strong>
                </div>

                <div class="progress-bg">
                    <div class="progress-fill no-disease"
                         style="width:{no_disease}%;">
                    </div>
                </div>
            </div>

            <div class="prob-row">
                <div class="prob-title">
                    <span>Thyroid Disease</span>
                    <strong>{disease:.2f}%</strong>
                </div>

                <div class="progress-bg">
                    <div class="progress-fill disease"
                         style="width:{disease}%;">
                    </div>
                </div>
            </div>

        </div>
        """

        # ----------------------------------------------------
        # XAI
        # ----------------------------------------------------

        if counterfactuals:

            xai_html = """
            <div class="xai-intro">
                <div class="xai-icon">🧠</div>
                <div>
                    <b>Counterfactual Explanation</b>
                    <p>
                    The following feature changes were tested to
                    identify conditions under which the model
                    prediction changes.
                    </p>
                </div>
            </div>
            """

            for cf in counterfactuals:

                xai_html += f"""
                <div class="xai-card">

                    <div class="xai-feature">
                        🔬 {cf['Feature']}
                    </div>

                    <div class="xai-grid">

                        <div>
                            <span>Original Value</span>
                            <strong>{cf['Original Value']}</strong>
                        </div>

                        <div>
                            <span>Counterfactual Value</span>
                            <strong>{cf['New Value']}</strong>
                        </div>

                        <div>
                            <span>Original Class</span>
                            <strong>{cf['Original Prediction']}</strong>
                        </div>

                        <div>
                            <span>New Class</span>
                            <strong>{cf['New Prediction']}</strong>
                        </div>

                    </div>

                </div>
                """

        else:

            xai_html = """
            <div class="xai-empty">

                <div class="xai-empty-icon">🧠</div>

                <h3>No Single-Feature Counterfactual Found</h3>

                <p>
                No single-feature change among the tested dataset
                values changed the current model prediction.
                </p>

                <small>
                This describes machine-learning model behavior
                and is not a medical diagnosis.
                </small>

            </div>
            """

        return result_html, probability_html, xai_html

    except Exception as e:

        return (
            f"""
            <div class="error-card">
                ❌ Analysis Error
                <br><br>
                {str(e)}
            </div>
            """,
            "",
            ""
        )


# ============================================================
# RESET FUNCTION
# ============================================================

def reset_values():

    return [
        float(X.iloc[0][feature])
        for feature in loaded_features
    ]


# ============================================================
# ADVANCED CSS
# ============================================================

custom_css = """

/* =========================================================
   GLOBAL
   ========================================================= */

body {
    background:
        radial-gradient(
            circle at top right,
            #173b52 0%,
            #081923 35%,
            #061018 75%
        ) !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
    background: transparent !important;
    color: #eaf7fb !important;
}


/* =========================================================
   HEADER
   ========================================================= */

.hero {
    padding: 35px;
    border-radius: 24px;
    margin-bottom: 24px;

    background:
        linear-gradient(
            135deg,
            rgba(15, 70, 90, 0.95),
            rgba(5, 28, 40, 0.98)
        );

    border: 1px solid rgba(66, 211, 225, 0.25);

    box-shadow:
        0 15px 45px rgba(0, 0, 0, 0.30);
}

.hero-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
}

.brand {
    font-size: 38px;
    font-weight: 800;
    letter-spacing: -1px;
}

.brand span {
    color: #43d7df;
}

.hero-subtitle {
    color: #a8c4ce;
    font-size: 16px;
    margin-top: 8px;
}

.status {
    background: rgba(48, 210, 151, 0.12);
    color: #55e6ae;
    border: 1px solid rgba(85, 230, 174, 0.35);
    border-radius: 30px;
    padding: 10px 16px;
    font-size: 13px;
    font-weight: 700;
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

.metric-card {
    background: rgba(13, 34, 46, 0.92);
    border: 1px solid rgba(100, 190, 210, 0.15);
    border-radius: 18px;
    padding: 20px;
    min-height: 100px;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.18);
}

.metric-icon {
    font-size: 25px;
}

.metric-title {
    color: #8ca9b4;
    font-size: 12px;
    margin-top: 7px;
}

.metric-value {
    font-size: 21px;
    font-weight: 800;
    margin-top: 4px;
    color: #e9fbff;
}


/* =========================================================
   SECTION HEADERS
   ========================================================= */

.section-header {
    font-size: 23px;
    font-weight: 800;
    margin: 15px 0 8px 0;
}

.section-description {
    color: #91abb5;
    margin-bottom: 18px;
}


/* =========================================================
   PANELS
   ========================================================= */

.panel {
    background: rgba(9, 27, 38, 0.92);
    border: 1px solid rgba(91, 187, 205, 0.14);
    border-radius: 20px;
    padding: 22px;
}


/* =========================================================
   INPUTS
   ========================================================= */

input {
    background: #0d2430 !important;
    color: #eafaff !important;
    border: 1px solid #214653 !important;
    border-radius: 10px !important;
}

label {
    color: #b7d2da !important;
}


/* =========================================================
   BUTTON
   ========================================================= */

.primary-button {
    background:
        linear-gradient(
            135deg,
            #19bfc9,
            #287ce0
        ) !important;

    color: white !important;

    border: none !important;

    border-radius: 13px !important;

    font-weight: 800 !important;

    min-height: 50px !important;

    box-shadow:
        0 8px 25px rgba(30, 184, 207, 0.25);
}


/* =========================================================
   RESULT CARD
   ========================================================= */

.result-card {
    display: flex;
    align-items: center;
    gap: 18px;

    padding: 22px;

    border-radius: 18px;

    margin-bottom: 18px;
}

.result-card.success {
    background:
        linear-gradient(
            135deg,
            rgba(24, 140, 108, 0.25),
            rgba(10, 53, 52, 0.65)
        );

    border: 1px solid rgba(62, 218, 168, 0.35);
}

.result-card.danger {
    background:
        linear-gradient(
            135deg,
            rgba(190, 67, 76, 0.25),
            rgba(65, 24, 32, 0.65)
        );

    border: 1px solid rgba(244, 112, 123, 0.35);
}

.result-icon {
    width: 55px;
    height: 55px;

    border-radius: 16px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 28px;

    background: rgba(255,255,255,0.08);
}

.result-label {
    color: #91adb7;
    font-size: 11px;
    font-weight: 700;
}

.result-title {
    font-size: 22px;
    font-weight: 800;
    margin-top: 3px;
}

.result-subtitle {
    color: #9eb6bf;
    margin-top: 4px;
}


/* =========================================================
   PROBABILITY
   ========================================================= */

.probability-container {
    background: #0a202c;
    border: 1px solid #193b48;
    border-radius: 18px;
    padding: 22px;
}

.probability-header {
    display: flex;
    justify-content: space-between;
    align-items: center;

    font-size: 18px;
    font-weight: 800;

    margin-bottom: 22px;
}

.confidence-badge {
    background: rgba(48, 210, 151, 0.13);
    color: #58e4ad;
    border: 1px solid rgba(88, 228, 173, 0.25);
    padding: 7px 12px;
    border-radius: 20px;
    font-size: 12px;
}

.prob-row {
    margin-bottom: 20px;
}

.prob-title {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    color: #b8d0d8;
}

.progress-bg {
    width: 100%;
    height: 10px;
    background: #122f3b;
    border-radius: 20px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 20px;
}

.progress-fill.no-disease {
    background: linear-gradient(
        90deg,
        #27c7b2,
        #52e3b1
    );
}

.progress-fill.disease {
    background: linear-gradient(
        90deg,
        #e56b78,
        #ff9a73
    );
}


/* =========================================================
   XAI
   ========================================================= */

.xai-intro {
    display: flex;
    gap: 15px;
    align-items: flex-start;

    padding: 20px;

    background:
        linear-gradient(
            135deg,
            rgba(45, 104, 180, 0.20),
            rgba(27, 65, 102, 0.35)
        );

    border: 1px solid rgba(86, 159, 229, 0.22);

    border-radius: 17px;

    margin-bottom: 15px;
}

.xai-icon {
    font-size: 30px;
}

.xai-intro p {
    color: #9eb7c1;
    margin: 5px 0 0 0;
}

.xai-card {
    background: #0a202c;
    border: 1px solid #193c4a;
    border-radius: 17px;
    padding: 20px;
    margin-bottom: 15px;
}

.xai-feature {
    font-size: 17px;
    font-weight: 800;
    color: #4ed8e0;
    margin-bottom: 17px;
}

.xai-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}

.xai-grid div {
    background: #102d39;
    border-radius: 12px;
    padding: 14px;
}

.xai-grid span {
    display: block;
    color: #819da7;
    font-size: 11px;
    margin-bottom: 6px;
}

.xai-grid strong {
    color: #e5f9fc;
}

.xai-empty {
    text-align: center;
    padding: 35px;

    background: #0a202c;
    border: 1px solid #193b48;
    border-radius: 18px;
}

.xai-empty-icon {
    font-size: 40px;
}

.xai-empty h3 {
    margin-bottom: 8px;
}

.xai-empty p {
    color: #91abb5;
}

.xai-empty small {
    color: #738e98;
}


/* =========================================================
   INFO CARDS
   ========================================================= */

.info-card {
    background: #0b202b;
    border: 1px solid #193a46;
    border-radius: 16px;
    padding: 20px;
    min-height: 130px;
}

.info-card h3 {
    margin-top: 0;
    color: #4ed8e0;
}

.info-card p {
    color: #91abb5;
    line-height: 1.6;
}


/* =========================================================
   ERROR
   ========================================================= */

.error-card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(180, 50, 60, 0.2);
    border: 1px solid rgba(240, 90, 100, 0.4);
    color: #ff9da5;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;
    padding: 30px 10px 10px 10px;
    color: #6e8993;
    font-size: 12px;
    line-height: 1.7;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 700px) {

    .hero {
        padding: 24px;
    }

    .brand {
        font-size: 29px;
    }

    .hero-top {
        display: block;
    }

    .status {
        display: inline-block;
        margin-top: 15px;
    }

    .xai-grid {
        grid-template-columns: repeat(2, 1fr);
    }

}

"""


# ============================================================
# USER INTERFACE
# ============================================================

with gr.Blocks(
    title="ThyroCare AI | Advanced Thyroid Analysis",
    css=custom_css,
    theme=gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate"
    )
) as app:

    # ========================================================
    # HERO HEADER
    # ========================================================

    gr.HTML(
        """
        <div class="hero">

            <div class="hero-top">

                <div>

                    <div class="brand">
                        🦋 <span>ThyroCare</span> AI
                    </div>

                    <div class="hero-subtitle">
                        Intelligent Thyroid Disease Prediction
                        & Counterfactual Explainable AI
                    </div>

                </div>

                <div class="status">
                    ● AI ENGINE ONLINE
                </div>

            </div>

        </div>
        """
    )


    # ========================================================
    # DASHBOARD METRICS
    # ========================================================

    with gr.Row():

        gr.HTML(
            """
            <div class="metric-card">

                <div class="metric-icon">🧠</div>

                <div class="metric-title">
                    ML MODEL
                </div>

                <div class="metric-value">
                    ACTIVE
                </div>

            </div>
            """
        )

        gr.HTML(
            f"""
            <div class="metric-card">

                <div class="metric-icon">🔢</div>

                <div class="metric-title">
                    FEATURES
                </div>

                <div class="metric-value">
                    {len(loaded_features)}
                </div>

            </div>
            """
        )

        gr.HTML(
            """
            <div class="metric-card">

                <div class="metric-icon">📊</div>

                <div class="metric-title">
                    PREDICTION ENGINE
                </div>

                <div class="metric-value">
                    READY
                </div>

            </div>
            """
        )

        gr.HTML(
            """
            <div class="metric-card">

                <div class="metric-icon">🔬</div>

                <div class="metric-title">
                    XAI ENGINE
                </div>

                <div class="metric-value">
                    ENABLED
                </div>

            </div>
            """
        )


    # ========================================================
    # TABS
    # ========================================================

    with gr.Tab

