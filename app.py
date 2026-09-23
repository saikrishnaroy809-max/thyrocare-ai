
import gradio as gr
import pandas as pd
import joblib
import os

# ==============================
# Load Model Files
# ==============================

MODEL_FILE = "thyroid_model.pkl"
SCALER_FILE = "thyroid_scaler.pkl"
FEATURES_FILE = "thyroid_features.pkl"
DATASET_FILE = "cleaned_dataset_Thyroid1.csv"

loaded_model = joblib.load(MODEL_FILE)
loaded_scaler = joblib.load(SCALER_FILE)
loaded_features = joblib.load(FEATURES_FILE)

X = pd.read_csv(DATASET_FILE)

# ==============================
# Prediction Function
# ==============================

def final_prediction(patient_data):

    patient_df = pd.DataFrame(
        [patient_data],
        columns=loaded_features
    )

    patient_scaled = loaded_scaler.transform(patient_df)

    prediction = loaded_model.predict(patient_scaled)[0]
    probabilities = loaded_model.predict_proba(patient_scaled)[0]

    confidence = probabilities[prediction] * 100

    if prediction == 1:
        result = "Thyroid disease detected"
    else:
        result = "No thyroid disease detected"

    return result, confidence, probabilities


# ==============================
# Counterfactual XAI
# ==============================

def counterfactual_explanation(patient_data):

    patient_df = pd.DataFrame(
        [patient_data],
        columns=loaded_features
    )

    patient_scaled = loaded_scaler.transform(patient_df)

    original_prediction = loaded_model.predict(patient_scaled)[0]

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


# ==============================
# Main Application
# ==============================

def thyroid_app(*values):

    patient_data = list(values)

    result, confidence, probabilities = final_prediction(
        patient_data
    )

    original_prediction, counterfactuals = \
        counterfactual_explanation(patient_data)

    if original_prediction == 1:

        result_text = """
## ⚠️ Thyroid Disease Detected
"""

    else:

        result_text = """
## ✅ No Thyroid Disease Detected
"""

    probability_text = f"""
### 📊 Prediction Probabilities

**No Thyroid Disease:** `{probabilities[0] * 100:.2f}%`

**Thyroid Disease:** `{probabilities[1] * 100:.2f}%`

**Model Confidence:** `{confidence:.2f}%`
"""

    if counterfactuals:

        xai_text = "### 🧠 Counterfactual Explanation\n\n"

        for cf in counterfactuals:

            xai_text += f"""
**Feature:** `{cf['Feature']}`

**Original Value:** `{cf['Original Value']}`

**Counterfactual Value:** `{cf['New Value']}`

**Prediction Change:** `{cf['Original Prediction']}` → `{cf['New Prediction']}`

---

"""

    else:

        xai_text = """
### 🧠 Counterfactual Explanation

No single-feature counterfactual change was found
among the tested feature values.

This explanation describes the machine-learning
model's behavior and is not a medical diagnosis.
"""

    return result_text, probability_text, xai_text


# ==============================
# User Interface
# ==============================

custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
}

.title {
    text-align: center;
    padding: 20px;
}

.subtitle {
    text-align: center;
    color: #667085;
    margin-bottom: 25px;
}

.disclaimer {
    text-align: center;
    color: #667085;
    font-size: 13px;
    padding: 20px;
}
"""


with gr.Blocks(
    title="ThyroCare AI"
) as app:

    gr.Markdown(
        """
<div class="title">

# 🦋 ThyroCare AI

### Thyroid Disease Prediction & Counterfactual Explainable AI

<div class="subtitle">
Machine Learning based academic research application
</div>

</div>
"""
    )

    with gr.Row():

        with gr.Column():

            gr.Markdown(
                "## 👤 Patient Information"
            )

            inputs = []

            for feature in loaded_features:

                component = gr.Number(
                    label=feature,
                    value=float(X.iloc[0][feature])
                )

                inputs.append(component)

            predict_button = gr.Button(
                "🔍 Analyze Patient",
                variant="primary"
            )

        with gr.Column():

            gr.Markdown(
                "## 📊 Analysis Result"
            )

            result_output = gr.Markdown(
                "Enter patient information and click **Analyze Patient**."
            )

            probability_output = gr.Markdown(
                "Prediction probabilities will appear here."
            )

            gr.Markdown(
                "## 🧠 Explainable AI"
            )

            xai_output = gr.Markdown(
                "Counterfactual explanation will appear here."
            )

    gr.Markdown(
        """
<div class="disclaimer">

⚠️ <b>Academic Use Only</b><br>

This application is developed for educational and research
purposes. It is not intended to replace professional medical
diagnosis or treatment.

</div>
"""
    )

    predict_button.click(
        fn=thyroid_app,
        inputs=inputs,
        outputs=[
            result_output,
            probability_output,
            xai_output
        ]
    )


print("✅ ThyroCare AI application created successfully!")
