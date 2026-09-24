# ============================================================
# TABS
# ============================================================

with gr.Tab("Thyroid Analysis"):

    # Patient input section
    gr.Markdown("## 🧑‍⚕️ Patient Information")
    gr.Markdown(
        "Enter the patient's clinical information below "
        "to generate a thyroid disease prediction."
    )

    input_components = []

    with gr.Row():
        for feature in loaded_features:
            component = gr.Number(
                label=feature,
                value=float(X.iloc[0][feature])
            )
            input_components.append(component)

    analyze_button = gr.Button(
        "🔍 Analyze Patient",
        elem_classes="primary-button"
    )

    reset_button = gr.Button("🔄 Reset")

    # Results
    gr.Markdown("## 📊 Prediction Result")

    result_output = gr.HTML()

    probability_output = gr.HTML()

    gr.Markdown("## 🧠 Counterfactual Explainable AI")

    xai_output = gr.HTML()

    # Button actions
    analyze_button.click(
        fn=thyroid_app,
        inputs=input_components,
        outputs=[
            result_output,
            probability_output,
            xai_output
        ]
    )

    reset_button.click(
        fn=reset_values,
        inputs=[],
        outputs=input_components
    )


# ============================================================
# FOOTER
# ============================================================

gr.HTML(
    """
    <div class="footer">
        ThyroCare AI — Thyroid Disease Prediction & Counterfactual XAI
        <br>
        This system is for educational and research purposes only.
        It is not a substitute for professional medical diagnosis.
    </div>
    """
)


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    app.launch(
        share=True,
        debug=True
    )
