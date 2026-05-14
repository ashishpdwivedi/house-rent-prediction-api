from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask("House Rent Prediction API")

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "rent_model.pkl")
model = joblib.load(MODEL_PATH)

print("Loaded model:", type(model))

@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        form_data={},
        error=None
    )

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form data
        user_input = {
            "Size": int(request.form["Size"]),
            "Area_sqft": int(request.form["Area_sqft"]),
            "Seller_type": request.form["Seller_type"],
            "Size_unit": request.form["Size_unit"],
            "Bathroom": int(request.form["Bathroom"]),
            "Location": request.form["Location"],
            "Property_type": request.form["Property_type"],
            "Status": request.form["Status"],
            "Facing_direction": request.form["Facing_direction"],
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([user_input])

        # Debug: print received input
        print("\nInput received from website:")
        print(input_df)

        print("Model path:", MODEL_PATH)
        print("Loaded model:", type(model))
        print("Pipeline steps:", model.named_steps)
        print("Random Forest parameters:", model.named_steps["model"])
                # Predict using fitted pipeline
        prediction = model.predict(input_df)[0]

        # Debug: print prediction
        print("Prediction:", prediction)

        # Return result to template
        return render_template(
            "index.html",
            prediction=f"{prediction:,.2f}",
            form_data=request.form,
            error=None
        )

    except Exception as e:
        # Print error in terminal
        print("Error:", str(e))

        # Return error message to page
        return render_template(
            "index.html",
            prediction=None,
            form_data=request.form,
            error=str(e)
        )


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)