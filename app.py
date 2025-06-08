from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import joblib  

# ========== Define the Model ==========
class ChurnMLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# ========== Initialize Flask ==========
app = Flask(__name__)
CORS(app)

# ========== Load Model and Assets ==========
model = torch.load("customer_churn_model.pt", map_location=torch.device("cpu"))
model.eval()

expected_columns = np.load("model_features.npy", allow_pickle=True)
scaler = joblib.load("scaler.pkl")  # load trained StandardScaler

# ========== Reason Generator ==========
def generate_reasons(row):
    reasons = []
    if row.get("tenure", 0) < 5:
        reasons.append("Short tenure")
    if "Month-to-month" in row.get("Contract", ""):
        reasons.append("Unstable contract")
    if "Electronic check" in row.get("PaymentMethod", ""):
        reasons.append("High-risk payment method")
    if row.get("MonthlyCharges", 0) > 80:
        reasons.append("High monthly charges")
    return reasons or ["No strong churn indicators"]

# ========== Cleanup ==========
def cleanup():
    for f in ["uploads/input.csv", "outputs/predicted_output.csv"]:
        if os.path.exists(f):
            os.remove(f)

# ========== Routes ==========
@app.route("/", methods=["GET"])
def home():
    return "✅ Churn Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        cleanup()
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        file.save("uploads/input.csv")
        df = pd.read_csv("uploads/input.csv")

        # Store original for reasoning
        original = df.copy()
        customer_ids = df["customerID"]
        df.drop("customerID", axis=1, inplace=True)

        # One-hot encode
        categorical_cols = df.select_dtypes(include="object").columns.tolist()
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

        # Align with expected features
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_columns]

        # Apply saved StandardScaler (trained on: tenure, MonthlyCharges, TotalCharges)
        scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df[scale_cols] = scaler.transform(df[scale_cols])  

        # Convert to tensor
        df = df.astype("float32")
        input_tensor = torch.tensor(df.values)

        # Predict
        with torch.no_grad():
            output = model(input_tensor).squeeze().numpy()

        results = []
        for i, prob in enumerate(output):
            churn = "Yes" if prob >= 0.5 else "No"
            reasons = ", ".join(generate_reasons(original.iloc[i]))
            results.append({
                "customerID": customer_ids.iloc[i],
                "Churn": churn,
                "Probability": f"{prob * 100:.2f}%",
                "Reason": reasons
            })

        pd.DataFrame(results).to_csv("outputs/predicted_output.csv", index=False)
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== Run Server ==========
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
