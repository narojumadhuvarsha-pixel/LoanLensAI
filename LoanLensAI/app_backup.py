from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("loan_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        applicant = pd.DataFrame([{
            "Age": int(data["age"]),
            "MonthlyIncome": float(data["monthly_income"]),
            "CreditScore": int(data["credit_score"]),
            "ExistingLoan": float(data["existing_loan"]),
            "LoanAmount": float(data["loan_amount"]),
            "LoanTerm": int(data["loan_term"]),
            "Dependents": int(data["dependents"]),
            "EmploymentStatus": data["employment_status"],
            "EmploymentYears": int(data["employment_years"]),
            "PaymentHistory": data["payment_history"]
        }])

        # Prediction
        prediction = model.predict(applicant)[0]

        # Probability
        probabilities = model.predict_proba(applicant)[0]
        confidence = max(probabilities) * 100

        if prediction == 1:
            eligibility = "Likely Eligible"
        else:
            eligibility = "Likely Not Eligible"

        # Risk calculation
        credit_score = int(data["credit_score"])
        income = float(data["monthly_income"])
        loan_amount = float(data["loan_amount"])
        existing_loan = float(data["existing_loan"])
        payment_history = data["payment_history"]

        risk_score = 0

        if credit_score < 600:
            risk_score += 35
        elif credit_score < 700:
            risk_score += 20
        else:
            risk_score += 5

        if payment_history == "Poor":
            risk_score += 30
        elif payment_history == "Average":
            risk_score += 15
        else:
            risk_score += 5

        if existing_loan > income * 10:
            risk_score += 20

        if loan_amount > income * 20:
            risk_score += 15

        risk_score = min(risk_score, 100)

        if risk_score < 30:
            risk_level = "Low"
        elif risk_score < 60:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Generate key factors
        factors = []

        if credit_score >= 700:
            factors.append("Strong credit score")
        elif credit_score < 600:
            factors.append("Low credit score")

        if payment_history == "Good":
            factors.append("Good payment history")
        elif payment_history == "Poor":
            factors.append("Poor payment history")

        if income >= 50000:
            factors.append("Strong monthly income")

        if existing_loan > income * 10:
            factors.append("High existing debt")

        if loan_amount > income * 20:
            factors.append("High requested loan amount")

        if not factors:
            factors.append("Multiple financial factors considered")

        return jsonify({
            "eligibility": eligibility,
            "confidence": round(confidence, 2),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "factors": factors
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)