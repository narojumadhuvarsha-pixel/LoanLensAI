from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Load trained ML model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "loan_model.pkl")
)

# -----------------------------
# DATABASE
# -----------------------------

def init_db():

    conn = sqlite3.connect("loanlens.db")

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT,

            age INTEGER,

            monthly_income REAL,

            credit_score INTEGER,

            existing_loan REAL,

            loan_amount REAL,

            loan_term INTEGER,

            dependents INTEGER,

            employment_status TEXT,

            employment_years INTEGER,

            payment_history TEXT,

            eligibility TEXT,

            confidence REAL,

            risk_score INTEGER,

            risk_level TEXT

        )
    """)

    conn.commit()

    conn.close()


# Create database when application starts
init_db()


# -----------------------------
# HOME PAGE
# -----------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -----------------------------
# PREDICTION
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()


        # Applicant information
        applicant = pd.DataFrame([{

            "Age": int(data["age"]),

            "MonthlyIncome":
                float(data["monthly_income"]),

            "CreditScore":
                int(data["credit_score"]),

            "ExistingLoan":
                float(data["existing_loan"]),

            "LoanAmount":
                float(data["loan_amount"]),

            "LoanTerm":
                int(data["loan_term"]),

            "Dependents":
                int(data["dependents"]),

            "EmploymentStatus":
                data["employment_status"],

            "EmploymentYears":
                int(data["employment_years"]),

            "PaymentHistory":
                data["payment_history"]

        }])


        # -----------------------------
        # ML PREDICTION
        # -----------------------------

        prediction = model.predict(applicant)[0]

        probabilities = model.predict_proba(applicant)[0]

        confidence = max(probabilities) * 100


        if prediction == 1:

            eligibility = "Likely Eligible"

        else:

            eligibility = "Likely Not Eligible"


        # -----------------------------
        # RISK SCORE
        # -----------------------------

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


        # -----------------------------
        # KEY FACTORS
        # -----------------------------

        factors = []


        if credit_score >= 700:

            factors.append(
                "Strong credit score"
            )

        elif credit_score < 600:

            factors.append(
                "Low credit score"
            )


        if payment_history == "Good":

            factors.append(
                "Good payment history"
            )

        elif payment_history == "Poor":

            factors.append(
                "Poor payment history"
            )


        if income >= 50000:

            factors.append(
                "Strong monthly income"
            )


        if existing_loan > income * 10:

            factors.append(
                "High existing debt"
            )


        if loan_amount > income * 20:

            factors.append(
                "High requested loan amount"
            )


        if not factors:

            factors.append(
                "Multiple financial factors considered"
            )


        # -----------------------------
        # SAVE PREDICTION
        # -----------------------------

        conn = sqlite3.connect(
            "loanlens.db"
        )

        cursor = conn.cursor()


        cursor.execute("""

            INSERT INTO predictions (

                timestamp,
                age,
                monthly_income,
                credit_score,
                existing_loan,
                loan_amount,
                loan_term,
                dependents,
                employment_status,
                employment_years,
                payment_history,
                eligibility,
                confidence,
                risk_score,
                risk_level

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        """, (

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            int(data["age"]),

            float(data["monthly_income"]),

            int(data["credit_score"]),

            float(data["existing_loan"]),

            float(data["loan_amount"]),

            int(data["loan_term"]),

            int(data["dependents"]),

            data["employment_status"],

            int(data["employment_years"]),

            data["payment_history"],

            eligibility,

            round(confidence, 2),

            risk_score,

            risk_level

        ))


        conn.commit()

        conn.close()


        # -----------------------------
        # RETURN RESULT
        # -----------------------------

        return jsonify({

            "eligibility":
                eligibility,

            "confidence":
                round(confidence, 2),

            "risk_score":
                risk_score,

            "risk_level":
                risk_level,

            "factors":
                factors

        })


    except Exception as e:

        return jsonify({

            "error":
                str(e)

        }), 400


# -----------------------------
# PREDICTION HISTORY
# -----------------------------

@app.route("/history")
def history():

    conn = sqlite3.connect(
        "loanlens.db"
    )

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()


    cursor.execute("""

        SELECT *

        FROM predictions

        ORDER BY id DESC

        LIMIT 20

    """)


    records = cursor.fetchall()

    conn.close()


    return jsonify([

        dict(record)

        for record in records

    ])


# -----------------------------
# RUN APPLICATION
# -----------------------------

if __name__ == "__main__":

    app.run(debug=True)
