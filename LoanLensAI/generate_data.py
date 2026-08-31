import pandas as pd
import numpy as np

np.random.seed(42)

N = 3000

age = np.random.randint(21, 61, N)
monthly_income = np.random.randint(15000, 150001, N)
credit_score = np.random.randint(300, 851, N)
existing_loan = np.random.randint(0, 500001, N)
loan_amount = np.random.randint(50000, 1000001, N)
loan_term = np.random.choice([12, 24, 36, 48, 60], N)
dependents = np.random.randint(0, 5, N)

employment_status = np.random.choice(
    ["Salaried", "Self-Employed", "Business", "Unemployed"],
    N,
    p=[0.50, 0.20, 0.20, 0.10]
)

employment_years = np.random.randint(0, 21, N)

payment_history = np.random.choice(
    ["Good", "Average", "Poor"],
    N,
    p=[0.60, 0.25, 0.15]
)

data = pd.DataFrame({
    "Age": age,
    "MonthlyIncome": monthly_income,
    "CreditScore": credit_score,
    "ExistingLoan": existing_loan,
    "LoanAmount": loan_amount,
    "LoanTerm": loan_term,
    "Dependents": dependents,
    "EmploymentStatus": employment_status,
    "EmploymentYears": employment_years,
    "PaymentHistory": payment_history
})


# Create a simple synthetic eligibility rule
score = (
    (data["CreditScore"] >= 650).astype(int) * 3
    + (data["MonthlyIncome"] >= 30000).astype(int) * 2
    + (data["ExistingLoan"] < 250000).astype(int) * 1
    + (data["LoanAmount"] <= data["MonthlyIncome"] * 20).astype(int) * 2
    + (data["EmploymentYears"] >= 2).astype(int) * 1
    + (data["PaymentHistory"] == "Good").astype(int) * 2
    + (data["EmploymentStatus"] != "Unemployed").astype(int) * 1
)

# Add a little randomness so the model isn't perfectly predictable
random_noise = np.random.randint(-2, 3, N)

data["Eligible"] = ((score + random_noise) >= 7).astype(int)

data.to_csv("loan_data.csv", index=False)

print("✅ Dataset created successfully!")
print(f"📊 Total applications: {len(data)}")
print("📁 Saved as: loan_data.csv")
print("\nFirst 5 records:")
print(data.head())