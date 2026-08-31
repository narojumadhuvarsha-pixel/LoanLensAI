import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
data = pd.read_csv("loan_data.csv")

# Separate features and target
X = data.drop("Eligible", axis=1)
y = data["Eligible"]


# Categorical columns
categorical_features = [
    "EmploymentStatus",
    "PaymentHistory"
]

# Numerical columns
numerical_features = [
    "Age",
    "MonthlyIncome",
    "CreditScore",
    "ExistingLoan",
    "LoanAmount",
    "LoanTerm",
    "Dependents",
    "EmploymentYears"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# Machine Learning Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=10
)


# Complete ML pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train
print("🤖 Training LoanLens AI...")

pipeline.fit(X_train, y_train)


# Evaluate
predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n✅ Model training completed!")
print(f"🎯 Accuracy: {accuracy * 100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y_test, predictions))


# Save model
joblib.dump(pipeline, "loan_model.pkl")

print("\n💾 Model saved as: loan_model.pkl")