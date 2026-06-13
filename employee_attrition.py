import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==================================================
# GENERATE DATASET
# ==================================================

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "Age": np.random.randint(18,60,n),
    "Gender": np.random.choice(["Male","Female"],n),
    "Department": np.random.choice(
        ["HR","Sales","IT","Finance","Marketing"],n),
    "JobRole": np.random.choice(
        ["Manager","Executive","Analyst","Developer","Associate"],n),
    "MonthlyIncome": np.random.randint(15000,120000,n),
    "JobSatisfaction": np.random.randint(1,5,n),
    "WorkLifeBalance": np.random.randint(1,5,n),
    "OverTime": np.random.choice(["Yes","No"],n),
    "YearsAtCompany": np.random.randint(0,25,n),
    "YearsSinceLastPromotion": np.random.randint(0,10,n)
})

# Attrition Logic

attrition = []

for _, row in data.iterrows():

    risk = 0

    if row["OverTime"] == "Yes":
        risk += 2

    if row["JobSatisfaction"] <= 2:
        risk += 2

    if row["WorkLifeBalance"] <= 2:
        risk += 2

    if row["YearsSinceLastPromotion"] > 5:
        risk += 1

    if row["MonthlyIncome"] < 30000:
        risk += 1

    attrition.append(
        "Yes" if risk >= 4 else "No"
    )

data["Attrition"] = attrition

print("Dataset Shape:", data.shape)

# ==================================================
# DATA PREVIEW
# ==================================================

print("\nFirst 5 Records")
display(data.head())

# ==================================================
# ATTRITION RATE
# ==================================================

attrition_rate = (
    data["Attrition"].value_counts(normalize=True)*100
)

print("\nAttrition Rate (%)")
print(attrition_rate)

# ==================================================
# VISUALIZATION 1
# ==================================================

plt.figure(figsize=(6,4))
sns.countplot(
    x="Attrition",
    data=data
)
plt.title("Attrition Distribution")
plt.show()

# ==================================================
# VISUALIZATION 2
# ==================================================

plt.figure(figsize=(8,5))

dept_attr = pd.crosstab(
    data["Department"],
    data["Attrition"]
)

dept_attr.plot(kind="bar")

plt.title("Department Wise Attrition")
plt.ylabel("Employees")
plt.show()

# ==================================================
# VISUALIZATION 3
# ==================================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x="Attrition",
    y="MonthlyIncome",
    data=data
)

plt.title("Salary vs Attrition")
plt.show()

# ==================================================
# ENCODING
# ==================================================

encoder = LabelEncoder()

for col in data.columns:

    if data[col].dtype == "object":

        data[col] = encoder.fit_transform(
            data[col]
        )

# ==================================================
# CORRELATION HEATMAP
# ==================================================

plt.figure(figsize=(10,7))

sns.heatmap(
    data.corr(),
    annot=True,
    cmap="Blues"
)

plt.title("Correlation Heatmap")
plt.show()

# ==================================================
# MODEL BUILDING
# ==================================================

X = data.drop(
    "Attrition",
    axis=1
)

y = data["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(
    max_iter=2000
)

model.fit(
    X_train,
    y_train
)

# ==================================================
# EVALUATION
# ==================================================

pred = model.predict(X_test)

print("\nAccuracy:")
print(
    round(
        accuracy_score(
            y_test,
            pred
        )*100,
        2
    ),
    "%"
)

print("\nClassification Report")
print(
    classification_report(
        y_test,
        pred
    )
)

# ==================================================
# CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_test,
    pred
)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix")
plt.show()

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": abs(
        model.coef_[0]
    )
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Attrition Factors")
display(
    importance.head(10)
)

plt.figure(figsize=(10,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title("Feature Importance")
plt.show()

# ==================================================
# PREDICTION MODULE
# ==================================================

print("\n==========================")
print("EMPLOYEE ATTRITION PREDICTOR")
print("==========================")

sample = pd.DataFrame({

    "Age":[30],
    "Gender":[1],
    "Department":[2],
    "JobRole":[1],
    "MonthlyIncome":[25000],
    "JobSatisfaction":[1],
    "WorkLifeBalance":[1],
    "OverTime":[1],
    "YearsAtCompany":[2],
    "YearsSinceLastPromotion":[7]

})

probability = model.predict_proba(
    sample
)[0][1]

risk_score = round(
    probability*100,
    2
)

prediction = model.predict(
    sample
)[0]

print("\nRisk Score:", risk_score,"%")

if prediction == 1:

    print("\nHIGH ATTRITION RISK")

    print("\nRecommendations:")
    print("- Improve Work-Life Balance")
    print("- Review Salary Package")
    print("- Reduce Overtime")
    print("- Offer Career Growth")
    print("- Conduct Employee Engagement Sessions")

else:

    print("\nLOW ATTRITION RISK")

# ==================================================
# TOP RISK EMPLOYEES
# ==================================================

all_probs = model.predict_proba(
    X
)[:,1]

data["RiskScore"] = all_probs*100

top_risk = data.sort_values(
    by="RiskScore",
    ascending=False
)

print("\nTop 10 Employees At Risk")
display(
    top_risk[
        ["Age","Department","RiskScore"]
    ].head(10)
)
