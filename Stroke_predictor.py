import pandas as pd
import numpy as np

df = pd.read_csv("Strokes.csv")

# -------------------------------------------
# FIX 1: Use separate LabelEncoders per column
# -------------------------------------------
from sklearn.preprocessing import LabelEncoder

gender_enc = LabelEncoder()
work_enc = LabelEncoder()
res_enc = LabelEncoder()
married_enc = LabelEncoder()
smoke_enc = LabelEncoder()

df["gender"] = gender_enc.fit_transform(df["gender"])
df["work_type"] = work_enc.fit_transform(df["work_type"])
df["Residence_type"] = res_enc.fit_transform(df["Residence_type"])
df["ever_married"] = married_enc.fit_transform(df["ever_married"])
df["smoking_status"] = smoke_enc.fit_transform(df["smoking_status"])

# -------------------------------------------
# Fix 2: Remove id
# -------------------------------------------
X = df.drop(["stroke", "id"], axis=1)
y = df["stroke"]

# -------------------------------------------
# Fix 3: Train-test split
# -------------------------------------------
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------
# Decision Tree
# -------------------------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score  # ← ADD THIS

model_dt = DecisionTreeClassifier()
model_dt.fit(xtrain, ytrain)

ypred_dt = model_dt.predict(xtest)
print("Decision Tree Accuracy:", accuracy_score(ytest, ypred_dt))

# -------------------------------------------
# Cross validation
# -------------------------------------------
from sklearn.model_selection import cross_val_score
scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=10, scoring='accuracy')
print("Decision Tree CV Accuracy:", scores.mean())

# -------------------------------------------
# FIX 4: Random Forest (do NOT overwrite)
# -------------------------------------------
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()
model_rf.fit(xtrain, ytrain)

ypred_rf = model_rf.predict(xtest)
print("Random Forest Accuracy:", accuracy_score(ytest, ypred_rf))
