# **Employee Attrition Prediction System**

# Steps:

##**1. Data Loading (5 Marks)**
# Load the chosen dataset into your environment and display the first few rows along with the shape to verify correctness.


import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


# Dataset load
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Display first few rows and shape
print(df.head())
print("Dataset shape:", df.shape)
# print(df.columns)

# list(df.columns)

"""## **2. Data Preprocessing (10 Marks)**
Perform and document at least 5 distinct preprocessing steps (e.g., handling missing values, encoding, scaling, outlier detection, feature engineering).
"""

# 1. Handle Missing Values
df = df.dropna()

# 2. Encode Target Variable
X = df[['Age', 'MonthlyIncome', 'JobRole', 'OverTime']]
y = df['Attrition'].map({'Yes':1, 'No':0}).astype(int)

# # 3. Feature Engineering (Age buckets)
# X['AgeGroup'] = pd.cut(X['Age'], bins=[18,30,40,50,60], labels=['Young','Mid','Senior','Late'])

# Separate categorical and numeric columns
categorical_cols = ['JobRole', 'OverTime']
numeric_cols = ['Age', 'MonthlyIncome']

# 4. Outlier Detection (IQR method)
Q1 = X[numeric_cols].quantile(0.25)
Q3 = X[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
mask = ~((X[numeric_cols] < (Q1 - 1.5 * IQR)) |
         (X[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
X = X[mask]
y = y[mask]

# 3. Preprocessor
categorical_cols = ['JobRole', 'OverTime']
numeric_cols = ['Age', 'MonthlyIncome']

# 5. Preprocessor (scaling + one-hot encoding)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])
# ------
# pipeline.fit(X, y)

"""## **3. Pipeline Creation (10 Marks)**
Construct a standard Machine Learning pipeline that integrates preprocessing and the model
"""

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

"""## **4. Primary Model Selection (5 Marks)**
Choose a suitable algorithm and justify why this specific model was selected for the dataset.    

**Answer:**  Random Forest Classifier chosen because:
*   Handles categorical + numerical features well.
*   Robust to outliers and scaling issues.
*   Provides feature importance for HR insights.

## **5. Model Training (10 Marks)**
Train your selected model using the training portion of your dataset.
"""

# Train the pipeline on the training data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

"""## **6. Cross-Validation (10 Marks)**
Apply Cross-Validation  to assess robustness and report the average score with standard deviation.
"""

# Apply 5-fold cross-validation on the training set

scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print("CV Mean:", scores.mean())
print("CV Std:", scores.std())

"""## **7. Hyperparameter Tuning (10 Marks)**
Optimize your model using search methods displaying both the parameters tested and the best results found.
"""

# 7. Hyperparameter tuning
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)

"""## **8. Best Model Selection (10 Marks)**
Select  the final best-performing model based on the hyperparameter tuning results.
"""

# Select the best model from GridSearchCV
best_model = grid.best_estimator_

print("Final Best Model:", best_model)

"""## **9. Model Performance Evaluation (10 Marks)**
Evaluate the model on the test set and print comprehensive metrics suitable for the problem type.
"""

# Predict on the test set

y_pred = best_model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

"""## **Save & Load Model**"""

# Save the pipeline instead of only model
import pickle
with open("employee_model.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)

"""**See rest of task on app.py file**"""