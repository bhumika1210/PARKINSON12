# %% [markdown]
# # Parkinsons Disease
# The objective is to find a model which will be able to predict whether a person is likely to have parkinsons disease or not given their medical records. 
# ___
# 
# ## Model Performamce
# Accuracy, F1-Score and Recall were the metrics used to evaluate the performance of the model
# 
# | Method    |  Accuracy (%)  | F1-Score (%) | Recall (%) |
# |-----------|---------|-----------|---------|
# | **Random Forest**   | **97.44**   | **98.41** | **100.00** |
# | XGBoost | 89.74   | 93.94 | **100.00** |
# | SVM | 89.74   | 93.33 | 90.32 |
# ___
# 
# ### Steps to Solve Problem
# * Import Dataset and Libraries
# * Data Preprocessing
#     * Train / Test Data split
#     * Missing Data Imputation
#     * Outlier Handling
#     * Feature Scaling
#     * Imbalanced Data
# * Model Build
#     * Model Initiation and Fitting
#     * Test predictions
# * Model Perfromance
#     * Recall
#     * Case Prediction
#         
# 

# %% [markdown]
# ### Import Libraries and Dataset

# %%
import pandas as pd # for data manipulation
import numpy as np # for numerical analysis

# For plottling graphs
import seaborn as sns 
import matplotlib.pyplot as plt

# for saving tools
import joblib

# %%
# Setting Plotting Settings

sns.set_style("darkgrid")

# %% [markdown]
# Import Dataset

# %%
parkinsons = pd.read_csv("parkinsons.csv")

# %%
# Checking First 5 rows of data
parkinsons.head()

# %% [markdown]
# Checking basic information about the dataset

# %%
parkinsons.info()

# %%
parkinsons.columns

# %% [markdown]
# ### Data Preprocessing

# %% [markdown]
# #### Data Shuffle and Split
# It is good pratice to split the dataset before preprocessing to avoid data leakage, shuffling the data adds randomness which can boost model performance

# %%
# Shuffling the data
parkinsons = parkinsons.sample(frac=1, random_state=42).copy()

# %%
# Splitting the data into train and test data
from sklearn.model_selection import train_test_split
X = parkinsons.drop(["name", "status"], axis=1) 
y = parkinsons["status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)

# %%
# Setting the train data to variable name "parkinsons" for data preprocessing
parkinsons = X_train.copy()
parkinsons

# %%
y_train

# %%
parkinsons.iloc[0]

# %% [markdown]
# #### Handling of Missing data 
# 

# %%
# Extracting features
features = [feature for feature in parkinsons.columns]

# %% [markdown]
# There is no missing data in this dataset

# %%
# Check total of missing values
parkinsons.isna().sum()

# %% [markdown]
# #### Feature Scaling
# Scaling values to a range of -3 to 3, so as to boost model perfomance

# %%
# import library for scaling
from sklearn.preprocessing import StandardScaler

# %%
# initialize and scale values
scaler = StandardScaler()
scaler.fit(parkinsons[features])
parkinsons[features] = scaler.transform(parkinsons[features])

# %%
joblib.dump(scaler, "tools/scaler_joblib")

# %% [markdown]
# ### Class Imbalance

# %%
#  Ratio of No Parkinson to Parkinson
y_train.value_counts(normalize=True)

# %%
# joining the data together
parkinsons = pd.concat([parkinsons, y_train], axis=1)

# %%
# Balancing the data
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=51)
X = parkinsons.drop("status", axis=1) 
y = parkinsons["status"]
X_train, y_train = smote.fit_resample(X, y)

# %%
#  Ratio of No Diabetes to Diabetes
y_train.value_counts(normalize=True)

# %% [markdown]
# ### Model Building

# %% [markdown]
# #### Preprocessing Test data

# %%
X_test[features] = scaler.transform(X_test[features]) # scaling features

# %%
# checking first 5 rows of data
X_test.head()

# %%
X_test.shape

# %% [markdown]
# #### Random Forest

# %% [markdown]
# Import Model

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
model = RandomForestClassifier(random_state=51, n_jobs=-1)

# %% [markdown]
# Train model and make predictions

# %%
model.fit(X_train, y_train)

# %%
predictions = model.predict(X_test)

# %% [markdown]
# #### RF Performance

# %%
# libraries to check performance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,  recall_score

# %%
print(f"The accuracy is {accuracy_score(y_test, predictions) * 100:.2f} %")
print(f"The f1 score is {f1_score(y_test, predictions) * 100:.2f} %") 
print(f"The recall is {recall_score(y_test, predictions) * 100:.2f} %")

# %% [markdown]
# Confusion matrix

# %%
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cbar=False);
# TN   FP
# FN*   TP - Recall

# %% [markdown]
# XGboost

# %% [markdown]
# Import Model

# %%
from xgboost import XGBClassifier

# %%
xgb = XGBClassifier(random_state=51)

# %% [markdown]
# Train model and make predictions

# %%
xgb.fit(X_train, y_train)

# %%
predictions = xgb.predict(X_test)

# %% [markdown]
# #### XGBoost Performance

# %%
print(f"The accuracy is {accuracy_score(y_test, predictions) * 100:.2f} %")
print(f"The f1 score is {f1_score(y_test, predictions) * 100:.2f} %") 
print(f"The recall is {recall_score(y_test, predictions) * 100:.2f} %")

# %% [markdown]
# Confusion Matrix

# %%
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cbar=False);
# TN   FP
# FN*   TP - Recall

# %% [markdown]
# ### SVM

# %% [markdown]
# Import Model

# %%
from sklearn.svm import SVC

# %%
svm = SVC()

# %% [markdown]
# Train model and make predictions

# %%
svm.fit(X_train, y_train)

# %%
predictions = svm.predict(X_test)

# %% [markdown]
# #### SVM Performance

# %%
print(f"The accuracy is {accuracy_score(y_test, predictions) * 100:.2f} %")
print(f"The f1 score is {f1_score(y_test, predictions) * 100:.2f} %") 
print(f"The recall is {recall_score(y_test, predictions) * 100:.2f} %")

# %% [markdown]
# Confusion Matrix

# %%
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cbar=False);
# TN   FP
# FN*   TP - Recall

# %% [markdown]
# ### Most important features
# Here we check the top 10 most important features that contribute to the prediction of parkinsons
# * spread1
# * MDVP:Fo(Hz)
# * PPE
# * MDVP:Fhi(Hz)
# * spread2
# * MDVP:APQ
# * MDVP:Flo(Hz)
# * Shimmer:APQ5
# * MDVP:PPQ
# * MDVP:Shimmer
# 

# %%
importance_df = pd.DataFrame({
    "Feature" : features,
    "Importance" : model.feature_importances_}).sort_values("Importance", ascending=False)

# %%
plt.figure(figsize=[10,6])
plt.title("Most Important Features")
sns.barplot(data=importance_df.head(10), y="Feature", x="Importance");

# %% [markdown]
# ### Saving The Model

# %% [markdown]
# Random Forest had the best recall and F1 score so that would be our final model

# %%
joblib.dump(model, "tools/model_joblib")


