# Fraud-Transaction-Detection
Build a machine learning model using credit card transaction data to detect potentially fraudulent transactions. 

!pip install imbalanced-learn
from google.colab import files
uploaded = files.upload()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
df = pd.read_csv('creditcard.csv')
print(df.head())
print(df['Class'].value_counts())
scaler = StandardScaler()
df['scaled_Amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_Time'] = scaler.fit_transform(df[['Time']])
df.drop(['Time', 'Amount'], axis=1, inplace=True)
X = df.drop('Class', axis=1)
y = df['Class']
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
# Reduce dataset size for faster training (optional)
X_res = X_res.sample(n=10000, random_state=42)
y_res = y_res.loc[X_res.index]
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
# Initialize and train the RandomForestClassifier model
model = RandomForestClassifier(random_state=42)  
model.fit(X_train, y_train) 

y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
