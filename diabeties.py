import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
#Taking diabaties data sets into csv file
df=pd.read_csv("/content/diabetes.csv")
print(df)
# Basic exploration
print("Data Head:")
print(data.head())
print("\nData Info:")
print(data.info())
print("\nData Description:")
print(data.describe())

# Visualize the data
print("\nData Pairplot:")
sns.pairplot(data, hue='Outcome')
plt.show()

# Check for missing values
print("Missing Values:")
print(data.isnull().sum())

# Split the data into features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Feature importance
print("\nFeature Importances:")
feature_importances = pd.Series(model.feature_importances_, index=column_names[:-1])
print(feature_importances.sort_values(ascending=False))
feature_importances.nlargest(10).plot(kind='barh')
plt.show()

# Model Summary
print("\nModel Summary:")
print("Number of trees in the forest:", model.n_estimators)
print("Number of features:", X_train.shape[1])
print("Classes:", model.classes_)
