import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data from the CSV file into a pandas DataFrame
file_path = "payment_fraud.csv"  
data = pd.read_csv(file_path)

# Data Cleaning and Preprocessing
# Check for missing values
print("\nMissing values in the DataFrame:")
print(data.isnull().sum())

# Drop rows with missing values
data.dropna(inplace=True)

# Convert 'localTime' to datetime format
data['localTime'] = pd.to_datetime(data['localTime'])

# Extract hour and minute from 'localTime'
data['hour'] = data['localTime'].dt.hour
data['minute'] = data['localTime'].dt.minute

# Drop 'localTime' column
data.drop('localTime', axis=1, inplace=True)

# Encoding categorical variables
data = pd.get_dummies(data, columns=['paymentMethod'])

# EDA
# Pairplot for visualization
sns.pairplot(data, hue='label')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Splitting the data into train and test sets
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building the Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = rf_model.predict(X_train_scaled)
y_pred_test = rf_model.predict(X_test_scaled)

# Model evaluation
print("\nTraining Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))
