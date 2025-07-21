import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the dataset
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

print("Dataset shape:", df.shape)
print("Target distribution:")
print(df['DEATH_EVENT'].value_counts())

# Define features (X) and target (y)
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create individual models
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
lr = LogisticRegression(random_state=42, max_iter=1000)
svm = SVC(probability=True, random_state=42)

# Create an ensemble model using VotingClassifier
ensemble_model = VotingClassifier(
    estimators=[('rf', rf), ('lr', lr), ('svm', svm)],
    voting='soft'
)

# Train the ensemble model
ensemble_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = ensemble_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Also train individual models to compare
print("\nIndividual Model Accuracies:")

# Random Forest
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# Logistic Regression
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

# SVM
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"SVM Accuracy: {svm_accuracy:.4f}")

# Choose the best model
best_accuracy = max(accuracy, rf_accuracy, lr_accuracy, svm_accuracy)
if best_accuracy == accuracy:
    best_model = ensemble_model
    model_name = "Ensemble"
elif best_accuracy == rf_accuracy:
    best_model = rf
    model_name = "Random Forest"
elif best_accuracy == lr_accuracy:
    best_model = lr
    model_name = "Logistic Regression"
else:
    best_model = svm
    model_name = "SVM"

print(f"\nBest Model: {model_name} with accuracy: {best_accuracy:.4f}")

# Save the best model and scaler
with open("heart_failure_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("Best model and scaler saved successfully!")

