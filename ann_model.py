# ann_model.py

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Load the dataset with selected features
df = pd.read_csv('breast_cancer_data_selected.csv')
X = df.drop('target', axis=1)
y = df['target']

# Create the model with best parameters found from Grid Search
mlp = MLPClassifier(hidden_layer_sizes=(50,100,50), activation='relu', solver='adam', max_iter=100, alpha=0.0001, learning_rate='adaptive')

# Train the model
mlp.fit(X, y)

# Make predictions
y_pred = mlp.predict(X)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))
print("\nClassification Report:")
print(classification_report(y, y_pred))
