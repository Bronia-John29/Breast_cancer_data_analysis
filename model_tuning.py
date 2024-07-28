# model_tuning.py

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import pandas as pd

# Load the dataset with selected features
df = pd.read_csv('breast_cancer_data_selected.csv')
X = df.drop('target', axis=1)
y = df['target']

# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

# Create the model
mlp = MLPClassifier(max_iter=100)

# Perform Grid Search CV
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, n_jobs=-1, cv=3)
grid_search.fit(X, y)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
