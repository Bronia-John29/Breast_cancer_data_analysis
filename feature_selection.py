# feature_selection.py

from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

# Load the dataset
df = pd.read_csv('breast_cancer_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Select K best features
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print("Selected Features:", selected_features)

# Save the selected features to a new CSV file
df_selected = pd.DataFrame(X_new, columns=selected_features)
df_selected['target'] = y
df_selected.to_csv('breast_cancer_data_selected.csv', index=False)
