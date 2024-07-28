from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the dataset
data = load_breast_cancer()

# Create a DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Save the DataFrame to a CSV file
df.to_csv('breast_cancer_data.csv', index=False)

print("Data preparation complete. Dataset saved as 'breast_cancer_data.csv'.")
