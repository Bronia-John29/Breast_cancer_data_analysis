# app.py

import streamlit as st
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# Load the dataset
df = pd.read_csv('breast_cancer_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Feature selection
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Create and train the model
mlp = MLPClassifier(hidden_layer_sizes=(50,100,50), activation='relu', solver='adam', max_iter=100, alpha=0.0001, learning_rate='adaptive')
mlp.fit(X_new, y)

# Streamlit app
st.title('Breast Cancer Prediction')

# User inputs
user_input = []
for feature in selected_features:
    value = st.number_input(f'Input {feature}', value=0.0)
    user_input.append(value)

# Predict
if st.button('Predict'):
    prediction = mlp.predict([user_input])
    st.write(f'Prediction: {"Malignant" if prediction[0] == 1 else "Benign"}')
