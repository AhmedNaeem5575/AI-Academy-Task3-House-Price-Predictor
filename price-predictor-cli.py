# CLI App to Predict House Price and Expensive Status

import pickle
import numpy as np

# Load models
with open("linear_model.pkl", "rb") as f:
    w_lin, b_lin = pickle.load(f)

with open("logistic_model.pkl", "rb") as f:
    w_log, b_log = pickle.load(f)

# User input
income = float(input("Enter median income in unit 10 thousand dollars (e.g., 3.5 for 35000): "))

# Normalize input (same method as training)
income_norm = (income - 3.8711616013311794) / 1.8992912493062482  # hardcoded mean & std from training

# Predict house price
price_pred = np.dot(income_norm, w_lin) + b_lin
price_pred = price_pred * 115435.6670985836 + 206864.41315519012  # unnormalize prediction

# Predict expensive or not
z = np.dot(income_norm, w_log) + b_log
prob = 1 / (1 + np.exp(-z))
expensive = int(prob.item() >= 0.5)

print(f"\nPredicted house price: ${price_pred.item():,.2f}")
print("Prediction:", "Expensive" if expensive else "Affordable")
