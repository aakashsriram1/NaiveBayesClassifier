# Naive Bayes Classifier for Basketball Game Outcomes

This project implements a Naive Bayes Classifier in Python from scratch to predict basketball game outcomes (home win/loss) based on historical game statistics.

## Features
- **Custom Implementation**: Mean, variance, and priors computed manually.
- **Logarithmic Probability Calculations**: Prevents underflow during prediction.
- **Simplified Preprocessing**: Includes categorical encoding and feature engineering for better accuracy.

## Project Structure
- `main`: Entry point for the script.
- `preprocess_data`: Cleans and transforms input data.
- `train_naive_bayes`: Calculates class-specific statistics for training.
- `predict_naive_bayes`: Predicts the outcome for test data using Gaussian distributions.

