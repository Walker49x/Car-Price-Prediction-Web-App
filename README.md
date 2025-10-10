Car Price Prediction Web App

End-to-end machine learning pipeline and Flask web app to predict used car prices.
Predicts price given name, company, year, kms_driven, and fuel_type. Built with Python, scikit-learn, and Flask, with a carousel-style UI for interactive predictions.

Features

Data cleaning & feature engineering for mixed categorical/numeric inputs.

Scikit-learn Pipeline (ColumnTransformer + LinearRegression) for robust preprocessing and inference.

Achieved R² ≈ 0.85 after random-state search and evaluation.

Flask web interface for real-time predictions with dropdowns and inputs.

Safe handling of unseen categorical values (OneHotEncoder(handle_unknown='ignore')).

Tech Stack

Python 3.10+

Flask

pandas, numpy

scikit-learn

HTML/CSS/JS for frontend (carousel UI)


