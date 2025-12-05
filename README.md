🚗 Car Price Prediction Web App


A Machine-Learning powered Flask web application that predicts the resale price of cars based on historical market data.



📌 Overview


This project integrates an end-to-end ML regression pipeline with a fully responsive web interface.
Users can select car details like model, company, year, kilometers driven, and fuel type — and instantly get the approximate market price.

The model is trained on cleaned automotive datasets and achieves strong generalization performance through systematic evaluation and hyperparameter tuning.



✨ Key Features


🔧 Automated ML Pipeline
Preprocessing + encoding + regression inside a single Scikit-Learn Pipeline

🎯 High-accuracy Price Prediction
Achieved R² ≈ 0.85, tested across 1000+ random states

🎨 Interactive Web UI
Built using HTML, CSS & JavaScript, featuring a carousel-style form layout

🌐 Real-time Inference
Flask backend returns predictions instantly via HTTP POST

💪 Robust Categorical Handling
OneHotEncoder(handle_unknown='ignore') ensures stability with unseen values

🛠 Tech Stack
Layer	Technologies
Frontend	HTML, CSS, Vanilla JavaScript
Backend API	Flask
ML Tools	Scikit-Learn, Pandas, NumPy
Model Type	Regression (Linear Regression inside Pipeline)
Deployment	Local Flask Server (future: cloud deploy)




📊 Model Workflow

1️⃣ Data cleaning & missing-value handling

2️⃣ Feature engineering (encoding + scaling where needed)

3️⃣ Train/test split + hyperparameter evaluation

4️⃣ Export trained pipeline using pickle

5️⃣ Web app loads the model → predicts from user input

🚀 How to Run Locally

🔹 1️⃣ Install Dependencies
pip install -r requirements.txt

🔹 2️⃣ Start Flask Server
python app.py

🔹 3️⃣ Visit the App in Browser
http://127.0.0.1:5000/




