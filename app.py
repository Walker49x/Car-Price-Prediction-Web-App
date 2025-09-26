from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load cleaned car dataset for dropdowns
car = pd.read_csv("Cleaned_Car1.csv")   # use the same name you saved during training
print(car.head())

# Load the trained pipeline (contains column transformer + model)
with open("CAR_MODEL2.pkl", "rb") as f:
    model = pickle.load(f)
print("Loaded model type:", type(model))

@app.route("/", methods=["GET", "POST"])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template(
        "index.html",
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types
    )

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
  try:
    # Get the input data
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kilo_driven = int(request.form.get('kilo_driven'))

    # Build input DataFrame (order of columns must match training)
    input_data = pd.DataFrame([[car_model, company, year, kilo_driven, fuel_type]],
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    # Load model (pipeline)
    with open("CAR_MODEL20.pkl", "rb") as f:
        model = pickle.load(f)

    # Directly predict (pipeline handles preprocessing)
    y_pred = model.predict(input_data)

    return str(np.round(y_pred[0], 2))

  except Exception as e:
    return "Error occurred: " + str(e), 500


if __name__ == "__main__":
    app.run(debug=True)
