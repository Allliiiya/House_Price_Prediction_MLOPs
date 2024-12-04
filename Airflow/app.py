from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model from pickle file
MODEL_PATH = '/Users/skyleraliya/House_Price_Prediction_MLOPs/Airflow/mlruns/3/a2f6a0ae32ba46369731c394a83ac4d0/artifacts/model/model.pkl'
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    
    if request.method == "POST":
        try:
            # Collect form data in the correct order
            input_data = {
                "Year Remod/Add": [float(request.form.get("Year_Remod_Add"))],
                "Bldg Type_2fmCon": [float(request.form.get("Bldg_Type_2fmCon"))],
                "Bldg Type_TwnhsE": [float(request.form.get("Bldg_Type_TwnhsE"))],
                "Lot Frontage": [float(request.form.get("Lot_Frontage"))],
                "Open Porch SF": [float(request.form.get("Open_Porch_SF"))],
                "Lot Area": [float(request.form.get("Lot_Area"))],
                "Full Bath": [float(request.form.get("Full_Bath"))],
                "Bldg Type_1Fam": [float(request.form.get("Bldg_Type_1Fam"))],
                "Bldg Type_Twnhs": [float(request.form.get("Bldg_Type_Twnhs"))],
                "Wood Deck SF": [float(request.form.get("Wood_Deck_SF"))],
                "Bldg Type_Duplex": [float(request.form.get("Bldg_Type_Duplex"))],
                "Overall Qual": [float(request.form.get("Overall_Qual"))],
                "Gr Liv Area": [float(request.form.get("Gr_Liv_Area"))],
                "BsmtFin SF 1": [float(request.form.get("BsmtFin_SF_1"))],
                "Total Bsmt SF": [float(request.form.get("Total_Bsmt_SF"))],
                "1st Flr SF": [float(request.form.get("1st_Flr_SF"))],
                "Fireplaces": [float(request.form.get("Fireplaces"))],
                "Garage Area": [float(request.form.get("Garage_Area"))],
                "Year Built": [float(request.form.get("Year_Built"))],
                "Mas Vnr Area": [float(request.form.get("Mas_Vnr_Area"))],
                "TotRms AbvGrd": [float(request.form.get("TotRms_AbvGrd"))]
            }
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(input_data)
            
            # Make prediction and format to 2 decimal places with comma separator
            raw_prediction = model.predict(df)[0]
            prediction = "{:,.2f}".format(raw_prediction)
            
        except Exception as e:
            error = f"Error occurred: {str(e)}"
    
    return render_template("index.html", prediction=prediction, error=error)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse incoming JSON data
        data = request.get_json()
        input_data = pd.DataFrame(data)
        
        # Make predictions and format them
        raw_predictions = model.predict(input_data)
        formatted_predictions = ["{:,.2f}".format(pred) for pred in raw_predictions]
        
        return jsonify(formatted_predictions)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
