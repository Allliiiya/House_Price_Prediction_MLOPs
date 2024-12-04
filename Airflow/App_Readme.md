# House Price Prediction

This project is a web application for predicting house prices using a trained machine learning model. The application is built using Flask for the backend and HTML/CSS for the frontend.

## Prerequisites

- Python 3.x
- `pip` (Python package installer)

## Setup

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Create a virtual environment**:
   ```sh
   python3 -m venv venv
   ```

3. **Activate the virtual environment**:
   - On macOS and Linux:
     ```sh
     source venv/bin/activate
     ```
   - On Windows:
     ```sh
     venv\Scripts\activate
     ```

4. **Install the required packages**:
   ```sh
   pip install -r requirements.txt
   ```

5. **Ensure the model file is in the correct path**:
   - Verify that the model file (model.pkl) is located at the path specified in app.py:
     ```python
     MODEL_PATH = '/Users/skyleraliya/House_Price_Prediction_MLOPs/Airflow/mlruns/3/a2f6a0ae32ba46369731c394a83ac4d0/artifacts/model/model.pkl'
     ```

## Running the Application

1. **Run the Flask app**:
   ```sh

   python app.py

   ```

2. **Access the UI frontend**:
   - Open your web browser and go to `http://127.0.0.1:5000/`.

## Using the Application

1. **Fill in the form**:
   - Enter the required details in the form fields. Each field has a placeholder explaining what data to enter.

2. **Submit the form**:
   - Click the "Predict" button to submit the form.

3. **View the prediction**:
   - The predicted house price will be displayed on the page.

   ## Example Input and Output

   Here is an example of the input data and the predicted house price:

   **Input Data**:
   ```
   Bldg Type: Twnhs
   Overall Qual: 6
   Year Remod/Add: 1971
   Lot Frontage: 21
   Open Porch SF: 0
   Lot Area: 1680
   Full Bath: 1
   Wood Deck SF: 275
   Gr Liv Area: 987
   BsmtFin SF 1: 156
   Total Bsmt SF: 483
   1st Flr SF: 483
   Garage Area: 264
   Year Built: 1971
   Bedroom AbvGr: 2
   Mas Vnr Area: 504
   Fireplaces: 0
   SalePrice: 96000
   ```

   **Predicted Price**: `$208,355.64`


   ## Testing Data

   Testing data can be accessed [here](https://docs.google.com/spreadsheets/d/1HCXG7RtkvwsvU5du14YxQw2HysRhICMNrrR1gENIuPI/edit?usp=sharing).
## Troubleshooting

- If you encounter any issues, check the terminal for error messages and ensure all dependencies are installed correctly.
- Verify that the model file exists at the specified path and is accessible.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
```
