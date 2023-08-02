# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
#import tensorflow.keras as keras
from tensorflow.keras.models import load_model


app = FastAPI()

# Cargar el modelo entrenado y el scaler
model = load_model("modeloCnn.keras")
#model = keras.models.load_model("modeloCnn.h5")
#model = joblib.load("cardioVascularCnn.pkl")
scaler = joblib.load("standardScalerCnn.pkl")


# Definir la estructura de entrada para las solicitudes
class HeartData(BaseModel):
    Age: float
    Sex: float
    ChestPainType: float
    RestingBP: float
    Cholesterol: float
    FastingBS: float
    RestingECG: float
    MaxHR: float
    ExerciseAngina: float
    Oldpeak: float
    ST_Slope: float


@app.post("/predict/")
def predict(data: HeartData):
    # Convertir los datos de entrada en un DataFrame
    data_dict = data.dict()
    input_data = pd.DataFrame([data_dict])

    # Escalar los datos de entrada
    input_data_scaled = scaler.transform(input_data)

    # Realizar la predicción con el modelo
    prediction = model.predict(input_data_scaled).round(0)
    #risk_percentage = prediction.item()


    # Convertir el resultado de la predicción en un valor entero (0 o 1)
    result = int(prediction[0, 0])

    return {"prediction": result}

