"""
Module: app.py
Description: 
    This module contains the main application logic for the FastAPI web application. 
    It sets up the API endpoints, handles requests, and integrates with the machine learning model 
    for predictions.
"""

import os

from fastapi import FastAPI
from loguru import logger
import pandas as pd
from pydantic import BaseModel

from src.inference import load_model, predict


FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASSES = ["setosa", "versicolor", "virginica"]

class IrisData(BaseModel):
    """
    Pydantic model for validating input data for the prediction endpoint.
    It defines the expected structure of the input data, which includes
    the features required for making predictions with the machine learning model.
    """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


app = FastAPI()

logger.info("Loading model...")
MODEL_PATH = os.path.join("models", "model.joblib")
MODEL = load_model(MODEL_PATH)
logger.info("Model loaded successfully.")

@app.get("/")
def healthcheck():
    """
    Health check endpoint to verify that the application is running.
    Returns a simple message indicating the status of the application.
    """
    return {"status": "ok"}


@app.post("/predict")
def prediction(features: IrisData):
    """
    Prediction endpoint to generate predictions using the machine learning model.
    Accepts input data in JSON format and returns the predicted class.

    Parameters
    ----------
    features : IrisData
        An instance of the IrisData Pydantic model containing the features for prediction.

    Returns
    -------
    dict
        A dictionary containing the predicted class based on the input features.
    """

    logger.info("Start prediction...")
    
    logger.debug(f"Received features: {features}")
    data = pd.DataFrame([features.model_dump()])
    logger.debug(f"Data for prediction: {data}")

    logger.info("Making prediction...")
    prediction = predict(MODEL, data[FEATURES])
    predicted_class = CLASSES[prediction[0]]

    logger.debug(f"Predicted class index: {prediction[0]}, Predicted class: {predicted_class}")
    logger.info("Prediction completed.")
    return {"prediction": predicted_class}
