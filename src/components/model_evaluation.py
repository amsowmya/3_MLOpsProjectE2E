import os
import sys
import pandas as pd 
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from src.utils.utils import load_object

from dataclasses import dataclass
from pathlib import Path

class ModelEvaluation:
    def __init__(self):
        logging.info("Evaluation started") 

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        logging.info("evaluation matrics captured")
        return rmse, mae, r2

    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = (test_array[:, :-1], test_array[:, -1])

            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            model.info("model has registered")

        except Exception as e:
            logging.info()
            raise customexception(e, sys)