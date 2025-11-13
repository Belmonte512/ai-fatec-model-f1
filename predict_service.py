# predict_service.py

import joblib
import pandas as pd

class FinishPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict_single(self, grid: int, constructorId: str, driverId: str, circuitId: str):
        # monta DataFrame com o mesmo formato que o treino
        data = pd.DataFrame([{
            'grid': grid,
            'constructorId': str(constructorId),
            'driverId': str(driverId),
            'circuitId': str(circuitId)
        }])

        proba = self.model.predict_proba(data)[0, 1]   # prob. de finished = 1
        pred = self.model.predict(data)[0]             # 0 ou 1

        return {
            'predicted_class': int(pred),
            'prob_finished': float(proba),
            'prob_not_finished': float(1 - proba)
        }
