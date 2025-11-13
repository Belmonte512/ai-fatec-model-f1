from predict_service import FinishPredictor

model_path = "best_model_RandomForest.joblib"  # ajuste pro seu nome real
predictor = FinishPredictor(model_path)

res = predictor.predict_single(
    grid=3,
    constructorId="6",   # ex: Mercedes
    driverId="1",        # ex: Hamilton
    circuitId="1"        # ex: Bahrain
)

print(res)
