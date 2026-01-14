import bentoml
from fastapi import FastAPI, Query

model_ref = bentoml.models.get("ops_sentiment:latest")
clf = bentoml.sklearn.load_model(model_ref)

app = FastAPI()

@app.get("/aiops/inference")
def inference(data: str = Query(..., min_length=1)):
    y = int(clf.predict([data])[0])
    p = float(clf.predict_proba([data])[0][1])
    return {"label": y, "risk_score": p, "text": data}