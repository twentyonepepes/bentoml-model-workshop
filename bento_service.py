import bentoml

model_ref = bentoml.models.get("ops_sentiment:latest")
clf = bentoml.sklearn.load_model(model_ref)

class OpsSentimentService(bentoml.Service):
    @bentoml.api
    def classify(self, text: str) -> dict:
        y = int(clf.predict([text])[0])
        p = float(clf.predict_proba([text])[0][1])
        return {"label": y, "risk_score": p, "text": text}

svc = OpsSentimentService()
