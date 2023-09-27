from cog import BasePredictor, Input
from typing import List
from cefr_predictor.inference import Model


class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = Model("cefr_predictor/models/xgboost.joblib")

    def predict(self,
                texts: List[str] = Input(description="List of texts to evaluate")
                ) -> List:
        """Run a single prediction on the model"""
        # logging.info(f"Processing texts - {texts}")
        preds, probas = self.model.predict_decode(texts)

        results = []
        for text, pred, proba in zip(texts, preds, probas):
            row = {"text": text, "level": pred, "scores": proba}
            results.append(row)
        # logging.info(f"Request processed. Got results - {results}")
        return results
