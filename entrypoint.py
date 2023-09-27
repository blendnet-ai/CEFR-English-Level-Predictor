from cefr_predictor.inference import Model
import argparse

def parse_text_files():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text_files", nargs="+", default=[])
    args = parser.parse_args()

    texts = []
    for path in args.text_files:
        with open(path, "r") as f:
            texts.append(f.read())
    return texts

if __name__ == "__main__":
    texts = parse_text_files()
    if len(texts) == 0:
        raise Exception("Specify one or more documents to evaluate.")

    model = Model("cefr_predictor/models/xgboost.joblib")
    preds, probas = model.predict_decode(texts)

    results = []
    for text, pred, proba in zip(texts, preds, probas):
        row = {"text": text, "level": pred, "scores": proba}
        results.append(row)

    print(results)