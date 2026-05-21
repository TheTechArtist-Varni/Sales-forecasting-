

# FLASK DASHBOARD APP


from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "amazon_final_model.pkl"
EDA_PATH = "eda_stats.json"
METRICS_PATH = "model_metrics.json"

model = None
eda_stats = {}
model_metrics = {}





def load_assets():
    global model, eda_stats, model_metrics

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)

    if os.path.exists(EDA_PATH):
        with open(EDA_PATH) as f:
            eda_stats = json.load(f)

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            model_metrics = json.load(f)

load_assets()  # ← add this




@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/overview")
def api_overview():
    return jsonify({
        "stats": {
            "total_products": eda_stats.get("total_products", 0),
            "avg_rating": eda_stats.get("avg_rating", 0),
            "avg_discount": eda_stats.get("avg_discount", 0),
            "avg_discounted_price": eda_stats.get("avg_discounted_price", 0),
            "total_reviews": eda_stats.get("total_reviews", 0),
        },
        "model": {
            "train_r2": model_metrics.get("train_r2", 0),
            "test_r2": model_metrics.get("test_r2", 0),
            "test_rmse": model_metrics.get("test_rmse", 0),
            "train_samples": model_metrics.get("train_samples", 0),
            "test_samples": model_metrics.get("test_samples", 0),
        }
    })


@app.route("/api/eda/rating_distribution")
def api_rating_dist():
    return jsonify(eda_stats.get("rating_distribution", {}))


@app.route("/api/eda/top_categories")
def api_top_categories():
    return jsonify(eda_stats.get("top_categories", {}))


@app.route("/api/eda/discount_vs_rating")
def api_discount_rating():
    return jsonify(eda_stats.get("discount_vs_rating", {}))


@app.route("/api/eda/price_distribution")
def api_price_dist():
    return jsonify(eda_stats.get("price_distribution", {}))


@app.route("/api/eda/top_reviewed")
def api_top_reviewed():
    return jsonify(eda_stats.get("top_reviewed", []))


@app.route("/api/model/actual_vs_predicted")
def api_avp():
    return jsonify(model_metrics.get("actual_vs_predicted", []))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.json
    try:
        sample = pd.DataFrame([{
            "product_name": data.get("product_name", "Unknown Product"),
            "category": data.get("category", "Electronics"),
            "discounted_price": float(data.get("discounted_price", 500)),
            "actual_price": float(data.get("actual_price", 1000)),
            "discount_percentage": float(data.get("discount_percentage", 50)),
            "rating_count": float(data.get("rating_count", 100)),
            "about_product": data.get("about_product", ""),
            "price_drop": float(data.get("actual_price", 1000)) - float(data.get("discounted_price", 500)),
            "price_drop_ratio": (float(data.get("actual_price", 1000)) - float(data.get("discounted_price", 500))) / float(data.get("actual_price", 1000)) if float(data.get("actual_price", 1000)) > 0 else 0
        }])

        prediction = model.predict(sample)[0]
        prediction = float(np.clip(prediction, 1.0, 5.0))

        return jsonify({"predicted_rating": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    load_assets()
    app.run(debug=True, port=5002)
