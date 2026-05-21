# Sales-forecasting

# Amazon Sales Intelligence Dashboard

XGBoost rating prediction and Flask dashboard for the Amazon product dataset.


## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put your amazon.csv in this folder

# 3. Train the model (generates .pkl and .json files)
python train_model.py

# 4. Start the dashboard
python app.py
```

Then open: http://localhost:5002

## Dashboard Pages

- **Overview** — KPI cards , rating distribution  and model metrics
- **EDA** — Categories, price distribution, discount vs rating correlation, top reviewed products
- **Model** — Actual vs Predicted chart, R² / RMSE metrics
- **Predict** — Live prediction form using the trained model

## Dataset

Download from Kaggle: https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset
Expected columns: product_name, category, discounted_price, actual_price, discount_percentage, rating, rating_count, about_product, review_title, review_content
