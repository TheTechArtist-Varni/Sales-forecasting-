

# AMAZON SALES FORECASTING 


import numpy as np
import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error


def load_and_clean(path="amazon.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df = df.replace('|', np.nan)
    df = df.replace('', np.nan)
    df = df.replace(' ', np.nan)

    for col in ['discounted_price', 'actual_price', 'discount_percentage', 'rating_count', 'rating']:
        df[col] = df[col].astype(str)

    df['discounted_price'] = df['discounted_price'].replace('[₹,]', '', regex=True)
    df['actual_price'] = df['actual_price'].replace('[₹,]', '', regex=True)
    df['discount_percentage'] = df['discount_percentage'].replace('%', '', regex=True)
    df['rating_count'] = df['rating_count'].replace('[,]', '', regex=True)

    for col in ['discounted_price', 'actual_price', 'discount_percentage', 'rating_count', 'rating']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['rating', 'discounted_price', 'actual_price', 'discount_percentage', 'rating_count'])

    # Feature engineering
    df['price_drop'] = df['actual_price'] - df['discounted_price']
    df['price_drop_ratio'] = df['price_drop'] / df['actual_price'].replace(0, np.nan)

    return df


def run_eda(df):
    stats = {}

    stats['total_products'] = int(len(df))
    stats['avg_rating'] = round(float(df['rating'].mean()), 2)
    stats['avg_discount'] = round(float(df['discount_percentage'].mean()), 2)
    stats['avg_discounted_price'] = round(float(df['discounted_price'].mean()), 2)
    stats['avg_actual_price'] = round(float(df['actual_price'].mean()), 2)
    stats['total_reviews'] = int(df['rating_count'].sum())

    # Rating distribution (bucketed)
    bins = [0, 1, 2, 3, 4, 5]
    labels = ['0-1', '1-2', '2-3', '3-4', '4-5']
    df['rating_bucket'] = pd.cut(df['rating'], bins=bins, labels=labels)
    rating_dist = df['rating_bucket'].value_counts().sort_index()
    stats['rating_distribution'] = {str(k): int(v) for k, v in rating_dist.items()}

    # Top 10 categories by count
    if 'category' in df.columns:
        cat = df['category'].astype(str).str.split('|').str[0].str.strip()
        top_cats = cat.value_counts().head(10)
        stats['top_categories'] = {str(k): int(v) for k, v in top_cats.items()}
    else:
        stats['top_categories'] = {}

    # Discount vs rating correlation buckets
    disc_bins = [0, 20, 40, 60, 80, 100]
    disc_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    df['disc_bucket'] = pd.cut(df['discount_percentage'], bins=disc_bins, labels=disc_labels)
    disc_rating = df.groupby('disc_bucket', observed=True)['rating'].mean().round(2)
    stats['discount_vs_rating'] = {str(k): float(v) for k, v in disc_rating.items()}

    # Price range distribution
    price_bins = [0, 500, 1000, 5000, 10000, 999999]
    price_labels = ['<₹500', '₹500-1K', '₹1K-5K', '₹5K-10K', '>₹10K']
    df['price_bucket'] = pd.cut(df['discounted_price'], bins=price_bins, labels=price_labels)
    price_dist = df['price_bucket'].value_counts().sort_index()
    stats['price_distribution'] = {str(k): int(v) for k, v in price_dist.items()}

    # Monthly-style: top 10 most reviewed products
    top_reviewed = df.nlargest(10, 'rating_count')[['product_name', 'rating_count', 'rating']].copy()
    top_reviewed['product_name'] = top_reviewed['product_name'].astype(str).str[:40]
    stats['top_reviewed'] = top_reviewed.to_dict(orient='records')

    return stats


def train(df):
    X = df[[
        'product_name', 'category',
        'discounted_price', 'actual_price',
        'discount_percentage', 'rating_count',
        'about_product', 'price_drop', 'price_drop_ratio'
    ]]
    y = df['rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = ['discounted_price', 'actual_price', 'discount_percentage', 'rating_count', 'price_drop', 'price_drop_ratio']
    categorical_features = ['product_name', 'category', 'about_product']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)

    metrics = {
        'train_r2': round(float(r2_score(y_train, train_pred)), 4),
        'test_r2': round(float(r2_score(y_test, test_pred)), 4),
        'test_rmse': round(float(np.sqrt(mean_squared_error(y_test, test_pred))), 4),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test))
    }

    # Actual vs predicted sample (first 50 test points)
    sample_df = y_test.reset_index(drop=True).head(50).to_frame('actual')
    sample_df['predicted'] = test_pred[:50].round(2)
    metrics['actual_vs_predicted'] = sample_df.to_dict(orient='records')

    return clf, metrics, X_test


if __name__ == "__main__":
    print("Loading & cleaning data...")
    df = load_and_clean()

    print(f"Clean dataset: {len(df)} rows")

    print("Running EDA...")
    eda = run_eda(df)
    with open("eda_stats.json", "w") as f:
        json.dump(eda, f)
    print("EDA saved to eda_stats.json")

    print("Training model...")
    clf, metrics, X_test = train(df)
    print("Metrics:", metrics)

    joblib.dump(clf, "amazon_final_model.pkl")
    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f)

    print("Model saved to amazon_final_model.pkl")
    print("Done.")
