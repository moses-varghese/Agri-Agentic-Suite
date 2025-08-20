# from statsmodels.tsa.arima.model import ARIMA
# from shared.core.config import settings
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# import pandas as pd
# import pickle
# import os
# import re
# import requests

# def train_and_save_model():
#     """
#     Downloads historical price data, trains an ARIMA model, and saves it.
#     """
#     model_dir = "app/models"
#     model_path = os.path.join(model_dir, "price_predictor.pkl")
    
#     # Create directory if it doesn't exist
#     os.makedirs(model_dir, exist_ok=True)

#     print("--- Starting Price Prediction Model Training ---")

#     # This is a direct link to a sample CSV of historical Onion prices in the Bangalore market
#     # In a real-world scenario, you would connect to a live API
#     BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
#     params = {
#     "api-key": settings.DatagovAPIKey,
#     "format": "json",
#     "offset": 0,
#     "limit": 1
#     }
#     resp = requests.get(BASE_URL, params=params)
#     # resp.raise_for_status()
#     data = resp.json()
#     total = data["total"]  
#     data_url = f"{BASE_URL}?api-key={API_KEY}&format=csv&offset=0&limit={total}"
#     try:
#         print(f"Downloading data from {data_url}...")
#         df = pd.read_csv(data_url)
#         print("Data downloaded successfully.")

#         # --- Data Preprocessing ---
#         df['date'] = pd.to_datetime(df['Arrival_Date'], dayfirst=True)
#         df.set_index('date', inplace=True)
#         df = df[df['commodity'] == 'Onion'] # Filter for Onion
        
#         # We'll use the 'modal_price' for prediction
#         time_series = df['modal_price'].asfreq('D').fillna(method='ffill')
        
#         if len(time_series) < 30:
#             print("Not enough data to train the model. Exiting.")
#             return

#         print(f"Training model on {len(time_series)} data points...")
        
#         # --- Model Training (ARIMA) ---
#         # These parameters (p,d,q) are standard defaults for many time series.
#         model = ARIMA(time_series, order=(5, 1, 0))
#         model_fit = model.fit()

#         # --- Save the Model ---
#         with open(model_path, 'wb') as pkl_file:
#             pickle.dump(model_fit, pkl_file)
        
#         print(f"✅ Model trained and saved successfully to '{model_path}'")

#     except Exception as e:
#         print(f"❌ ERROR: Model training failed: {e}")


# if __name__ == "__main__":
#     train_and_save_model()

################################################################

# def train_and_save_estimator():
#     """
#     Downloads daily data, cleans it, trains a price estimation model, and saves it.
#     """
#     model_dir = "app/models"
#     model_path = os.path.join(model_dir, "price_estimator.pkl")
#     data_path = os.path.join(model_dir, "cleaned_commodities.json")
    
#     os.makedirs(model_dir, exist_ok=True)

#     print("--- Starting Price Estimation Model Training ---")

#     # This is the direct API link for daily agricultural commodity prices

#     BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
#     params = {
#     "api-key": DATA_GOV_API_Key,
#     "format": "json",
#     "offset": 0,
#     "limit": 1
#     }
#     API_KEY = DATA_GOV_API_Key
#     resp = requests.get(BASE_URL, params=params)
#     # resp.raise_for_status()
#     data = resp.json()
#     total = data["total"]  
#     data_url = f"{BASE_URL}?api-key={API_KEY}&format=csv&offset=0&limit={total}"

#     try:
#         print(f"Downloading data from data.gov.in with url {data_url}...")
#         df = pd.read_csv(data_url)
#         print("Data downloaded successfully.")

#         # --- 1. Data Cleaning & Feature Engineering ---
#         # Rename columns for easier access
#         df.rename(columns={
#             'Modal_x0020_Price': 'price',
#             'Commodity': 'commodity',
#             'State': 'state',
#             'District': 'district',
#             'Arrival_Date': 'date'
#         }, inplace=True)

#         # Select relevant columns and drop rows with missing values
#         df = df[['price', 'commodity', 'state', 'district', 'date']].dropna()
        
#         # Clean the commodity names
#         def clean_commodity(name):
#             name = re.sub(r'\(.*\)', '', name) # Remove anything in parentheses
#             name = name.strip()
#             return name
        
#         df['commodity'] = df['commodity'].apply(clean_commodity)

#         # Engineer the 'month' feature
#         df['date'] = pd.to_datetime(df['date'], dayfirst=True)
#         df['month'] = df['date'].dt.month

#         # Save the unique, cleaned commodity names for later use
#         unique_commodities = df['commodity'].unique().tolist()
#         with open(data_path, 'w') as f:
#             json.dump(unique_commodities, f)
#         print(f"✅ Cleaned commodity list saved to '{data_path}'")

#         # --- 2. Model Training ---
#         features = ['commodity', 'state', 'district', 'month']
#         target = 'price'

#         X = df[features]
#         y = df[target]

#         # Define preprocessing for categorical features
#         categorical_features = ['commodity', 'state', 'district']
#         one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        
#         preprocessor = ColumnTransformer(
#             transformers=[('cat', one_hot_encoder, categorical_features)],
#             remainder='passthrough'
#         )

#         # Create the model pipeline
#         model = Pipeline(steps=[
#             ('preprocessor', preprocessor),
#             ('regressor', GradientBoostingRegressor(n_estimators=50, random_state=42))
#         ])

#         print("Training Gradient Boosting model...")
#         model.fit(X, y)

#         # --- 3. Save the Model ---
#         with open(model_path, 'wb') as pkl_file:
#             pickle.dump(model, pkl_file)
        
#         print(f"✅ Model trained and saved successfully to '{model_path}'")

#     except Exception as e:
#         print(f"❌ ERROR: Model training failed: {e}")


# if __name__ == "__main__":
#     import json # Add json import for the script to run standalone
#     train_and_save_estimator()



import pandas as pd
import pickle
import os
import re
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sqlalchemy import create_engine, text
from shared.core.config import settings
import requests

def train_and_save_estimator():
    """
    Connects to PostgreSQL, syncs the latest daily data, and retrains the model
    on the entire persistent dataset.
    """
    model_dir = "app/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "price_estimator.pkl")
    # data_path = os.path.join(model_dir, "cleaned_commodities.json")
    
    # --- 1. Connect to PostgreSQL ---
    #db_url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    engine = create_engine(settings.DATABASE_URL.replace("asyncpg", "psycopg2"))
    #engine = create_engine(db_url)

    table_name = "commodity_prices"
    
    print("--- Starting Price Estimation Model Training ---")

    try:
        BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        params = {
        "api-key": settings.DATA_GOV_API_KEY,
        "format": "json",
        "offset": 0,
        "limit": 1
        }
        resp = requests.get(BASE_URL, params=params)
        # resp.raise_for_status()
        data = resp.json()
        total = data["total"]
        daily_data_url = f"{BASE_URL}?api-key={settings.DATA_GOV_API_KEY}&format=csv&offset=0&limit={total}"
        # --- 2. Sync Daily Data with Database ---
        # Download the latest daily data
        print(f"Downloading data from data.gov.in...")
        daily_df = pd.read_csv(daily_data_url)
        print("Data downloaded successfully.")

        #Rename columns for easier access
        daily_df.rename(columns={
            'Modal_x0020_Price': 'price',
            'Min_x0020_Price': 'min_price',
            'Max_x0020_Price': 'max_price',
            'Commodity': 'commodity',
            'State': 'state',
            'District': 'district',
            'Arrival_Date': 'date'
        }, inplace=True)

        # Clean and prepare the new data
        daily_df = daily_df[['date', 'state', 'district', 'commodity', 'min_price', 'max_price', 'price']].dropna()
        daily_df['date'] = pd.to_datetime(daily_df['date'], dayfirst=True)
        daily_df['month'] = daily_df['date'].dt.month
        daily_df['year'] = daily_df['date'].dt.year

        def clean_commodity(name):
            name = re.sub(r'\(.*\)', '', name) # Remove anything in parentheses
            name = name.strip()
            return name
        
        daily_df['commodity'] = daily_df['commodity'].apply(clean_commodity)
        
        # Check if table exists, if so, load existing data
        with engine.connect() as connection:
            if pd.io.sql.has_table(table_name, connection):
                print(f"Table '{table_name}' exists. Fetching existing data for deduplication.")
                existing_df = pd.read_sql_table(table_name, connection, index_col='index')
                
                # Find rows in daily_df that are not in existing_df
                merged = daily_df.merge(existing_df, on=['date', 'month', 'year', 'state', 'district', 'commodity', 'min_price', 'max_price'], how='left', indicator=True)
                new_records_df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge', 'price_y'])
                new_records_df.rename(columns={'price_x': 'price'}, inplace=True)
            else:
                print(f"Table '{table_name}' does not exist. Treating all downloaded data as new.")
                new_records_df = daily_df

        # Append only the new records to the database
        if not new_records_df.empty:
            print(f"Appending {len(new_records_df)} new records to the database...")
            new_records_df.to_sql(table_name, engine, if_exists='append')
            print("✅ New data saved to PostgreSQL.")
        else:
            print("✅ No new daily data to add. Database is up to date.")

        # --- 3. Train Model on the ENTIRE Dataset from PostgreSQL ---
        print("Loading full dataset from PostgreSQL for retraining...")
        full_df = pd.read_sql_table(table_name, engine, index_col='index')

        if full_df.empty:
            print("⚠️ No data available in the database to train the model. Skipping training.")
            return
        
        # unique_commodities = sorted(full_df['commodity'].unique().tolist())
        # with open(data_path, 'w') as f: json.dump(unique_commodities, f)

        features = ['commodity', 'state', 'district', 'month', 'year', 'min_price', 'max_price']
        target = 'price'
        X = full_df[features]
        y = full_df[target]

        # Define and train the model pipeline
        # preprocessor = ColumnTransformer(
        #     transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['commodity', 'state', 'district'])],
        #     remainder='passthrough').set_output(transform="pandas")

        # preprocessor = ColumnTransformer(
        # transformers=[
        #     ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
        #     ['commodity', 'state', 'district']),
        #     ('num', 'passthrough', ['month', 'year', 'min_price', 'max_price'])
        # ]).set_output(transform="pandas")


        preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            ['commodity', 'state', 'district']),
            # Use FunctionTransformer(identity) instead of 'passthrough'
            ('num', FunctionTransformer(validate=False), ['month', 'year', 'min_price', 'max_price'])]).set_output(transform="pandas")

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=50, random_state=42))])

        print(f"Retraining model on {len(full_df)} total data points...")
        model.fit(X, y)

        with open(model_path, 'wb') as pkl_file: pickle.dump(model, pkl_file)
        print(f"✅ Model retrained and saved successfully to '{model_path}'")

    except Exception as e:
        print(f"❌ ERROR: Model training failed: {e}")

if __name__ == "__main__":
    train_and_save_estimator()