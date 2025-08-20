# -*- coding: utf-8 -*-
# === Injected config shim for sanitized version ===
import os, os.path as _osp

# DB schema/table indirection (medium risk → parameterized)
SCHEMA_PUBLIC = os.getenv("DB_SCHEMA_PUBLIC", "public")
TABLE_TRANSACTION_ITEMS = os.getenv("TABLE_TRANSACTION_ITEMS", "transaction_items_partitioned")
TABLE_RELATIONSHIPS = os.getenv("TABLE_RELATIONSHIPS", "relationships")
TABLE_THREAD_DEAL = os.getenv("TABLE_THREAD_DEAL", "thread_deal")

# Business group UUID (medium risk → parameterized)
SHOP_GROUP_UUID = os.getenv("SHOP_GROUP_UUID", "{SHOP_GROUP_UUID}")

# Paths (medium risk → parameterized)
DATA_DIR = os.getenv("DATA_DIR", "data")
INTERNAL_SHARE_PATH = os.getenv("INTERNAL_SHARE_PATH", "data/share")
REF_STORE_CSV = os.getenv("REF_STORE_CSV", _osp.join(DATA_DIR, "Ref_StoreNameCode.csv"))
\1

# === Business-Logic Strategy Layer (sanitized) ===
# This repository ships with *placeholder* logic so you can run the pipeline end-to-end
# without revealing proprietary modelling decisions. Override by providing your own
# module on PYTHONPATH named `pricing_strategy.py` that defines the same functions.

try:
    import pricing_strategy as _strategy
except Exception:  # pragma: no cover - fall back to public-safe defaults
    class _strategy:
        @staticmethod
        def select_cannibal_pairs(df_weekly, **kwargs):
            """
            Return pairs of (product_id, competitor_id) that likely interact.
            Public-safe default: returns empty list (no cannibalisation).
            """
            return []

        @staticmethod
        def build_demand_model(train_df, features, target):
            """
            Train a demand model. Public-safe default: returns a constant-elasticity
            pseudo-model with elasticity=-1.0 and intercept auto-fitted from data.
            The returned object must have a .predict(df) method.
            """
            class _CEModel:
                def __init__(self, elasticity=-1.0):
                    self.elasticity = elasticity
                    self.y_mean = float(train_df[target].mean()) if len(train_df) else 0.0
                    self.p_mean = float(train_df['price'].mean()) if 'price' in train_df else 1.0

                def predict(self, df):
                    import numpy as np
                    p = df['price'].values if 'price' in df else self.p_mean
                    # Q = Q0 * (p/p0)^elasticity
                    base = self.y_mean if self.p_mean == 0 else self.y_mean
                    p0 = self.p_mean if self.p_mean else 1.0
                    return base * (p / p0) ** self.elasticity
            return _CEModel()

        @staticmethod
        def feasible_cost(product_row, cost_df):
            """
            Pick a usable cost for the product_row. Public-safe default:
            - use the minimum non-null cost column if present, else 0.0
            """
            import numpy as np
            if cost_df is None or len(cost_df)==0:
                return 0.0
            cols = [c for c in cost_df.columns if 'cost' in c.lower()]
            if not cols:
                return 0.0
            return float(np.nanmin(cost_df[cols].to_numpy()))

        @staticmethod
        def simulate_prices(model, grid, cost):
            """
            Given a model, price grid, and unit cost, return dataframe with
            columns ['price','qty_pred','profit'].
            Public-safe default: single-product CE model + profit=(price-cost)*qty.
            Negative-profit rows are filtered out by caller.
            """
            import pandas as pd
            df = pd.DataFrame({'price': grid})
            q = model.predict(df)
            profit = (df['price'] - cost) * q
            return pd.DataFrame({'price': df['price'], 'qty_pred': q, 'profit': profit})
# === End Strategy Layer ===
import psycopg2
import csv
import pandas as pd
import datetime as dt

########## Block1: Fetching raw data from database ##########

##### Step1: Connect to your your database database
db_params = {
            'host': "your host",
            'database': "your database",
            'user': "your user name",
            'password': "your password"
            }



##### Step2: Pagination parameters

### Define batch size
batch_size = 10000  # You can adjust the batch size as needed

### Define start & end date
date_start_str = '2024-01-01'
date_end_str = '2024-03-31'


### Step3-1: Find out total row counts of the query
try:
    # Establish a connection to the database
    conn = psycopg2.connect(**db_params)

    # Create a cursor object
    cursor = conn.cursor()

    # Execute the total count query
    total_count_query = ff"""
                        SELECT COUNT(*)
                        FROM
                        (
                        SELECT 	
                        		shop_title AS Store
                        		,substring(department_title FROM strpos(department_title, '-') + 2) AS Department
                        		,substring(space_title FROM strpos(space_title, '-') + 2) AS Space
                        		,substring(category_title FROM strpos(category_title, '-') + 2) AS Category
                        		,product_title AS Product
                        		,DATE(transacted_at AT TIME ZONE 'UTC' AT TIME ZONE 'Australia/Melbourne') AS Date
                        		,ROUND(price_sell,2) AS Price
                        		,COALESCE(SUM(quantity), 0) AS QTY
                        FROM {SCHEMA_PUBLIC}.{TABLE_TRANSACTION_ITEMS}
                        WHERE category_title IN	
                        				(
                        				--- Soft Drinks
                                        ,'SOFT DRINKS'
                                        ,'SOFT DRINKS, BOTTLED'
                                        ,'SOFT DRINKS, CANNED'
                                        ,'SOFT DRINKS, MIXERS'
                        				,'SOFT DRINKS & MIXERS'
                                        ,'SOFT DRINKS, SINGLE'
                                        
                        				--- CONFECTIONERY
                        				,'CONFECTIONERY'
                         				--,'BULK CONFECTIONERY' in box
                        
                        				--- Chocolates
                        				,'CHOCOLATES'
                        				,'CHOCOLATES, COOKING' -- Contain some chocolates for cooking
                         				--,'CHOCOLATES, BOXED' 
                        				--,'397 - CHOCOLATES, COOKING' -- Contain some chocolates for cooking
                        				
                        				--- Biscuits
                        				,'BISCUITS'
                        				
                        				--- Wines
                        				,'WINES RED, BOTTLED'
                        					
                        				,'WINES RED - LOCAL'
                        					
                        				,'WINES WHITE - LOCAL'
                        					
                        				,'WINES ROSE - LOCAL'
                        
                        				--- Health food = healthy snacks / Organic = un-processed food
                        				,'HEALTH FOODS'
                        				,'HEALTH FOODS ORGANIC'
                        
                        				--- Beer
                        				,'BEER - FULL STRENGTH'
                        
                                    	)
                        		AND
                        		transacted_at BETWEEN (('{date_start_str} 00:00:00+11:00'::timestamptz) AT TIME ZONE 'UTC')::timestamp
                        						  AND (('{date_end_str} 23:59:59+11:00'::timestamptz) AT TIME ZONE 'UTC')::timestamp
 
                        GROUP BY 1,2,3,4,5,6,7
                        ORDER BY 1,2,3,4,5,6
                        ) SUB
                        ;
    """
    cursor.execute(total_count_query)

    # Fetch the total count
    total_count = cursor.fetchone()[0]
    print(f"Total count: {total_count}")

    # Calculate the number of batches
    num_batches = (total_count + batch_size - 1) // batch_size
    print(f"Number of batches: {num_batches}")

### End of Step3-1: Find out total row counts of the query



### Step4-1: Fetch data by batch

    df_list = []
    for batch_num in range(num_batches):
        # Calculate offset and limit for the current batch
        #offset: Represents the starting index of the current batch
        offset = batch_num * batch_size
        # limit: Represents the number of items to retrieve for the current batch
        limit = batch_size
        
        # Execute the main query with pagination
        main_query = ff"""
                    SELECT 	
                    		shop_title AS Store
                    		,substring(department_title FROM strpos(department_title, '-') + 2) AS Department
                    		,substring(space_title FROM strpos(space_title, '-') + 2) AS Space
                    		,substring(category_title FROM strpos(category_title, '-') + 2) AS Category
                    		,product_title AS Product
                    		--- Move the first day of the week to Wednesday
                    		,DATE(transacted_at AT TIME ZONE 'UTC' AT TIME ZONE 'Australia/Melbourne') AS Date
                    		,ROUND(price_sell,2) AS Price
                    		,COALESCE(SUM(quantity), 0) AS QTY
                    FROM {SCHEMA_PUBLIC}.{TABLE_TRANSACTION_ITEMS}
                    WHERE category_title IN	
                    				(
                    				--- Soft Drinks
                                    ,'SOFT DRINKS'
                                    ,'SOFT DRINKS, BOTTLED'
                                    ,'SOFT DRINKS, CANNED'
                                    ,'SOFT DRINKS, MIXERS'
                    				,'SOFT DRINKS & MIXERS'
                                    ,'SOFT DRINKS, SINGLE'
                                    
                    				--- CONFECTIONERY
                    				,'CONFECTIONERY'
                     				--,'BULK CONFECTIONERY' in box
                    
                    				--- Chocolates
                    				,'CHOCOLATES'
                    				,'CHOCOLATES, COOKING' -- Contain some chocolates for cooking
                     				--,'CHOCOLATES, BOXED'
                    				
                    				--- Biscuits
                    				,'BISCUITS'
                    				
                    				--- Wines
                    				,'WINES RED, BOTTLED'
                    					
                    				,'WINES RED - LOCAL'
                    					
                    				,'WINES WHITE - LOCAL'
                    					
                    				,'WINES ROSE - LOCAL'
                    
                    				--- Health food = healthy snacks / Organic = un-processed food
                    				,'HEALTH FOODS'
                    				,'HEALTH FOODS ORGANIC'
                    
                    				--- Beer
                    				,'BEER - FULL STRENGTH'
                    
                                	)
                    		AND
                    		transacted_at BETWEEN (('{date_start_str} 00:00:00+11:00'::timestamptz) AT TIME ZONE 'UTC')::timestamp
                    						  AND (('{date_end_str} 23:59:59+11:00'::timestamptz) AT TIME ZONE 'UTC')::timestamp

                    GROUP BY 1,2,3,4,5,6,7
                    ORDER BY 1,2,3,4,5,6
                    LIMIT {limit}
                    OFFSET {offset}
                ;
        """
        cursor.execute(main_query)
        
        rows = cursor.fetchall()
        
        # Create a DataFrame from the fetched rows
        df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
        df_list.append(df)
        
        print(f"Batch {batch_num + 1} appended to df_list")
    
    # Concatenate all DataFrames into a single DataFrame
    result_df = pd.concat(df_list, ignore_index=True)

### End of Step4-1: Fetch data by batch



### Step5: Error handeling 

except psycopg2.Error as e:
    print(f"Error: {e}")



### Step6: Close the cursor and connection

finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()

########## End of Block1: Fetching raw data from database ##########





########## Block2: Rename columns and save out ##########

##### Step1: Change column name
# Create new column names
col_new =   [ 'Store'
            , 'Department'
            , 'Space'
            , 'Category'
            , 'Product'
            , 'Date'
            , 'Price'
            , 'QTY']

# Rename columns
dict_col = {old_col: new_col for old_col, new_col in zip(result_df.columns, col_new)}
result_df = result_df.rename(columns=dict_col)


##### Step2: Save out raw data

# Save out result
fld_cnblz_dataset = r"os.path.join(DATA_DIR, '')"
file_path = fld_cnblz_dataset + "DailyQTY_SelectedCategories_24Q1_FillNA.csv"
result_df.to_csv(file_path, index=False, header=True)

########## End of Block2: Rename columns and save out ##########
