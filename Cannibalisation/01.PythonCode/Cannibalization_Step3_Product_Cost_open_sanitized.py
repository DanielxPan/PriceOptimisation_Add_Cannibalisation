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
import numpy as np



#################### Block 1: Database connection setting #####################

### Step1: Connect to your your database
db_params = {
            'host': "your host",
            'database': "your database",
            'user': "your user name",
            'password': "your password"
            }

#################### End of Block 1: Database connection setting #####################





######################## Block 2: Get product cost from transactions ###########################

##### Step1: Set up parameters for embedded SQL

### Set batch size
batch_size = 10000

### Set date range variable
date_start_str = '2024-01-01'
date_end_str = '2024-03-31'


##### Step2: Try block to fetch data by batch

### Step2-1: Find out total row counts of the query
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
                        ----- 41,546 ROWS 20 SECS
                        SELECT
                        shop_title
                        ,product_id
                        ,product_title
                        ,category_title
                        ,MAX(transacted_at) AS Last_Txn_Date
                        FROM {SCHEMA_PUBLIC}.{TABLE_TRANSACTION_ITEMS}
                        WHERE category_title IN (
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
                        
                        						--- Chips Added on 2nd July
                        						,'SNACKS AND SAVOURIES'
                        
                        						--- Wines
                        						,'WINES RED, BOTTLED'
                        
                        						,'WINES RED - LOCAL'
                        
                        						,'WINES WHITE - LOCAL'
                        
                        						,'WINES ROSE - LOCAL'	
                        
                        						--- Beer
                        						,'BEER - FULL STRENGTH'
                        						)
                        	AND transacted_at BETWEEN (('{date_start_str} 00:00:00+11:00'::timestamptz) AT TIME ZONE 'UTC')::timestamp
                                            	  AND (('{date_end_str} 23:59:59+11:00'::timestamptz) AT TIME ZONE 'UTC')::timestamp
                        GROUP BY 1,2,3,4
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

### End of Step2-1: Find out total row counts of the query



### Step2-2: Fetch dataset by batch

    df_list = []
    for batch_num in range(num_batches):
        # Calculate offset and limit for the current batch
        #offset: Represents the starting index of the current batch
        offset = batch_num * batch_size
        # limit: Represents the number of items to retrieve for the current batch
        limit = batch_size
        
        # Execute the main query with pagination
        main_query = ff"""
                    WITH Prod_Max_Date AS
                    (
                    ----- 41,546 ROWS 20 SECS
                    SELECT
                    shop_title
                    ,product_id
                    ,product_title
                    ,substring(category_title FROM strpos(category_title, '-') + 2) AS Category
                    ,MAX(transacted_at) AS Last_Txn_Date
                    FROM {SCHEMA_PUBLIC}.{TABLE_TRANSACTION_ITEMS}
                    WHERE category_title IN (
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
                    
                    						--- Chips Added on 2nd July
                    						,'SNACKS AND SAVOURIES'
                    
                    						--- Wines
                    						,'WINES RED, BOTTLED'
                    
                    						,'WINES RED - LOCAL'
                    
                    						,'WINES WHITE - LOCAL'
                    
                    						,'WINES ROSE - LOCAL'
                    
                    						--- Beer
                    						,'BEER - FULL STRENGTH'
                    						)
                    	AND transacted_at BETWEEN (('{date_start_str} 00:00:00+11:00'::timestamptz) AT TIME ZONE 'UTC')::timestamp
                                        	  AND (('{date_end_str} 23:59:59+11:00'::timestamptz) AT TIME ZONE 'UTC')::timestamp
                    GROUP BY 1,2,3,4
                    LIMIT {limit}
                    OFFSET {offset}
                    )
                    SELECT 	pmd.*
                    		,ti.price_sell - ti.price_margin AS Cost_Txn
                    FROM Prod_Max_Date pmd
                    LEFT JOIN {SCHEMA_PUBLIC}.{TABLE_TRANSACTION_ITEMS} ti
                    		  ON pmd.shop_title = ti.shop_title
                    		     AND pmd.product_id = ti.product_id
                    			 AND pmd.Last_Txn_Date = ti.transacted_at   
                ;
        """
        cursor.execute(main_query)
        
        rows = cursor.fetchall()
        
        # Create a DataFrame from the fetched rows
        df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
        df_list.append(df)
        
        print(f"Batch {batch_num + 1} appended to df_list")
    
    ### End of Step2-2: Fetch dataset by batch
    
    # Concatenate all DataFrames into a single DataFrame
    df_cost_txn = pd.concat(df_list, ignore_index=True)



##### Step3: except block for error handeling
except psycopg2.Error as e:
    print(f"Error: {e}")



##### Step4: Close the cursor and connection
finally:
    ### Close cursor
    if cursor:
        cursor.close()
    
    ### Close connection
    if conn:
        conn.close()

######################## End of Block 2: Get product cost from transactions ###########################





######################## Block 3: Process cost from transaction dfs ###########################

##### Step1: Check dupliacte
### The voided txn will create 2 rows to offset
duplicates = df_cost_txn[df_cost_txn.duplicated(keep=False)]



##### Step2: Drop dupliacte
df_cost_txn = df_cost_txn.drop_duplicates()



##### Step3: Rename columns
df_cost_txn = df_cost_txn.rename(columns={ 'shop_title': 'Store'
                                          ,'product_title':'Product'
                                          ,'category': 'Category'
                                          , 'cost_txn': 'Cost_Txn'})



##### Step4: Filter columns
df_cost_txn = df_cost_txn[['Store'
                           ,'Product'
                           ,'Category'
                           ,'Cost_Txn']]



##### Step5: Convert 'Cost_Txn' to numeric, forcing errors to NaN
df_cost_txn['Cost_Txn'] = pd.to_numeric(df_cost_txn['Cost_Txn'], errors='coerce')



##### Step6: Select the lowest cost deal Added on 27Nov
df_cost_txn = df_cost_txn.loc[df_cost_txn.groupby(['Store'
                                                   ,'Category'
                                                   ,'Product'])['Cost_Txn'].idxmin()]



##### Step7: Drop dupliacte by columns
df_cost_txn = df_cost_txn[['Store', 'Product','Cost_Txn']].drop_duplicates()



##### Step8: Save out product cost from transaction
fld_raw_data = r"os.path.join(DATA_DIR, '')"
file_cost_txn = fld_raw_data + "Product_Cost_24Q1_TXN.csv"

df_cost_txn.to_csv(file_cost_txn
                   ,index=False
                   ,header=True)

######################## End of Block 3: Process cost from transaction dfs #########################





######################## Block 4: Get product cost from supplier deal ###########################

##### Step1: Set up parameters for embedded SQL

### Set batch size
batch_size = 10000


### Set date range variable
date_start_str = '2024-01-01'
date_end_str = '2024-03-31'


# Database connection parameters
db_params = {
    'host': "your host",
    'database': "your database",
    'user': "your user name",
    'password': "your password"
}



##### Step2: Try block to fetch data by batch

### Step2-1: Find out total row counts of the query
try:
    # Establish a connection to the database
    # ** means passing a dictionary as keyword arguments to a function
    conn = psycopg2.connect(**db_params)
    
    # Create a cursor object
    # Cursors in database operations are used to execute SQL queries
    cursor = conn.cursor()
    
    # Execute the total count query
    total_count_query = f"""
                        ------------------------- 20 MINS 354,646 rows --------------------------------------
                        ----- Start from deal 491,495 rows
                        SELECT COUNT(*)
                        FROM
                        (
                        SELECT
                        		td.id,
                        		td.price,
                        		td.title,
                        		td.minimum_qty,
                        		td.event_start,
                        		td.event_end
                        FROM thread_deal td
                        WHERE td.event_start >= (('{date_start_str} 00:00:00+11:00'::timestamptz) AT TIME ZONE 'UTC')::timestamp
                        	  AND -----
                        	  (('{date_end_str} 23:59:59+11:00'::timestamptz) AT TIME ZONE 'UTC')::timestamp <= td.event_end
                        ) AS subquery
                        ;
                        """
                        
    ### Execute the query                    
    cursor.execute(total_count_query)
    
    # Fetch the total count
    # fetchone() retrieves rows one by one
    total_count = cursor.fetchone()[0]
    print(f"Total row count: {total_count}")
    
    # Calculate the number of batches
    num_batches = (total_count + batch_size - 1) // batch_size
    print(f"Number of batches: {num_batches}")
    
    df_list = []
    for batch_num in range(num_batches):
        # Calculate offset and limit for the current batch
        
        #offset: Represents the starting index of the current batch / determines the starting point for the current batch
        offset = batch_num * batch_size
        # limit: Represents the number of items to retrieve for the current batch
        limit = batch_size
        
        # Execute the main query with pagination
        main_query = ff"""----------- Main query
                        WITH deal AS
                        (
                        ----- For row count 2,405,020(AND)
                        SELECT
                        		td.id,
                        		td.price,
                        		td.title,
                        		td.minimum_qty,
                        		td.event_start,
                        		td.event_end
                        FROM thread_deal td
                        WHERE td.event_start >= (('{date_start_str} 00:00:00+11:00'::timestamptz) AT TIME ZONE 'UTC')::timestamp
                        	  AND -----
                        	  (('{date_end_str} 23:59:59+11:00'::timestamptz) AT TIME ZONE 'UTC')::timestamp <= td.event_end
                        LIMIT {limit}
                        OFFSET {offset}
                        )
                        
                        , rdbp AS --- Get relationship from deal to baseproduct
                        (
                        
                        SELECT *
                        FROM relationships
                        WHERE 	parent_type = 'deal'
                        		AND
                        		child_type = 'baseproduct'
                        		AND
                        		parent_id IN
                        					(
                        					SELECT id
                        					FROM deal
                        					)
                        
                        )
                        
                        , rbpp AS --- Get relationship from baseproduct to product
                        (
                        
                        SELECT *
                        FROM relationships
                        WHERE 	parent_type = 'baseproduct'
                        		AND
                        		child_type = 'product'
                        		AND
                        		parent_id IN
                        					(
                        					SELECT child_id
                        					FROM rdbp
                        					)
                        
                        )
                        
                        ,rshpp AS
                        
                        (
                        SELECT *
                        FROM relationships
                        WHERE 	parent_type = 'shop'
                        		AND
                        		child_type = 'product'
                        		AND
                        		child_id IN
                        					(
                        					SELECT child_id
                        					FROM rbpp
                        					)
                        		AND
                        		parent_id IN  (
                        					  -- Group Filter
                        					  SELECT *
                        					  FROM get_shop_ids_in_group(('{SHOP_GROUP_UUID}')::UUID)
                        					  )
                        )
                        
                        --- Get relationship from product 585,828 rows
                        SELECT 	
                        			tsp.title AS Store,
                        			tp.title AS Product,
                        			rshp.child_id AS Product_ID,
                        			COALESCE(td.price, tp.unit_price_cost) AS Fixed_Cost
                        ----- Start from Shop / Product --- 63,088
                        FROM rshpp rshp 
                        
                        --- Step into shop
                        JOIN thread_shop tsp 
                        	ON tsp.id = rshp.parent_id -- Step to Supplier
                        
                        -- Step into product
                        JOIN thread_product tp ON tp.id = rshp.child_id
                        AND tp.id IN 
                        				(
                        				SELECT child_id
                        				FROM rshpp
                        				)
                        		AND
                        		tp.unit_price_cost > 0
                        
                        -- From Product to get Base Product ID 
                        LEFT JOIN rbpp ON rshp.child_id = rbpp.child_id
                        
                        -- From Base Product to Deal
                        LEFT JOIN rdbp ON rbpp.parent_id = rdbp.child_id
                        
                        -- From to deal x baseproduct
                        LEFT JOIN deal td ON td.id = rdbp.parent_id 
                        
                        ORDER BY 1,2
                        ;


                        """
        
        ### Execute the query
        cursor.execute(main_query)
        
        ### Fetch all rows
        rows = cursor.fetchall()
        
        ### Create a DataFrame from the fetched rows
        df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
        
        ### Append all dfs into the list
        df_list.append(df)
        
        ### Print out execution process
        print(f"Batch {batch_num + 1} appended to df_list")
    
    ### Concatenate all DataFrames into a single DataFrame
    df_cost_deal = pd.concat(df_list, ignore_index=True)


##### Step3: except block for error handeling    
except psycopg2.Error as e:
    print(f"Error: {e}")


    
##### Step4: Close the cursor and connection
finally:
    ### Close cursor
    if cursor:
        cursor.close()
    
    ### Close connection
    if conn:
        conn.close()


##### Step5: Output head for checking
df_check = df_cost_deal.head(60)


##### Step6: Drop duplicates by product id
df_cost_deal = df_cost_deal.drop(columns=['product_id']).drop_duplicates()


##### Step7: Save out deal cost result
fld_raw_data = r"os.path.join(DATA_DIR, '')"
file_cost_deal = fld_raw_data + "Product_Cost_24Q1_Deal.csv"

df_cost_deal.to_csv(file_cost_deal
                   ,index=False
                   ,header=True)


######################## End of Block 4: Get product cost from supplier deal ###########################





######################## Block 5: Get product cost from product table ###########################

##### Step1: Connect to your your database database
db_params = {
    'host': "your host",
    'database': "your database",
    'user': "your user name",
    'password': "your password"
}


# Part1-1: Pagination parameters
batch_size = 10000  # You can adjust the batch size as needed



##### Step2: Try block to fetch data by batch

### Step2-1: Find out total row counts of the query
try:
    ### Step1-1: Find out total row counts of the query
    # Establish a connection to the database
    conn = psycopg2.connect(**db_params)

    # Create a cursor object
    cursor = conn.cursor()

    # Execute the total count query
    total_count_query = f"""
                        SELECT COUNT(*)
                        FROM
                        (
                        SELECT  s.title AS Shop
                        		,tp.title AS Product
                        		,tp.price_cost AS Cost
                        FROM public.thread_product tp
                        LEFT JOIN public.shops s ON tp.shop_id = s.id
                        WHERE shop_id IN (
                        				  -- Group Filter
                        				  SELECT *
                        				  FROM get_shop_ids_in_group(('{SHOP_GROUP_UUID}')::UUID)
                        				  )
                              AND
                              removed IS false
                            	  AND
                            	  tp.price_cost > 0
                            	  AND
                            	  is_stocked IS true
                            	  AND
                            	  tp.title IS NOT Null
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

    ### End of Step1-1: Find out total row counts of the query

    df_list = []
    for batch_num in range(num_batches):
        # Calculate offset and limit for the current batch
        #offset: Represents the starting index of the current batch
        offset = batch_num * batch_size
        # limit: Represents the number of items to retrieve for the current batch
        limit = batch_size
        
        # Execute the main query with pagination
        main_query = ff"""
                    SELECT  s.title AS Shop
                    		,tp.title AS Product
                    		,tp.price_cost AS Cost
                    FROM public.thread_product tp
                    LEFT JOIN public.shops s ON tp.shop_id = s.id
                    WHERE shop_id IN (
                    				  -- Group Filter
                    				  SELECT *
                    				  FROM get_shop_ids_in_group(('{SHOP_GROUP_UUID}')::UUID)
                    				  )
                      AND
                      removed IS false
                 	  AND
                 	  tp.price_cost > 0
                 	  AND
                 	  is_stocked IS true
                 	  AND
                 	  tp.title IS NOT Null
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
    df_cost_prd = pd.concat(df_list, ignore_index=True)



##### Step3: except block for error handeling    
except psycopg2.Error as e:
    print(f"Error: {e}")



##### Step4: Close the cursor and connection
finally:
    ### Close cursor
    if cursor:
        cursor.close()
    
    ### Close connection
    if conn:
        conn.close()



##### Step5: Check & drop dupliacte
### The voided txn will create 2 rows to offset
duplicates = df_cost_prd[df_cost_prd.duplicated(keep=False)]

### Drop duplicates
df_cost_prd = df_cost_prd.drop_duplicates()



##### Step6: Change column names
df_cost_prd = df_cost_prd.rename(columns={ 'shop': 'Store'
                                          ,'product':'Product'
                                          , 'cost': 'Cost_Product'})



##### Step7: Remove negative cost
df_cost_prd.dtypes
df_cost_prd = df_cost_prd[df_cost_prd['Cost_Product']>0]



##### Step8: Save out result
fld_raw_data = r"os.path.join(DATA_DIR, '')"
file_cost_prd = fld_raw_data + "Product_Cost_24Q1_Product.csv"

df_cost_prd.to_csv(file_cost_prd
                   ,index=False
                   ,header=True)
######################## End of Block 5: Get product cost from product table ###########################





##################### Block 6: Combine cost tables ####################

##### Step1: Read cost file

### Step1-1: Cost from txn
fld_raw_data = r"os.path.join(DATA_DIR, '')"
file_cost_txn = fld_raw_data + "Product_Cost_24Q1_TXN.csv"
print(file_cost_txn)

df_cost_txn = pd.read_csv(file_cost_txn)



### Step1-2: Deal cost
fld_raw_data = r"os.path.join(DATA_DIR, '')"
file_cost_deal = fld_raw_data + "Product_Cost_24Q1_Deal.csv"
print(file_cost_deal)

df_cost_deal = pd.read_csv(file_cost_deal)

# Change column names
df_cost_deal = df_cost_deal.rename(columns={ 'store': 'Store'
                                          ,'product':'Product'
                                          , 'fixed_cost': 'Cost_Deal'})



##### Step2: Process dfs- drop dup/negative get min

### Step2-1: Cost from txn
df_cost_txn.dtypes
# Remove negative
df_cost_txn = df_cost_txn[df_cost_txn['Cost_Txn']>0]

# Get min
df_cost_txn = df_cost_txn.groupby(['Store','Product'])['Cost_Txn'].min().reset_index()

### Step2-2: Cost from deal
df_cost_deal.dtypes
# Remove negative
df_cost_deal = df_cost_deal[df_cost_deal['Cost_Deal']>0]

# Get min
df_cost_deal = df_cost_deal.groupby(['Store','Product'])['Cost_Deal'].min().reset_index()


### Step2-3: Cost from product
df_cost_prd.dtypes
# Remove negative
df_cost_prd = df_cost_prd[df_cost_prd['Cost_Product']>0]

# Get min
df_cost_prd = df_cost_prd.groupby(['Store','Product'])['Cost_Product'].min().reset_index()


##### Step3: Merge dfs into 1

### Merge cost from product & deal tables
df_cost = pd.merge(
                  df_cost_prd
                 ,df_cost_deal
                 ,how='left'
                 ,on=['Store', 'Product'])


### Merge with cost from transaction tables
df_cost = pd.merge(
                  df_cost
                 ,df_cost_txn
                 ,how='left'
                 ,on=['Store', 'Product'])


##### Step4: Take lowest cost as final cost
# Use 'Cost_deal' if not null, otherwise use 'Cost_txn'
# Modified as take the lowers cost from 2 columns
# Replace NaN values in 'Cost_Deal' with a very large number to make sure nan wont be selected
df_cost['Cost_Lowest'] = np.minimum.reduce([
                                    df_cost['Cost_Deal'].fillna(np.inf)
                                    ,df_cost['Cost_Txn'].fillna(np.inf)
                                    ,df_cost['Cost_Product'].fillna(np.inf)])



##### Step5: Drop the intermediate columns if needed
df_cost = df_cost[['Store', 'Product', 'Cost_Lowest']]



##### Step6: Save out result for validation
# Define folders & path
file_cost_lowest = fld_raw_data + "Product_Cost_24Q1_Lowest.csv"
print(file_cost_lowest)

# Save out result
df_cost.to_csv(file_cost_lowest,header=True,index=False)


##################### End of Block 6: Combine cost tables ####################



