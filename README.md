# Price Optimization with Cannibalisation

A Python pipeline for retail **price optimization** that incorporates **cannibalisation** (cross-product effects within a category).  
Sensitive business logic is abstracted behind a pluggable strategy layer; this repo ships with safe defaults and **dummy data only**.

## Project Structure

```
01.PythonCode/
  └─ src/                     # Main pipeline scripts (sanitized & strategy-ready)
02.DataSet/                   # Dummy dataset (fully replaced; no real data)
03.Result/
  └─ deidentified/            # Sample outputs to illustrate result schema & metrics
99.Integration/               # (Optional) final artifacts for system integration
.env.example                  # Example environment variables (no secrets)
.gitignore                    # Ignore rules; keep code & dummy, exclude real secrets
requirements.txt             # Python dependencies
README.md
```

## Key Features
- End-to-end flow from **data extraction** to **profit-based price simulation**.
- **Cannibalisation-aware** modelling via a replaceable strategy interface (keep proprietary rules private).
- All paths, table names, and IDs are **parameterized** via environment variables.
- Ships with **dummy inputs** and **de-identified example outputs** for clarity.

## Pipeline Overview (scripts in `01.PythonCode/src/`)
1. **Step 0 – Daily Sales Extraction**  
   Extract transactional data (paged/batched) from your DB into a local dataset (dates, stores, products, prices, quantities).

2. **Step 1 – Weekly Correlations & Candidate Pairs**  
   Aggregate to weekly level and compute correlations to propose product pairs with potential cannibalisation relationships.

3. **Step 2 – Demand Modelling (with Cannibalisation)**  
   Fit baseline price–demand models per product and extend with cannibalisation features from Step 1; control multicollinearity; export coefficients and diagnostics.

4. **Step 3 – Product Cost Assembly**  
   Merge cost sources (e.g., transaction-implied costs and deal/contract costs) and select a feasible minimum cost per product over the relevant period.

5. **Step 4 – Price Simulation & Profit Maximization**  
   Generate price grids per product, predict demand, compute profit `(price − cost) × qty`, filter infeasible ranges, and identify **profit-optimal** price points.

## Quick Start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` → `.env` and adjust as needed (do **not** commit `.env`).  
Then run from the project root:

```bash
python 01.PythonCode/src/Cannibalization_Step0_Get_DailySale_From_HB_open_sanitized.py
python 01.PythonCode/src/Cannibalization_Step1_Price_CorrelationCoefficient_ByWeek_open_sanitized.py
python 01.PythonCode/src/Cannibalization_Step2_Form_RegModel_InLoop_open_sanitized.py
python 01.PythonCode/src/Cannibalization_Step3_Product_Cost_open_sanitized.py
python 01.PythonCode/src/Cannibalization_Step4_PriceSimulation_open_sanitized.py
```

## Configuration (Environment Variables)
These are read inside the scripts via `os.getenv` (defaults provided). Edit your `.env` as needed:

- `DB_SCHEMA_PUBLIC` (default: `public`)  
- `TABLE_TRANSACTION_ITEMS` (default: `transaction_items_partitioned`)  
- `TABLE_RELATIONSHIPS` (default: `relationships`)  
- `TABLE_THREAD_DEAL` (default: `thread_deal`)  
- `SHOP_GROUP_UUID` (default: `00000000-0000-0000-0000-000000000000`)  
- `DATA_DIR` (e.g., `02.DataSet`)  
- `INTERNAL_SHARE_PATH` (optional; local/share path if used)  
- `REF_STORE_CSV` (optional; path to store reference file if used)

## Inputs & Outputs

- **Inputs**: `02.DataSet/` (dummy only). Align your local data structure/column names with the scripts if you run with real data internally.  
- **Sample Outputs**: `03.Result/deidentified/` shows **result schema** and typical metrics (e.g., weekly correlations, model summaries, costs table, and optimal prices).  
- **System Integration**: `99.Integration/` is a placeholder for final export artefacts when integrating with downstream systems.

## Strategy Layer (Keep Your Logic Private)
The scripts try to import a `pricing_strategy` module. If present in your environment, it overrides the default behaviour for:
- selecting cannibalisation pairs,
- building demand models,
- choosing feasible costs,
- simulating prices.

This lets you keep proprietary modelling decisions outside the public repo while preserving the same interfaces.

## License & Contributions
This open-sanitized version is intended for collaboration without revealing trade secrets.  
Contributions are welcome—please avoid committing any real data or confidential rules.
