import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import io

st.set_page_config(page_title="House Price Predictor", page_icon="house", layout="wide")


def preprocess(df, train_columns=None):
    df = df.copy()
    df.drop(columns=[c for c in ["PoolQC","MiscFeature","Alley","Fence"] if c in df.columns], inplace=True)
    if "FireplaceQu" in df.columns:
        df["FireplaceQu"] = df["FireplaceQu"].fillna("None")
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = pd.to_numeric(df["LotFrontage"], errors="coerce")
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
    for col in df.columns:
        num = pd.to_numeric(df[col], errors="coerce")
        if num.notna().mean() > 0.5:
            df[col] = num
        else:
            df[col] = df[col].astype(str)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median()
            df[col] = df[col].fillna(med if pd.notna(med) else 0)
        else:
            df[col] = df[col].fillna("Unknown").astype(str)
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)
    if train_columns is not None:
        df = df.reindex(columns=train_columns, fill_value=0)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df.astype(float)


@st.cache_resource(show_spinner="Training model...")
def train_model(train_bytes):
    try:
        df = pd.read_csv(io.BytesIO(train_bytes))
    except Exception:
        df = pd.read_csv(io.BytesIO(train_bytes), on_bad_lines="skip", encoding="latin1")
    df["SalePrice"] = pd.to_numeric(df["SalePrice"], errors="coerce")
    df = df.dropna(subset=["SalePrice"])
    y = df["SalePrice"].astype(float)
    X = df.drop(columns=["SalePrice", "Id"], errors="ignore")
    X = preprocess(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return model, X_train.columns.tolist(), X_test, y_test, preds, mae, r2


# Sidebar
with st.sidebar:
    st.title("House Price AI")
    st.markdown("Optionally upload your own `train.csv` to retrain the model.")
    train_file = st.file_uploader("Training CSV (optional)", type="csv", key="train")
    if train_file:
        st.success("Custom training file ready")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1. Adjust house features in the form\n2. Click **Predict Price**\n3. Get instant estimate\n4. Optionally upload CSV for batch predictions")

# Header
st.title("House Sale Price Predictor")
st.markdown("Fill in the house details below and get an **instant price estimate**.")

# Load training data
if train_file:
    train_bytes = train_file.read()
else:
    try:
        with open("train.csv", "rb") as f:
            train_bytes = f.read()
    except FileNotFoundError:
        st.error("No training data found. Please upload a train.csv in the sidebar.")
        st.stop()

try:
    model, train_cols, X_test, y_test, preds, mae, r2 = train_model(train_bytes)
except Exception as e:
    st.error(f"Could not train the model: {e}")
    st.markdown("Please make sure your train.csv is a valid Kaggle House Prices dataset.")
    st.stop()

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Model MAE", f"${mae:,.0f}")
col2.metric("R2 Accuracy", f"{r2 * 100:.2f}%")
col3.metric("Features", len(train_cols))

st.markdown("---")

# Input form
st.subheader("Enter House Details")
st.markdown("Adjust the values below then click **Predict Price**.")

with st.form("house_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Size & Layout")
        gr_liv_area   = st.number_input("Above Ground Living Area (sq ft)", 500, 6000, 1500, step=50)
        total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800, step=50)
        first_flr_sf  = st.number_input("1st Floor Area (sq ft)", 300, 4000, 1000, step=50)
        second_flr_sf = st.number_input("2nd Floor Area (sq ft)", 0, 2000, 0, step=50)
        lot_area      = st.number_input("Lot Area (sq ft)", 1000, 100000, 8000, step=500)
        lot_frontage  = st.number_input("Lot Frontage (linear ft)", 0, 300, 70, step=5)
        garage_area   = st.number_input("Garage Area (sq ft)", 0, 1500, 400, step=50)
        wood_deck_sf  = st.number_input("Wood Deck Area (sq ft)", 0, 1000, 0, step=25)
        open_porch_sf = st.number_input("Open Porch Area (sq ft)", 0, 600, 0, step=10)

    with c2:
        st.markdown("#### Quality & Condition")
        overall_qual  = st.slider("Overall Quality (1-10)", 1, 10, 6)
        overall_cond  = st.slider("Overall Condition (1-10)", 1, 10, 5)
        exter_qual    = st.selectbox("Exterior Quality", ["Ex","Gd","TA","Fa","Po"], index=2)
        kitchen_qual  = st.selectbox("Kitchen Quality", ["Ex","Gd","TA","Fa","Po"], index=1)
        bsmt_qual     = st.selectbox("Basement Quality", ["Ex","Gd","TA","Fa","Po","NA"], index=1)
        heating_qc    = st.selectbox("Heating Quality", ["Ex","Gd","TA","Fa","Po"], index=0)
        fireplace_qu  = st.selectbox("Fireplace Quality", ["None","Ex","Gd","TA","Fa","Po"], index=0)
        garage_finish = st.selectbox("Garage Finish", ["Fin","RFn","Unf","NA"], index=0)
        garage_qual   = st.selectbox("Garage Quality", ["Ex","Gd","TA","Fa","Po","NA"], index=2)

    with c3:
        st.markdown("#### Structure & Features")
        year_built  = st.number_input("Year Built", 1870, 2025, 1995)
        year_remod  = st.number_input("Year Remodelled", 1870, 2025, 1995)
        full_bath   = st.selectbox("Full Bathrooms", [0,1,2,3,4], index=2)
        half_bath   = st.selectbox("Half Bathrooms", [0,1,2], index=0)
        bedroom     = st.selectbox("Bedrooms Above Ground", [0,1,2,3,4,5,6], index=3)
        tot_rms     = st.selectbox("Total Rooms Above Ground", [2,3,4,5,6,7,8,9,10,11,12], index=5)
        fireplaces  = st.selectbox("Fireplaces", [0,1,2,3], index=0)
        garage_cars = st.selectbox("Garage Capacity (cars)", [0,1,2,3,4], index=2)
        central_air = st.selectbox("Central Air Conditioning", ["Y","N"], index=0)

    st.markdown("#### Location & Type")
    lc1, lc2, lc3, lc4 = st.columns(4)
    with lc1:
        neighborhood = st.selectbox("Neighborhood", [
            "NAmes","CollgCr","OldTown","Edwards","Somerst","NridgHt","Gilbert",
            "Sawyer","NWAmes","SawyerW","BrkSide","Crawfor","Mitchel","NoRidge",
            "Timber","IDOTRR","ClearCr","StoneBr","SWISU","Blmngtn","MeadowV",
            "Veenker","NPkVill","Blueste","BrDale","Greens","GrnHill","Landmrk"
        ], index=0)
        ms_zoning = st.selectbox("Zoning", ["RL","RM","FV","RH","C (all)"], index=0)
    with lc2:
        bldg_type   = st.selectbox("Building Type", ["1Fam","2fmCon","Duplex","TwnhsE","Twnhs"], index=0)
        house_style = st.selectbox("House Style", ["1Story","2Story","1.5Fin","1.5Unf","SFoyer","SLvl","2.5Unf","2.5Fin"], index=0)
    with lc3:
        foundation = st.selectbox("Foundation", ["PConc","CBlock","BrkTil","Wood","Slab","Stone"], index=0)
        roof_style = st.selectbox("Roof Style", ["Gable","Hip","Flat","Gambrel","Mansard","Shed"], index=0)
    with lc4:
        sale_type = st.selectbox("Sale Type", ["WD","New","COD","Con","ConLw","ConLI","ConLD","Oth","CWD"], index=0)
        sale_cond = st.selectbox("Sale Condition", ["Normal","Abnorml","Partial","AdjLand","Alloca","Family"], index=0)

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

# Prediction
if submitted:
    input_data = {
        "MSSubClass": 20, "MSZoning": ms_zoning, "LotFrontage": lot_frontage,
        "LotArea": lot_area, "Street": "Pave", "LotShape": "Reg",
        "LandContour": "Lvl", "Utilities": "AllPub", "LotConfig": "Inside",
        "LandSlope": "Gtl", "Neighborhood": neighborhood, "Condition1": "Norm",
        "Condition2": "Norm", "BldgType": bldg_type, "HouseStyle": house_style,
        "OverallQual": overall_qual, "OverallCond": overall_cond,
        "YearBuilt": year_built, "YearRemodAdd": year_remod,
        "RoofStyle": roof_style, "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd", "Exterior2nd": "VinylSd",
        "MasVnrType": "None", "MasVnrArea": 0, "ExterQual": exter_qual,
        "ExterCond": "TA", "Foundation": foundation, "BsmtQual": bsmt_qual,
        "BsmtCond": "TA", "BsmtExposure": "No", "BsmtFinType1": "GLQ",
        "BsmtFinSF1": total_bsmt_sf * 0.5, "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0, "BsmtUnfSF": total_bsmt_sf * 0.5,
        "TotalBsmtSF": total_bsmt_sf, "Heating": "GasA", "HeatingQC": heating_qc,
        "CentralAir": central_air, "Electrical": "SBrkr",
        "1stFlrSF": first_flr_sf, "2ndFlrSF": second_flr_sf,
        "LowQualFinSF": 0, "GrLivArea": gr_liv_area,
        "BsmtFullBath": 1, "BsmtHalfBath": 0,
        "FullBath": full_bath, "HalfBath": half_bath,
        "BedroomAbvGr": bedroom, "KitchenAbvGr": 1, "KitchenQual": kitchen_qual,
        "TotRmsAbvGrd": tot_rms, "Functional": "Typ", "Fireplaces": fireplaces,
        "FireplaceQu": fireplace_qu, "GarageType": "Attchd",
        "GarageYrBlt": year_built, "GarageFinish": garage_finish,
        "GarageCars": garage_cars, "GarageArea": garage_area,
        "GarageQual": garage_qual, "GarageCond": "TA", "PavedDrive": "Y",
        "WoodDeckSF": wood_deck_sf, "OpenPorchSF": open_porch_sf,
        "EnclosedPorch": 0, "3SsnPorch": 0, "ScreenPorch": 0,
        "PoolArea": 0, "MiscVal": 0, "MoSold": 6, "YrSold": 2010,
        "SaleType": sale_type, "SaleCondition": sale_cond,
    }
    input_df = pd.DataFrame([input_data])
    try:
        input_processed = preprocess(input_df, train_columns=train_cols)
        predicted_price = model.predict(input_processed)[0]
        st.markdown("---")
        st.success(f"### Estimated Sale Price: ${predicted_price:,.0f}")
        min_price, max_price = 50000, 800000
        pct = min(max((predicted_price - min_price) / (max_price - min_price), 0), 1.0)
        st.progress(pct)
        c_lo, c_mid, c_hi = st.columns(3)
        c_lo.caption("$50,000")
        c_mid.caption(f"<-- ${predicted_price:,.0f} -->")
        c_hi.caption("$800,000+")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")

# Performance charts
with st.expander("Model Performance Charts", expanded=False):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, preds, alpha=0.45, color="steelblue", s=30)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    axes[0].plot(lims, lims, "r--", lw=1.8, label="Perfect prediction")
    axes[0].set_xlabel("Actual ($)")
    axes[0].set_ylabel("Predicted ($)")
    axes[0].set_title("Actual vs Predicted")
    axes[0].legend()
    residuals = np.array(y_test) - preds
    axes[1].hist(residuals, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--", lw=1.8)
    axes[1].set_xlabel("Residual ($)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residuals Distribution")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with st.expander("Top 20 Feature Importances", expanded=False):
    importances = pd.Series(model.feature_importances_, index=train_cols)
    top20 = importances.nlargest(20).sort_values()
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, 20))
    top20.plot(kind="barh", ax=ax2, color=colors)
    ax2.set_xlabel("Importance Score")
    ax2.set_title("Feature Importance (Random Forest)")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

with st.expander("Batch Predict from CSV", expanded=False):
    st.markdown("Upload a `test.csv` to generate predictions for multiple houses at once.")
    test_file = st.file_uploader("Test CSV", type="csv", key="test")
    if test_file:
        try:
            test_df_raw = pd.read_csv(test_file)
            id_col = test_df_raw["Id"] if "Id" in test_df_raw.columns else None
            test_df = test_df_raw.drop(columns=["SalePrice","Id"], errors="ignore")
            test_processed = preprocess(test_df, train_columns=train_cols)
            test_preds = model.predict(test_processed)
            submission = pd.DataFrame({"SalePrice": test_preds})
            if id_col is not None:
                submission.insert(0, "Id", id_col.values)
            st.success(f"Predictions generated for {len(submission):,} rows.")
            st.dataframe(submission.head(20), use_container_width=True)
            st.download_button("Download submission.csv",
                               submission.to_csv(index=False).encode(),
                               "submission.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not process test file: {e}")
    else:
        st.info("Upload a test CSV above.")

st.markdown("---")
st.caption("Built with Streamlit | Random Forest Regressor | Kaggle House Prices dataset")
