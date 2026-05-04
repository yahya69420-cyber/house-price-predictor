import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import io

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# ── Required columns ──────────────────────────────────────────────────────────
REQUIRED_COLUMNS = {
    "Id": "Unique identifier for each house",
    "MSSubClass": "Type of dwelling (e.g. 20 = 1-story 1946+)",
    "MSZoning": "General zoning classification (e.g. RL = Residential Low Density)",
    "LotFrontage": "Linear feet of street connected to property",
    "LotArea": "Lot size in square feet",
    "Street": "Type of road access (Grvl / Pave)",
    "LotShape": "General shape of property",
    "LandContour": "Flatness of the property",
    "Neighborhood": "Physical location within Ames city limits",
    "BldgType": "Type of dwelling (1Fam / 2FmCon / Duplex etc.)",
    "HouseStyle": "Style of dwelling (1Story / 2Story etc.)",
    "OverallQual": "Overall material and finish quality (1–10)",
    "OverallCond": "Overall condition rating (1–10)",
    "YearBuilt": "Original construction year",
    "YearRemodAdd": "Remodel year",
    "RoofStyle": "Type of roof",
    "Exterior1st": "Exterior covering on house",
    "MasVnrType": "Masonry veneer type",
    "MasVnrArea": "Masonry veneer area in square feet",
    "ExterQual": "Exterior material quality (Ex/Gd/TA/Fa/Po)",
    "Foundation": "Type of foundation",
    "BsmtQual": "Height of basement",
    "BsmtFinSF1": "Type 1 finished square feet",
    "TotalBsmtSF": "Total square feet of basement area",
    "CentralAir": "Central air conditioning (Y / N)",
    "1stFlrSF": "First floor square feet",
    "2ndFlrSF": "Second floor square feet",
    "GrLivArea": "Above grade living area square feet",
    "FullBath": "Full bathrooms above grade",
    "HalfBath": "Half bathrooms above grade",
    "BedroomAbvGr": "Number of bedrooms above basement level",
    "KitchenQual": "Kitchen quality",
    "TotRmsAbvGrd": "Total rooms above grade",
    "Fireplaces": "Number of fireplaces",
    "FireplaceQu": "Fireplace quality (leave blank if none)",
    "GarageType": "Garage location",
    "GarageYrBlt": "Year garage was built",
    "GarageCars": "Size of garage in car capacity",
    "GarageArea": "Size of garage in square feet",
    "WoodDeckSF": "Wood deck area in square feet",
    "OpenPorchSF": "Open porch area in square feet",
    "MoSold": "Month sold",
    "YrSold": "Year sold",
    "SaleType": "Type of sale",
    "SaleCondition": "Condition of sale",
    "SalePrice": "⭐ TARGET — price to predict (training file only)",
}

NUMERIC_COLS = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "TotalBsmtSF",
    "1stFlrSF", "2ndFlrSF", "GrLivArea", "GarageArea", "WoodDeckSF",
    "OpenPorchSF", "MoSold", "YrSold", "YearBuilt", "YearRemodAdd",
    "OverallQual", "OverallCond", "GarageCars", "FullBath", "HalfBath",
    "BedroomAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt",
    "MSSubClass", "SalePrice"
]

# ── Data validation ───────────────────────────────────────────────────────────
def validate_dataframe(df: pd.DataFrame, is_train: bool = True) -> list:
    errors = []

    # Check required columns
    required = list(REQUIRED_COLUMNS.keys())
    if not is_train:
        required = [c for c in required if c != "SalePrice"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")

    # Check SalePrice for training
    if is_train and "SalePrice" in df.columns:
        if not pd.to_numeric(df["SalePrice"], errors="coerce").notna().all():
            errors.append("SalePrice column contains non-numeric values — it must be numbers only.")

    # Check for completely empty dataframe
    if df.empty:
        errors.append("The uploaded file is empty.")

    # Check numeric columns that should not be text
    for col in NUMERIC_COLS:
        if col in df.columns and col != "SalePrice":
            numeric_check = pd.to_numeric(df[col], errors="coerce")
            bad_pct = numeric_check.isna().mean()
            if bad_pct > 0.5:
                errors.append(f"Column '{col}' should be numeric but contains mostly text values.")

    return errors


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame, train_columns=None) -> pd.DataFrame:
    df = df.copy()

    # Drop high-null columns
    drop_cols = ["PoolQC", "MiscFeature", "Alley", "Fence"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Special fills
    if "FireplaceQu" in df.columns:
        df["FireplaceQu"] = df["FireplaceQu"].fillna("None")
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = pd.to_numeric(df["LotFrontage"], errors="coerce")
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # Separate numeric and categorical columns explicitly
    numeric_cols = []
    cat_cols = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= len(df) * 0.5:
            df[col] = converted
            numeric_cols.append(col)
        else:
            df[col] = df[col].astype(str)
            cat_cols.append(col)

    # Fill nulls
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    # One-hot encode
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)

    # Bool → int
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # Align with training columns
    if train_columns is not None:
        df = df.reindex(columns=train_columns, fill_value=0)

    # Force everything to float
    df = df.astype(float)

    return df


# ── Model training ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model …")
def train_model(train_bytes: bytes):
    df = pd.read_csv(io.BytesIO(train_bytes))
    y = df["SalePrice"].astype(float)
    X = df.drop(columns=["SalePrice", "Id"], errors="ignore")
    X = preprocess(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return model, X_train.columns.tolist(), X_test, y_test, preds, mae, r2


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏠 House Price AI")
    st.markdown("Upload a **training CSV** (must contain `SalePrice`) to train the model.")
    train_file = st.file_uploader("Training CSV", type="csv", key="train")
    if train_file:
        st.success("Training file ready ✔")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1. Upload `train.csv`\n2. Model trains automatically\n3. Explore metrics & charts\n4. Upload `test.csv` for batch predictions\n5. Download `submission.csv`")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏠 House Sale Price Predictor")
st.markdown("A **Random Forest** model trained on the Kaggle House Prices dataset.")

# ── Required columns expander ─────────────────────────────────────────────────
with st.expander("📋 Required CSV Columns — click to expand", expanded=False):
    st.markdown("Your CSV must contain these columns. ⭐ = only needed in the **training** file.")
    st.warning("⚠️ Only use clean datasets from the [Kaggle House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). Custom or modified CSVs may cause errors.")
    col_a, col_b = st.columns(2)
    items = list(REQUIRED_COLUMNS.items())
    half = len(items) // 2
    with col_a:
        for col, desc in items[:half]:
            st.markdown(f"**`{col}`** — {desc}")
    with col_b:
        for col, desc in items[half:]:
            st.markdown(f"**`{col}`** — {desc}")

st.markdown("---")

if not train_file:
    st.info("👈  Upload `train.csv` in the sidebar to begin.")
    st.stop()

# ── Validate training file ────────────────────────────────────────────────────
train_bytes = train_file.read()
train_df_check = pd.read_csv(io.BytesIO(train_bytes))
validation_errors = validate_dataframe(train_df_check, is_train=True)

if validation_errors:
    st.error("❌ Your dataset has the following issues. Please upload a clean dataset:")
    for err in validation_errors:
        st.markdown(f"- {err}")
    st.info("💡 Download the correct dataset from [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)")
    st.stop()

# ── Train model ───────────────────────────────────────────────────────────────
try:
    model, train_cols, X_test, y_test, preds, mae, r2 = train_model(train_bytes)
except Exception as e:
    st.error("❌ Could not train the model. Your dataset may be corrupted or incorrectly formatted.")
    st.markdown("**Please make sure you are using the official Kaggle House Prices `train.csv` file.**")
    st.info("💡 Download it from [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)")
    st.stop()

# ── Metrics ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Mean Absolute Error", f"${mae:,.0f}")
col2.metric("R² Accuracy", f"{r2 * 100:.2f}%")
col3.metric("Features used", len(train_cols))

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🔍 Feature Importance", "📂 Batch Predict"])

with tab1:
    st.subheader("Actual vs Predicted Sale Price")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_test, preds, alpha=0.45, color="steelblue", s=30)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    axes[0].plot(lims, lims, "r--", lw=1.8, label="Perfect prediction")
    axes[0].set_xlabel("Actual Sale Price ($)")
    axes[0].set_ylabel("Predicted Sale Price ($)")
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

with tab2:
    st.subheader("Top 20 Most Important Features")
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

with tab3:
    st.subheader("Predict on New Data")
    st.markdown("Upload a `test.csv` (without `SalePrice`) to generate predictions.")
    test_file = st.file_uploader("Test CSV", type="csv", key="test")

    if test_file:
        test_df_raw = pd.read_csv(test_file)

        # Validate test file
        test_errors = validate_dataframe(test_df_raw, is_train=False)
        if test_errors:
            st.error("❌ Your test dataset has issues:")
            for err in test_errors:
                st.markdown(f"- {err}")
            st.stop()

        try:
            id_col = test_df_raw["Id"] if "Id" in test_df_raw.columns else None
            test_df = test_df_raw.drop(columns=["SalePrice", "Id"], errors="ignore")
            test_processed = preprocess(test_df, train_columns=train_cols)
            test_preds = model.predict(test_processed)

            submission = pd.DataFrame({"SalePrice": test_preds})
            if id_col is not None:
                submission.insert(0, "Id", id_col.values)

            st.success(f"✅ Predictions generated for {len(submission):,} rows.")
            st.dataframe(submission.head(20), use_container_width=True)

            csv_bytes = submission.to_csv(index=False).encode()
            st.download_button(
                label="⬇️  Download submission.csv",
                data=csv_bytes,
                file_name="submission.csv",
                mime="text/csv",
            )
        except Exception:
            st.error("❌ Could not generate predictions. Make sure your test CSV is clean and matches the required columns.")
    else:
        st.info("Upload a test CSV above to generate a submission file.")

st.markdown("---")
st.caption("Built with Streamlit · Random Forest Regressor · Kaggle House Prices dataset")
