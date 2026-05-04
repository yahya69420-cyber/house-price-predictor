import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
COLS_TO_DROP = ["PoolQC", "MiscFeature", "Alley", "Fence"]


def preprocess(df: pd.DataFrame, train_columns=None) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], inplace=True)

    if "FireplaceQu" in df.columns:
        df["FireplaceQu"] = df["FireplaceQu"].fillna("None")
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # Fill remaining nulls
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    df = pd.get_dummies(df, columns=cat_cols)

    if train_columns is not None:
        df = df.reindex(columns=train_columns, fill_value=0)

    return df


@st.cache_resource(show_spinner="Training model …")
def train_model(train_bytes: bytes):
    df = pd.read_csv(io.BytesIO(train_bytes))
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])
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
    st.image("https://img.icons8.com/fluency/96/home.png", width=80)
    st.title("🏠 House Price AI")
    st.markdown("Upload a **training CSV** (must contain `SalePrice`) to train the model.")

    train_file = st.file_uploader("Training CSV", type="csv", key="train")

    if train_file:
        st.success("Training file ready ✔")

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. Upload `train.csv`\n"
        "2. Model trains automatically\n"
        "3. Explore metrics & charts\n"
        "4. Upload `test.csv` for batch predictions\n"
        "5. Download `submission.csv`"
    )

# ── Main content ──────────────────────────────────────────────────────────────
st.title("🏠 House Sale Price Predictor")
st.markdown(
    "A **Random Forest** model trained on the Kaggle House Prices dataset. "
    "Upload your training data on the left to get started."
)

# ── Required columns info ─────────────────────────────────────────────────────
REQUIRED_COLUMNS = {
    "Id": "Unique identifier for each house",
    "MSSubClass": "Type of dwelling (e.g. 20 = 1-story 1946+)",
    "MSZoning": "General zoning classification (e.g. RL = Residential Low Density)",
    "LotFrontage": "Linear feet of street connected to property",
    "LotArea": "Lot size in square feet",
    "Street": "Type of road access (Grvl / Pave)",
    "LotShape": "General shape of property (Reg / IR1 / IR2 / IR3)",
    "LandContour": "Flatness of the property",
    "Utilities": "Type of utilities available",
    "LotConfig": "Lot configuration (Inside / Corner / CulDSac etc.)",
    "LandSlope": "Slope of property (Gtl / Mod / Sev)",
    "Neighborhood": "Physical location within Ames city limits",
    "BldgType": "Type of dwelling (1Fam / 2FmCon / Duplex etc.)",
    "HouseStyle": "Style of dwelling (1Story / 2Story etc.)",
    "OverallQual": "Overall material and finish quality (1–10)",
    "OverallCond": "Overall condition rating (1–10)",
    "YearBuilt": "Original construction year",
    "YearRemodAdd": "Remodel year (same as construction if no remodel)",
    "RoofStyle": "Type of roof (Flat / Gable / Hip etc.)",
    "Exterior1st": "Exterior covering on house",
    "MasVnrType": "Masonry veneer type",
    "MasVnrArea": "Masonry veneer area in square feet",
    "ExterQual": "Exterior material quality (Ex/Gd/TA/Fa/Po)",
    "ExterCond": "Current condition of exterior material",
    "Foundation": "Type of foundation (BrkTil / CBlock / PConc etc.)",
    "BsmtQual": "Height of basement (Ex/Gd/TA/Fa/Po/NA)",
    "BsmtCond": "General condition of basement",
    "BsmtExposure": "Walkout or garden level basement walls",
    "BsmtFinType1": "Quality of basement finished area",
    "BsmtFinSF1": "Type 1 finished square feet",
    "BsmtUnfSF": "Unfinished square feet of basement area",
    "TotalBsmtSF": "Total square feet of basement area",
    "Heating": "Type of heating",
    "HeatingQC": "Heating quality and condition",
    "CentralAir": "Central air conditioning (Y / N)",
    "Electrical": "Electrical system type",
    "1stFlrSF": "First floor square feet",
    "2ndFlrSF": "Second floor square feet",
    "GrLivArea": "Above grade (ground) living area square feet",
    "BsmtFullBath": "Basement full bathrooms",
    "BsmtHalfBath": "Basement half bathrooms",
    "FullBath": "Full bathrooms above grade",
    "HalfBath": "Half bathrooms above grade",
    "BedroomAbvGr": "Number of bedrooms above basement level",
    "KitchenAbvGr": "Number of kitchens above grade",
    "KitchenQual": "Kitchen quality",
    "TotRmsAbvGrd": "Total rooms above grade (not including bathrooms)",
    "Functional": "Home functionality rating",
    "Fireplaces": "Number of fireplaces",
    "FireplaceQu": "Fireplace quality (leave blank if no fireplace)",
    "GarageType": "Garage location (Attchd / Detchd / BuiltIn etc.)",
    "GarageYrBlt": "Year garage was built",
    "GarageFinish": "Interior finish of garage",
    "GarageCars": "Size of garage in car capacity",
    "GarageArea": "Size of garage in square feet",
    "GarageQual": "Garage quality",
    "GarageCond": "Garage condition",
    "PavedDrive": "Paved driveway (Y / P / N)",
    "WoodDeckSF": "Wood deck area in square feet",
    "OpenPorchSF": "Open porch area in square feet",
    "EnclosedPorch": "Enclosed porch area in square feet",
    "MoSold": "Month sold",
    "YrSold": "Year sold",
    "SaleType": "Type of sale (WD / New / COD etc.)",
    "SaleCondition": "Condition of sale (Normal / Abnorml / Partial etc.)",
    "SalePrice": "⭐ TARGET — the price to predict (only in training file)",
}

with st.expander("📋 Required CSV Columns — click to expand", expanded=False):
    st.markdown("Your CSV must contain these columns for the model to work correctly. "
                "Columns marked ⭐ are only needed in the **training** file.")

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

# Train / load from cache
train_bytes = train_file.read()
model, train_cols, X_test, y_test, preds, mae, r2 = train_model(train_bytes)

# ── Metrics row ───────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Mean Absolute Error", f"${mae:,.0f}")
col2.metric("R² Accuracy", f"{r2 * 100:.2f}%")
col3.metric("Features used", len(train_cols))

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🔍 Feature Importance", "📂 Batch Predict"])

# ── Tab 1 – Performance charts ────────────────────────────────────────────────
with tab1:
    st.subheader("Actual vs Predicted Sale Price")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_test, preds, alpha=0.45, color="steelblue", edgecolors="none", s=30)
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

# ── Tab 2 – Feature importance ────────────────────────────────────────────────
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

# ── Tab 3 – Batch predictions ─────────────────────────────────────────────────
with tab3:
    st.subheader("Predict on New Data")
    st.markdown("Upload a `test.csv` (without `SalePrice`) to generate predictions.")

    test_file = st.file_uploader("Test CSV", type="csv", key="test")

    if test_file:
        test_df_raw = pd.read_csv(test_file)
        id_col = test_df_raw["Id"] if "Id" in test_df_raw.columns else None

        test_df = test_df_raw.drop(columns=["SalePrice"], errors="ignore")
        if "Id" in test_df.columns:
            test_df = test_df.drop(columns=["Id"])

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
    else:
        st.info("Upload a test CSV above to generate a submission file.")

st.markdown("---")
st.caption("Built with Streamlit · Random Forest Regressor · Kaggle House Prices dataset")