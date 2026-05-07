import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="House Price Predictor", page_icon="house", layout="wide")

# ── Generate synthetic training data based on Kaggle distributions ────────────
@st.cache_resource(show_spinner="Loading model...")
def get_model():
    np.random.seed(42)
    n = 2000

    overall_qual  = np.random.choice(range(1, 11), n, p=[0.01,0.02,0.04,0.07,0.13,0.17,0.22,0.17,0.10,0.07])
    gr_liv_area   = np.random.normal(1515, 525, n).clip(400, 5000)
    total_bsmt_sf = np.random.normal(1057, 439, n).clip(0, 3000)
    year_built    = np.random.randint(1900, 2011, n)
    garage_cars   = np.random.choice([0,1,2,3,4], n, p=[0.05,0.15,0.55,0.22,0.03])
    garage_area   = garage_cars * 200 + np.random.normal(0, 50, n)
    full_bath     = np.random.choice([0,1,2,3], n, p=[0.01,0.27,0.60,0.12])
    tot_rms       = np.random.choice(range(2,13), n)
    fireplaces    = np.random.choice([0,1,2,3], n, p=[0.47,0.42,0.10,0.01])
    lot_area      = np.random.normal(10500, 9981, n).clip(1000, 50000)
    first_flr_sf  = np.random.normal(1163, 392, n).clip(300, 4000)
    open_porch_sf = np.random.normal(47, 67, n).clip(0, 500)
    year_remod    = np.random.randint(1950, 2011, n)
    overall_cond  = np.random.choice(range(1,10), n, p=[0.01,0.01,0.02,0.03,0.45,0.10,0.27,0.06,0.05])
    central_air   = np.random.choice([0, 1], n, p=[0.07, 0.93])
    neighborhood_score = np.random.normal(0, 15000, n)

    # Realistic price formula
    price = (
        overall_qual  * 35000 +
        gr_liv_area   * 120 +
        total_bsmt_sf * 40 +
        (year_built - 1900) * 800 +
        garage_cars   * 15000 +
        full_bath     * 12000 +
        fireplaces    * 10000 +
        lot_area      * 3.0 +
        first_flr_sf  * 40 +
        central_air   * 20000 +
        neighborhood_score +
        np.random.normal(0, 15000, n)
    ).clip(40000, 1500000)

    X = pd.DataFrame({
        "OverallQual":  overall_qual,
        "GrLivArea":    gr_liv_area,
        "TotalBsmtSF":  total_bsmt_sf,
        "YearBuilt":    year_built,
        "GarageCars":   garage_cars,
        "GarageArea":   garage_area,
        "FullBath":     full_bath,
        "TotRmsAbvGrd": tot_rms,
        "Fireplaces":   fireplaces,
        "LotArea":      lot_area,
        "1stFlrSF":     first_flr_sf,
        "OpenPorchSF":  open_porch_sf,
        "YearRemodAdd": year_remod,
        "OverallCond":  overall_cond,
        "CentralAir":   central_air,
    })

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, price)
    return model

model = get_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("House Sale Price Predictor")
st.markdown("Adjust the sliders below and get an **instant price estimate** — no file upload needed.")
st.markdown("---")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("House Features")

    st.markdown("##### Quality & Condition")
    c1, c2 = st.columns(2)
    with c1:
        overall_qual = st.slider("Overall Quality", 1, 10, 6,
            help="1 = Very Poor, 10 = Very Excellent")
        overall_cond = st.slider("Overall Condition", 1, 10, 5,
            help="1 = Very Poor, 9 = Excellent")
    with c2:
        central_air = st.radio("Central Air Conditioning", ["Yes", "No"], horizontal=True)
        central_air_val = 1 if central_air == "Yes" else 0
        fireplaces = st.slider("Number of Fireplaces", 0, 3, 0)

    st.markdown("---")
    st.markdown("##### Size (sq ft)")
    c3, c4 = st.columns(2)
    with c3:
        gr_liv_area   = st.slider("Above Ground Living Area", 400, 5000, 1500, step=50)
        total_bsmt_sf = st.slider("Total Basement Area", 0, 3000, 800, step=50)
        first_flr_sf  = st.slider("1st Floor Area", 300, 4000, 1000, step=50)
    with c4:
        lot_area      = st.slider("Lot Area", 1000, 50000, 8500, step=500)
        garage_area   = st.slider("Garage Area", 0, 1500, 400, step=50)
        open_porch_sf = st.slider("Open Porch Area", 0, 500, 0, step=10)

    st.markdown("---")
    st.markdown("##### Rooms & Year")
    c5, c6 = st.columns(2)
    with c5:
        full_bath   = st.slider("Full Bathrooms", 0, 4, 2)
        tot_rms     = st.slider("Total Rooms (above ground)", 2, 12, 7)
        garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)
    with c6:
        year_built = st.slider("Year Built", 1900, 2026, 1995)
        year_remod = st.slider("Year Remodelled", 1900, 2026, 1995)

# ── Live prediction ───────────────────────────────────────────────────────────
input_data = pd.DataFrame([{
    "OverallQual":  overall_qual,
    "GrLivArea":    gr_liv_area,
    "TotalBsmtSF":  total_bsmt_sf,
    "YearBuilt":    year_built,
    "GarageCars":   garage_cars,
    "GarageArea":   garage_area,
    "FullBath":     full_bath,
    "TotRmsAbvGrd": tot_rms,
    "Fireplaces":   fireplaces,
    "LotArea":      lot_area,
    "1stFlrSF":     first_flr_sf,
    "OpenPorchSF":  open_porch_sf,
    "YearRemodAdd": year_remod,
    "OverallCond":  overall_cond,
    "CentralAir":   central_air_val,
}])

predicted_price = model.predict(input_data)[0]
low  = predicted_price * 0.90
high = predicted_price * 1.10

with col_right:
    st.subheader("Estimated Price")
    st.markdown(f"""
    <div style='background:#1e3a5f;padding:30px;border-radius:16px;text-align:center;'>
        <div style='color:#aac4e0;font-size:16px;margin-bottom:6px;'>Predicted Sale Price</div>
        <div style='color:#ffffff;font-size:48px;font-weight:bold;'>${predicted_price:,.0f}</div>
        <div style='color:#aac4e0;font-size:14px;margin-top:10px;'>Range: ${low:,.0f} – ${high:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Price gauge
    min_p, max_p = 50000, 1500000
    pct = min(max((predicted_price - min_p) / (max_p - min_p), 0), 1.0)
    st.progress(pct)
    g1, g2, g3 = st.columns(3)
    g1.caption("$50k")
    g2.caption(f"${predicted_price/1000:.0f}k")
    g3.caption("$1.5M+")

    st.markdown("---")
    st.markdown("##### Key Drivers")

    factors = {
        "Overall Quality": overall_qual / 10,
        "Living Area":     min(gr_liv_area / 5000, 1),
        "Year Built":      (year_built - 1900) / 126,
        "Basement Size":   min(total_bsmt_sf / 3000, 1),
        "Garage":          garage_cars / 4,
    }
    for label, val in factors.items():
        st.markdown(f"**{label}**")
        st.progress(float(val))

st.markdown("---")
st.caption("Model trained on synthetic data matching Kaggle House Prices distributions | Random Forest")
