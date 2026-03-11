import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Car Price Prediction - Tunisia",
    page_icon="🚗",
    layout="wide"
)

# ================== MODERN DESIGN ==================
st.markdown("""
    <style>
    .main {padding: 1.5rem 3rem; background-color: #f8f9fa;}
    h1 {color: #e30613; font-size: 3.2rem; text-align: center; margin-bottom: 0.5rem;}
    h2, h3 {color: #1e3a8a;}
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
    .footer {
        text-align: center;
        padding: 2.5rem 0;
        color: #64748b;
        font-size: 1rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 4rem;
    }
    </style>
""", unsafe_allow_html=True)

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    try:
        return joblib.load('car_prediction_model.pkl')
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error("❌ Model file not found!")
    st.info("Please run first: `python car_price_prediction.py` to train and save the model.")
    st.stop()

# ================== HEADER ==================
st.title("🚗 Car Price Prediction System - Tunisia")
st.markdown("### Get Instant & Accurate Valuation for Your Used Car")

st.caption(f"📅 Today: {datetime.now().strftime('%B %d, %Y')} • Tunisian Market")

# ================== SIDEBAR ==================
st.sidebar.title("📋 Car Details")

st.sidebar.subheader("Basic Information")
year = st.sidebar.slider("Manufacturing Year", 2005, 2026, 2018)
present_price = st.sidebar.number_input("Current Ex-Showroom Price (TND)", 
                                       min_value=8000, max_value=350000, 
                                       value=45000, step=1000)
kms_driven = st.sidebar.number_input("Kilometers Driven (Km)", 
                                    min_value=0, max_value=500000, 
                                    value=75000, step=5000)

st.sidebar.subheader("Car Specifications")
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.sidebar.selectbox("Number of Previous Owners", [0, 1, 2, 3, 4])

# Car age
car_age = 2026 - year

# Predict button
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮 Get Price Estimate", type="primary", use_container_width=True)

# ================== PREDICTION ==================
if predict_btn:
    # Encode categorical variables (same as your trained model)
    fuel_encoded = {"Petrol": 0, "Diesel": 1, "CNG": 2}[fuel_type]
    seller_encoded = {"Dealer": 0, "Individual": 1}[seller_type]
    transmission_encoded = {"Manual": 0, "Automatic": 1}[transmission]

    # Prepare input (scale price back to the model's Lakhs scale)
    input_data = pd.DataFrame({
        'Year': [year],
        'Present_Price': [present_price / 10000],   # TND → model scale
        'Kms_Driven': [kms_driven],
        'Fuel_Type': [fuel_encoded],
        'Seller_Type': [seller_encoded],
        'Transmission': [transmission_encoded],
        'Owner': [owner]
    })

    predicted_price = model.predict(input_data)[0] * 10000   # Back to real TND

    # Depreciation
    depreciation = present_price - predicted_price
    depreciation_percent = (depreciation / present_price) * 100 if present_price > 0 else 0

    # ================== RESULTS ==================
    st.markdown("---")
    st.header("📊 Price Estimation Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estimated Selling Price", f"{predicted_price:,.0f} TND")
    with col2:
        st.metric("Current Showroom Price", f"{present_price:,.0f} TND")
    with col3:
        st.metric("Total Depreciation", f"{depreciation:,.0f} TND", f"-{depreciation_percent:.1f}%")

    # Analysis + Gauge
    col1, col2 = st.columns([2, 1])
    with col1:
        st.success(f"**Realistic Market Range:** {int(predicted_price*0.9):,} - {int(predicted_price*1.1):,} TND")
        
        st.write("**Price Influencing Factors:**")
        for factor in [
            "Very recent car" if car_age <= 5 else "Moderate age",
            "Low mileage" if kms_driven < 80000 else "High mileage",
            "Automatic transmission - premium value" if transmission == "Automatic" else "",
            "Diesel engine - preferred for long trips" if fuel_type == "Diesel" else ""
        ]:
            if factor:
                st.markdown(f"- {factor}")

    with col2:
        # Beautiful gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_price,
            number={'suffix': " TND"},
            title={'text': "Estimated Price"},
            gauge={
                'axis': {'range': [0, present_price * 1.3]},
                'bar': {'color': "#e30613"},
                'steps': [
                    {'range': [0, present_price * 0.4], 'color': "#e2e8f0"},
                    {'range': [present_price * 0.4, present_price * 0.8], 'color': "#fed7aa"},
                    {'range': [present_price * 0.8, present_price * 1.3], 'color': "#86efac"}
                ]
            }
        ))
        fig.update_layout(height=320, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Car summary
    st.subheader("Your Car Details")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Year:** {year} ({car_age} years old)")
        st.write(f"**Kilometers Driven:** {kms_driven:,} km")
    with col2:
        st.write(f"**Fuel Type:** {fuel_type}")
        st.write(f"**Transmission:** {transmission}")
        st.write(f"**Seller Type:** {seller_type}")

# ================== INITIAL SCREEN ==================
else:
    st.info("👈 Fill in the car details in the sidebar and click **Get Price Estimate**")
    st.markdown("---")
    st.subheader("Example Valuations (Tunisia Market)")

# ================== FOOTER WITH YOUR NAME ==================
st.markdown("""
    <div class="footer">
        <strong>Developed by Jasser Abbassi</strong><br>
        Final Year Project (PFE) • Data science • 2026
    </div>
""", unsafe_allow_html=True)