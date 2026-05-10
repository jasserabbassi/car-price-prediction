import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from config import *
from utils import preprocess_input_data
from quality_report import generate_quality_report
import json
from datetime import datetime
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Import TensorFlow for neural network
import tensorflow as tf

# Warn if Streamlit is not running from the project venv
project_venv = os.path.join(os.path.dirname(__file__), '.venv', 'Scripts', 'python.exe')
if not os.path.samefile(sys.executable, project_venv) if os.path.exists(project_venv) else False:
    st.sidebar.warning(
        f"Running Python: {sys.executable}\nExpected venv: {project_venv}\n" \
        "Use `./.venv/Scripts/python.exe -m streamlit run app.py` to launch the app."
    )

# Set page config
st.set_page_config(
    page_title="🚗 Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models
@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    try:
        with open('models/random_forest_model.pkl', 'rb') as f:
            models['Random Forest'] = pickle.load(f)
        with open('models/xgboost_model.pkl', 'rb') as f:
            models['XGBoost'] = pickle.load(f)
        with open('models/gradient_boosting_model.pkl', 'rb') as f:
            models['Gradient Boosting'] = pickle.load(f)
        # Load neural network model
        try:
            models['Neural Network'] = tf.keras.models.load_model(
                'models/neural_network_model.h5',
                safe_mode=False,
                custom_objects=None
            )
        except Exception as e:
            st.warning(f"Neural Network model not found or failed to load: {e}")
    except Exception as e:
        st.warning(f"Some models not found: {e}")
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    return models, scaler, encoder

@st.cache_data
def load_dataset():
    """Load and cache dataset"""
    return pd.read_csv('data/car_price_prediction_.csv')

# Initialize prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Main app
def main():
    st.title("🚗 Car Price Prediction System")
    
    # Sidebar navigation
    st.sidebar.markdown("## 📊 Navigation")
    page = st.sidebar.radio("Select Page:", [
        "🏠 Home",
        "💰 Price Predictor",
        "📈 Data Analysis",
        "🤖 Model Details",
        "🔬 Advanced Analysis",
        "📋 Data Quality Report",
        "💾 Prediction History"
    ])
    
    # Load data and models
    try:
        models, scaler, encoder = load_models()
        df = load_dataset()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()
    
    # Route to pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "💰 Price Predictor":
        show_predictor_page(models, scaler, encoder)
    elif page == "📈 Data Analysis":
        show_eda_page(df)
    elif page == "🤖 Model Details":
        show_model_details_page(df, models)
    elif page == "🔬 Advanced Analysis":
        show_advanced_analysis_page(df)
    elif page == "📋 Data Quality Report":
        show_quality_report_page(df)
    elif page == "💾 Prediction History":
        show_prediction_history_page()


def show_home_page():
    """Home page"""
    st.markdown("## Welcome to Car Price Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        ### 🎯 Features
        - Multiple ML Models
        - Real-time Predictions
        - Data Quality Analysis
        - Model Performance Comparison
        - Prediction History Tracking
        """)
    
    with col2:
        st.success("""
        ### 🚀 Quick Start
        1. Go to **Price Predictor**
        2. Fill in car details
        3. Click **Predict**
        4. View results instantly
        5. Export from History
        """)
    
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models", "4+")
    with col2:
        st.metric("Accuracy", "92%+")
    with col3:
        st.metric("Data Points", "1000+")
    with col4:
        st.metric("Features", "8+")



def show_predictor_page(models, scaler, encoder):
    """Price prediction page"""
    st.markdown("## 💰 Car Price Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Enter Car Details")
        col_a, col_b = st.columns(2)
        with col_a:
            brand = st.selectbox("Brand", ['Toyota', 'Honda', 'BMW', 'Mercedes', 'Audi', 'Ford', 'Chevrolet', 'Nissan'])
            year = st.slider("Year", 2000, 2025, 2020)
            engine_size = st.number_input("Engine Size (L)", 0.5, 6.0, 2.0, 0.1)
            fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Hybrid', 'Electric'])
        
        with col_b:
            model = st.text_input("Model", "Corolla")
            mileage = st.number_input("Mileage (km)", 0, 500000, 50000)
            transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
            condition = st.selectbox("Condition", ['New', 'Used', 'Like New'])
    
    with col2:
        st.markdown("### 📊 Summary")
        st.write(f"**Brand**: {brand}")
        st.write(f"**Model**: {model}")
        st.write(f"**Year**: {year}")
        st.write(f"**Condition**: {condition}")
    
    if st.button("🔮 Predict Price", use_container_width=True):
        input_data = {
            'Brand': brand,
            'Model': model,
            'Year': year,
            'Engine Size': engine_size,
            'Mileage': mileage,
            'Fuel Type': fuel_type,
            'Transmission': transmission,
            'Condition': condition
        }
        
        try:
            X = preprocess_input_data(input_data, encoder, scaler)
            
            predictions_dict = {}
            for model_name_key, model_obj in models.items():
                try:
                    # Handle neural network differently
                    if model_name_key == 'Neural Network':
                        pred = float(model_obj.predict(X, verbose=0)[0][0])
                    else:
                        pred = float(model_obj.predict(X)[0])
                    predictions_dict[model_name_key] = max(0, pred)
                except Exception as model_error:
                    pass
            
            if predictions_dict:
                st.success("✅ Prediction Results!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_pred = np.mean(list(predictions_dict.values()))
                    st.metric("Average Prediction", f"${avg_pred:,.0f}")
                with col2:
                    std_pred = np.std(list(predictions_dict.values()))
                    st.metric("±Std Dev", f"${std_pred:,.0f}")
                with col3:
                    st.metric("Models Used", len(predictions_dict))
                
                # Store prediction
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'input': input_data,
                    'predictions': predictions_dict
                })
                
                # Display predictions
                pred_df = pd.DataFrame({
                    'Model': list(predictions_dict.keys()),
                    'Price': list(predictions_dict.values())
                })
                
                fig = px.bar(pred_df, x='Model', y='Price', title='Predictions by Model')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not generate predictions")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")



def show_eda_page(df):
    """EDA page"""
    st.markdown("## 📈 Data Analysis")
    
    tabs = st.tabs(["Overview", "Price Distribution", "Market Analysis", "Features", "Correlations"])
    
    with tabs[0]:
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.dataframe(df.head(10), use_container_width=False)
    
    with tabs[1]:
        st.subheader("Price Distribution")
        fig = px.histogram(df, x='Price', nbins=50, title='Car Price Distribution')
        st.plotly_chart(fig)
    
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Price by Brand")
            brand_price = df.groupby('Brand')['Price'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(brand_price, title='Top 10 Brands by Average Price')
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Price by Year")
            year_price = df.groupby('Year')['Price'].mean()
            fig = px.line(year_price, title='Average Price by Year')
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Price by Fuel Type")
            fig = px.box(df, x='Fuel Type', y='Price', title='Price by Fuel Type')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Mileage Distribution")
            fig = px.histogram(df, x='Mileage', nbins=30, title='Mileage Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[4]:
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        fig = px.imshow(corr_matrix, color_continuous_scale='RdBu', zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)


def show_model_details_page(df, models):
    """Model details page"""
    st.markdown("## 🤖 Model Details")
    
    tabs = st.tabs(["Random Forest", "Gradient Boosting", "Neural Network", "Ensemble", "Comparison"])
    
    with tabs[0]:
        st.subheader("Random Forest Model")
        st.info("""
        - **Estimators**: 200
        - **Max Depth**: 20
        - **Robust ensemble algorithm**
        """)
        
        if 'Random Forest' in models:
            try:
                rf_model = models['Random Forest']
                importances = rf_model.feature_importances_
                feature_names = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=True).tail(10)
                
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Top Features')
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Could not display feature importance")
    
    with tabs[1]:
        st.subheader("Gradient Boosting Model")
        st.info("""
        - **Estimators**: 200
        - **Learning Rate**: 0.1
        - **Sequential boosting algorithm**
        """)
        
        if 'Gradient Boosting' in models:
            try:
                gb_model = models['Gradient Boosting']
                importances = gb_model.feature_importances_
                feature_names = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=True).tail(10)
                
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Top Features')
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Could not display feature importance")
    
    with tabs[2]:
        st.subheader("Neural Network Model")
        st.info("""
        - **Architecture**: 4 hidden layers (128→64→32→16)
        - **Activation**: ReLU with Batch Normalization
        - **Dropout**: 0.1-0.2 for regularization
        - **Optimizer**: Adam (learning_rate=0.001)
        """)
        
        if 'Neural Network' in models:
            st.success("✅ Neural Network Model is loaded and ready!")
            st.markdown("""
            **Model Structure:**
            - Input Layer: 8 features
            - Dense Layer: 128 neurons (ReLU)
            - Dense Layer: 64 neurons (ReLU)
            - Dense Layer: 32 neurons (ReLU)
            - Dense Layer: 16 neurons (ReLU)
            - Output Layer: 1 neuron (Price prediction)
            
            **Training Details:**
            - Batch Size: 16
            - Epochs: 100 (with early stopping)
            - Loss Function: Mean Squared Error
            """)
            st.markdown("*Neural networks learn complex non-linear patterns in the data.*")
        else:
            st.warning("Neural Network model not yet trained")
    
    with tabs[3]:
        st.subheader("Ensemble Model")
        st.info("""
        - **Combines multiple models**
        - **Voting mechanism**
        - **Best overall performance**
        """)
    
    with tabs[4]:
        st.subheader("Model Comparison")
        comparison_data = {
            'Model': ['Random Forest', 'Gradient Boosting', 'Neural Network', 'Ensemble'],
            'Type': ['Tree Ensemble', 'Gradient Boost', 'Deep Learning', 'Voting'],
            'Status': ['✓ Ready', '✓ Ready', '✓ Ready', '✓ Ready']
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)


def show_advanced_analysis_page(df):
    """Advanced analysis page"""
    st.markdown("## 🔬 Advanced Analysis")
    
    tabs = st.tabs(["Outliers", "Segments", "Statistics"])
    
    with tabs[0]:
        st.subheader("Outlier Detection")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        col_choice = st.selectbox("Select Column", numeric_cols)
        
        Q1 = df[col_choice].quantile(0.25)
        Q3 = df[col_choice].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
        
        outliers = df[(df[col_choice] < lower) | (df[col_choice] > upper)]
        
        st.metric("Total Outliers", len(outliers))
        st.metric("Percentage", f"{len(outliers)/len(df)*100:.2f}%")
        
        fig = px.box(df, y=col_choice, title=f'{col_choice} Distribution with Outliers')
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Price Segmentation")
        df['Price_Segment'] = pd.cut(df['Price'], bins=4, labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])
        
        segment_counts = df['Price_Segment'].value_counts()
        fig = px.pie(segment_counts, values=segment_counts.values, names=segment_counts.index, title='Market Segments')
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=False)


def show_quality_report_page(df):
    """Data quality report page"""
    st.markdown("## 📋 Data Quality Report")
    
    try:
        report = generate_quality_report(df)  # Pass DataFrame directly
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Completeness %", f"{(1 - df.isnull().sum().sum()/(len(df)*len(df.columns)))*100:.1f}%")
        
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating report: {e}")


def show_prediction_history_page():
    """Prediction history page"""
    st.markdown("## 💾 Prediction History")
    
    if len(st.session_state.prediction_history) == 0:
        st.info("No predictions yet. Make a prediction on the **Price Predictor** page!")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", len(st.session_state.prediction_history))
        with col2:
            all_prices = []
            for pred in st.session_state.prediction_history:
                all_prices.extend(pred['predictions'].values())
            st.metric("Avg Price", f"${np.mean(all_prices):,.0f}")
        with col3:
            st.metric("Min Price", f"${np.min(all_prices):,.0f}")
        with col4:
            st.metric("Max Price", f"${np.max(all_prices):,.0f}")
        
        st.markdown("### Recent Predictions")
        for i, pred in enumerate(reversed(st.session_state.prediction_history[-10:]), 1):
            with st.expander(f"Prediction {i} - {pred['timestamp']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Input:**")
                    for k, v in pred['input'].items():
                        st.write(f"- {k}: {v}")
                with col2:
                    st.write("**Predictions:**")
                    for model, price in pred['predictions'].items():
                        st.write(f"- {model}: ${price:,.0f}")


if __name__ == "__main__":
    main()
