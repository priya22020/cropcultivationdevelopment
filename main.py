import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set Streamlit page config
st.set_page_config(page_title="Crop Recommendation System", layout="centered")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("crop_recommendation.csv")

# Train model
@st.cache_resource
def train_model(data):
    X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# Load data and model
data = load_data()
model, accuracy = train_model(data)

# App Title
st.title("🌱 Crop Recommendation System")
st.markdown("Predict the most suitable crop to cultivate based on soil and weather conditions.")

# Input fields
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=90)
        P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=42)
        K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=43)
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)

    with col2:
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
        humidity = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=80.0)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=200.0)

    submitted = st.form_submit_button("🌾 Recommend Crop")

# Prediction
if submitted:
    input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    prediction = model.predict(input_df)[0]
    st.success(f"✅ **Recommended Crop:** {prediction.capitalize()}")

# Sidebar
st.sidebar.header("ℹ️ Model Details")
st.sidebar.write(f"**Model:** Random Forest Classifier")
st.sidebar.write(f"**Accuracy:** {accuracy * 100:.2f}%")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ using Streamlit")
