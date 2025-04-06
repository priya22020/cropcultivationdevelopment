import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('crop_recommendation.csv')
    return data

# Train model
@st.cache_resource
def train_model(data):
    X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# Load and train
data = load_data()
model, accuracy = train_model(data)

# App UI
st.title("🌾 Crop Recommendation System")
st.write("Enter the environmental and soil values below to get a crop recommendation.")

N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=90)
P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=42)
K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=43)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=80.0)
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=200.0)

if st.button("Recommend Crop"):
    input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Recommended Crop: **{prediction}**")

st.sidebar.markdown("### ℹ️ Model Info")
st.sidebar.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")
