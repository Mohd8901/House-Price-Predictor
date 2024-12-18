import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("USA_Housing.csv")
df.drop(columns='Address', inplace=True)

# Feature and target selection
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Area Population']]
y = df['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# Load the pre-trained model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and model evaluation
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred) * 100
error = mean_squared_error(y_test, y_pred) ** 0.5

# Streamlit App
st.title("House Price Prediction App")
st.write("This app predicts house prices based on various features and displays the model's performance.")

# Input features
st.sidebar.header("Input Features")
avg_income = st.sidebar.number_input("Average Area Income", value=65000.0)
house_age = st.sidebar.number_input("Average Area House Age", value=5.0)
num_rooms = st.sidebar.number_input("Average Area Number of Rooms", value=7.0)
population = st.sidebar.number_input("Area Population", value=30000.0)

# Predict button
if st.sidebar.button("Predict Price"):
    # Create input array
    input_features = np.array([[avg_income, house_age, num_rooms, population]])
    predicted_price = model.predict(input_features)[0]
    st.subheader(f"Predicted House Price: ${predicted_price:,.2f}")
    st.write("""
    The predicted price is based on the input features you provided. These values are used to calculate the price using a trained linear regression model.
    """)

# Model Performance
st.header("Model Performance")
st.subheader("Accuracy:")
st.write(f"{accuracy:.2f}%")
st.write("""
Accuracy (R² Score) represents how well the model explains the variation in the target variable. Higher accuracy indicates better model performance.
""")

st.subheader("Error:")
st.write(f"Root Mean Squared Error: ${error:,.2f}")
st.write("""
The error represents the average difference between the predicted and actual house prices. Lower values indicate better model performance.
""")
