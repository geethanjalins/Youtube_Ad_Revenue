import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

YouTubeAD_df=pd.read_csv("YouTubeAD_df.csv")
YouTubeAD_df.head(2)
#views, likes, comments, watch_time_minutes, subscribers, category, country

X = YouTubeAD_df.drop(columns=['ad_revenue_usd', 'date', 'time','video_length_minutes'])
y=YouTubeAD_df["ad_revenue_usd"]

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=42)

std_scale=StandardScaler()
X_train_scaled=std_scale.fit_transform(X_train)
X_test_scaled=std_scale.transform(X_test)

X_train_scaled.shape, X_test_scaled.shape, y_train.shape, y_test.shape
# Example: Load or create your trained model
# For demonstration, we'll create a dummy model
# X_train = np.array([[1000, 50, 200], [2000, 80, 300], [1500, 60, 250]])
# y_train = np.array([100, 200, 150])
LR_model = LinearRegression()
LR_model.fit(X_train_scaled, y_train)

# Get user inputs
views = st.number_input("Enter views:", min_value=0)
likes = st.number_input("Enter likes:", min_value=0)
subscribers = st.number_input("Enter subscriptions:", min_value=0)
watch_time_minutes = st.number_input("Enter watch time minutes:", min_value=0)
comments = st.number_input("Enter comments:", min_value=0)
st.write("Select Category:")
st.write("1: Education\n 2: Lifestyle\n3: Tech\n 4: Music\n 5: Entertainment\n 6: Gaming")
category = st.selectbox('Choose a category:', ['1', '2', '3', '4', '5', '6'])   
st.write("Select Country:")
st.write("CA: 1\n DE 2 \n IN  3 \n AU  4 \n UK 5 \n US 6")
country = st.selectbox('Choose a country:', ['1', '2', '3', '4', '5', '6'])

# Prepare input for prediction
#views, likes, comments, watch_time_minutes, subscribers, category, country
input_data = np.array([[views, likes, comments, watch_time_minutes, subscribers, int(category), int(country)]], dtype=float)

# Scale input using the same scaler used for training
input_data_scaled = std_scale.transform(input_data)

# Make prediction using the trained LinearRegression model
prediction = LR_model.predict(input_data_scaled)

# Show the result
st.write(f"Estimated revenue based on inputs: ${prediction[0]:.2f}")
