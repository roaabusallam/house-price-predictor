import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="House Price Predictor"
)


model = joblib.load(
    "house_price_model.pkl"
)


st.title("House Price Predictor")

st.write(
    "Enter the information below to predict the house price."
)


average_income = st.number_input(
    "Average Area Income",
    min_value=0.0, #يمنع ادخال رقم سالب
    value=65000.0 # القيمة الافتراضية
)
average_house_age = st.number_input(
    "Average Area House Age",
    min_value=0,
    value=6,
    step=1
)

average_rooms = st.number_input(
    "Average Area Number of Rooms",
    min_value=0,
    value=6,
    step=1
)

average_bedrooms = st.number_input(
    "Average Area Number of Bedrooms",
    min_value=0,
    value=3,
    step=1
)

area_population = st.number_input(
    "Area Population",
    min_value=0,
    value=35000,
    step=1
)


if st.button("Predict Price"):

    input_data = pd.DataFrame([{
        "Avg. Area Income": average_income,
        "Avg. Area House Age": average_house_age,
        "Avg. Area Number of Rooms": average_rooms,
        "Avg. Area Number of Bedrooms": average_bedrooms,
        "Area Population": area_population
    }])

    predicted_price = model.predict(
        input_data
    )[0] #أول توقع

    st.success(
        f"Predicted House Price: ${predicted_price:,.0f}"
    )