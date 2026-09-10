# House Price Predictor

A machine learning regression application that predicts house prices.

## Features Used

- Average area income
- Average house age
- Average number of rooms
- Average number of bedrooms
- Area population

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit

## Machine Learning Workflow

1. Load housing data
2. Select features and target price
3. Split data into training and testing sets
4. Fill missing values
5. Scale numeric features
6. Train a Linear Regression model
7. Evaluate using MAE, RMSE, and R2
8. Build a Streamlit interface

## Run the Project

pip install -r requirements.txt

python train_model.py

streamlit run app.py