import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.number_input('Age', min_value=0, max_value=100)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=0.0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.number_input('Tenure', min_value=0, max_value=10)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4)
has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])
is_self_employed = st.selectbox('Is Self Employed', ['Yes', 'No'])

# Prepare the input data
try:
    # Encode geography (only transform one-hot encoding for the selected geography)
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    
    # Get feature names from onehot_encoder_geo directly
    geo_encoded_columns = onehot_encoder_geo.get_feature_names_out(['Geography'])
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=geo_encoded_columns
    )

    # Check the number of columns created by one-hot encoding
    st.write(f"Geo-encoded columns: {geo_encoded_df.shape[1]}")  # Should match number of geo categories

    # Encode gender
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Prepare numeric and binary inputs
    input_data = {
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Balance': [balance],
        'Tenure': [tenure],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
        'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
        'IsSelfEmployed': [1 if is_self_employed == 'Yes' else 0],
        'EstimatedSalary': [estimated_salary]
    }

    input_df = pd.DataFrame(input_data)

    # Combine numerical and one-hot encoded categorical data
    input_data_combined = pd.concat([input_df, geo_encoded_df], axis=1)

    # Check the number of columns after concatenation
    st.write(f"Columns after concatenation: {input_data_combined.shape[1]}")  # Should be 12

    # If there's an extra column in geo_encoded_df, drop it (dummy variable issue)
    if input_data_combined.shape[1] > 12:
        st.warning("Dropping extra column from one-hot encoding.")
        input_data_combined = input_data_combined.iloc[:, :-1]  # Remove the last column

    # Ensure the correct number of columns (12 features) for the model
    if input_data_combined.shape[1] != 12:
        st.error(f"Input data has {input_data_combined.shape[1]} features, but model expects 12.")
    else:
        # Check the scaler type
        st.write(f"Scaler type: {type(scaler)}")

        if isinstance(scaler, StandardScaler):
            # Scale the input data using the scaler
            input_data_scaled = scaler.transform(input_data_combined)
        else:
            # If the scaler is not a StandardScaler, scale the input manually
            st.warning("Scaler is not a StandardScaler, performing manual scaling.")
            
            # Manually scale using the mean and standard deviation from the scaler
            mean = np.mean(input_data_combined, axis=0)
            std = np.std(input_data_combined, axis=0)
            input_data_scaled = (input_data_combined - mean) / std

        # Make the prediction
        prediction = model.predict(input_data_scaled)
        prediction_prob = prediction[0][0]

        # Display results
        st.write(f'Prediction Probability: {prediction_prob:.2f}')
        if prediction_prob > 0.5:
            st.write('The customer is likely to churn.')
        else:
            st.write('The customer is likely to stay.')

except Exception as e:
    st.error(f"An error occurred: {e}")