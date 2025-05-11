import streamlit as st
import joblib
import pandas as pd

# تحميل الموديل
model = joblib.load("random_forest_model.joblib")

# تعريف الأعمدة
feature_names = [
    'Age', 'Gender', 'Income_Level', 'Marital_Status', 
    'Education_Level', 'Occupation', 'Location', 'Purchase_Category', 
    'Purchase_Amount', 'Frequency_of_Purchase', 'Purchase_Channel',
    'Brand_Loyalty', 'Product_Rating', 'Time_Spent_on_Product_Research(hours)', 
    'Social_Media_Influence', 'Discount_Sensitivity', 'Return_Rate', 
    'Customer_Satisfaction', 'Engagement_with_Ads', 'Device_Used_for_Shopping',
    'Payment_Method', 'Time_of_Purchase', 'Discount_Used', 
    'Customer_Loyalty_Program_Member', 'Shipping_Preference', 'Time_to_Decision'
]

st.title("Customer Behavior Prediction App")

user_input = {}
for feature in feature_names:
    user_input[feature] = st.text_input(f"{feature}", key=feature)

# لما المستخدم يضغط Predict
if st.button("Predict"):
    try:
        # تحويل الإدخالات لأرقام أو تهيئتها كما يجب (لو كانت كلها أرقام)
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)
        st.success(f"Prediction: {int(prediction[0])}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
