import streamlit as st
import pandas as pd


def clear():
    selected_column = st.session_state.get("reset_dropdown")
    

    defaults = {
        "col1_tenure": 12,
        "col1_contractType": "Select the Contract Type",
        "col1_gender": None,
        "col1_partner": None,
        "col1_dependents": None,

        "col2_phoneService": None,
        "col2_multipleLines": None,
        "col2_internet_service": "Select the Internet Service Type",
        "col2_onlineSecurity": None,
        "col2_onlineBackup": None,
        "col2_deviceProtection": None,
        "col2_techSupport": None,
        "col2_streamingTV": None,
        "col2_streamingMovies": None,

        "col3_paperlessBilling": None,
        "col3_paymentMethod": "Select Payment Method",
        "col3_monthlyCharges": 0.0,
        "col3_totalCharges": 0.0,
    }


    if selected_column == "All columns":
        for key, val in defaults.items():
            st.session_state[key] = val
            
    elif selected_column == "Account Basics":
        for key, val in defaults.items():
            if key.startswith("col1_"):
                st.session_state[key] = val
                
    elif selected_column == "Services Subscribed":
        for key, val in defaults.items():
            if key.startswith("col2_"):
                st.session_state[key] = val
                
    elif selected_column == "Financial Metrics":
        for key, val in defaults.items():
            if key.startswith("col3_"):
                st.session_state[key] = val
                
    st.rerun()  





st.set_page_config(layout="wide")
st.title("Customer Churn Dashboard")

st.sidebar.markdown("<div style='padding-top: 40px;'>Clear Options</div>", unsafe_allow_html=True)
clear_dd_width,clear_btn_width=st.sidebar.columns([4,3 ])
with clear_dd_width:
    selected_column=st.selectbox("Clear Options",options=["All columns","Account Basics","Services Subscribed","Financial Metrics"],key="reset_dropdown",label_visibility="collapsed")
with clear_btn_width:
    clear_btn=st.button("Clear Inputs",on_click=clear)





col1, col2,col3=st.columns(3)

with col1:
    st.markdown("## **Account Basics** ##")
    component_width,empty_space=st.columns([2,1])

    with component_width:
        tenure=st.number_input(label="Tenure",min_value=0,max_value=72,value=12,key="col1_tenure")
        contract=st.selectbox("Contract Type",options=["Select the Contract Type","Month-to-month","One year","Two year "],key="col1_contractType")
        gender=st.radio("Gender",options=["Male","Female"],index=None,horizontal=True,key="col1_gender")
        partner=st.radio("Partner",options=["Yes","No"],index=None,horizontal=True,key="col1_partner")
        dependents=st.radio("Dependents",options=["Yes","No"],index=None,horizontal=True,key="col1_dependents")
        senior_citizen=st.radio("Senior Citizen",options=["Yes","No"],index=None,horizontal=True,key="col1_seniorCitizen")


with col2:
    st.markdown("## **Services Subscribed** ##")
    component_width,empty_space=st.columns([2,1])

    with component_width:
        phone_service=st.radio("Phone Service",options=["Yes","No"],index=None,horizontal=True,key="col2_phoneService")
        if (phone_service=="Yes"):
            multiple_lines=st.radio("Multiple Lines",options=["Yes","No"],index=None,horizontal=True,key="col2_multipleLines")
        else:
            multiple_lines="No phone service"

        internet_service = st.selectbox("Internet Service Type", options=["Select the Internet Service Type","Fiber optic", "DSL", "No"],key="col2_internet_service")
        if (internet_service!="No" and not(internet_service.startswith("Select"))):
            online_security = st.radio("Online Security", options=["Yes","No"],index=None,horizontal=True,key="col2_onlineSecurity")
            online_backup=st.radio("Online Backup", options=["Yes","No"],index=None,horizontal=True,key="col2_onlineBackup")
            device_protection=st.radio("Device Protection", options=["Yes","No"],index=None,horizontal=True,key="col2_deviceProtection")
            tech_support= st.radio("Tech Support", options=["Yes","No"],index=None,horizontal=True,key="col2_techSupport")
            streaming_tv=st.radio("Streaming TV", options=["Yes","No"],index=None,horizontal=True,key="col2_streamingTV")
            streaming_movies=st.radio("Streaming Movies", options=["Yes","No"],index=None,horizontal=True,key="col2_streamingMovies")
        else:
            online_security=online_backup=device_protection=tech_support=streaming_tv=streaming_movies="No internet service"


with col3:
    st.markdown("## **Financial Metrics** ##")
    component_width,empty_space=st.columns([2,1])

    with component_width:
        paperless_billing=st.radio("Paperless Billing", options=["Yes","No"],index=None,horizontal=True,key="col3_paperlessBilling")
        payment_method = st.selectbox("Payment Method", options=["Select Payment Method","Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],key="col3_paymentMethod")
        monthly_charges=st.number_input(label="Enter Monthly Charges",min_value=0.0,max_value=150.0,value=0.0,step=1.0,key="col3_monthlyCharges")
        total_charges=st.number_input(label="Enter Total Charges",min_value=0.0,max_value=10000.0,value=0.0,step=1.0,key="col3_totalCharges")



raw_user_data={
    'gender':[gender],
    'SeniorCitizen':[senior_citizen],
        'Partner':[partner],
        'Dependents':[dependents], 
        'tenure':[tenure],
        'PhoneService':[phone_service], 
        'PaperlessBilling':[paperless_billing], 
        'MonthlyCharges':[monthly_charges], 
        'TotalCharges':[total_charges],
        'MultipleLines':[multiple_lines], 
        'InternetService':[internet_service],
        'OnlineSecurity':[online_security], 
        'OnlineBackup':[online_backup],
        'DeviceProtection':[device_protection], 
        'TechSupport':[tech_support],
    'StreamingTV':[streaming_tv],
    'StreamingMovies':[streaming_movies], 
    'Contract':[contract], 
    'PaymentMethod':[payment_method]
}
input_df=pd.DataFrame(raw_user_data)


preprocess_fn=st.session_state.get('processed_df')

if preprocess_fn is not None:
    try:
        model_ready_df=preprocess_fn(input_df)
    except Exception as e:
        st.error(f"Error during Preprocessing Execution: {e}")




active_model=st.session_state['model']

if active_model is not None:
    if st.button("Calculate Churn Risk",type="primary"):
        churn_percent=active_model.predict_proba(model_ready_df)[0][1]*100

        st.subheader("Churn Prediction and Discounts")
        
        col_churn,col_discount=st.columns(2)
        with col_churn:
            st.markdown("## **Churning Probability** ##")
            st.metric("Calculated Churn Probability",value=f"{churn_percent:.2f}%")
            if churn_percent>50:
                st.error("This customer is more likely to churn")
            else:
                st.success("This customer will most probably stay")

        with col_discount:
            st.markdown("## **Discounts** ##")
            offer_count=st.session_state.get("offer_count",1)
            offer_found=False
            for i in range(offer_count):
                if(churn_percent>st.session_state.get(f"prob_offer {i+1}")):
                    offer_found=True
                    st.success(f"Customer gets a offer of {st.session_state.get(f"percent_offer {st.session_state.get(f"prob_offer {i+1}")}")}% under offer scheme {i+1}")
                    st.metric("Discounted Monthly Charge",value=f"${monthly_charges*(100-st.session_state.get(f"percent_offer {st.session_state.get(f"prob_offer {i+1}")}"))/100}")
                    break
            if (not offer_found):
                st.error("Customer is not eligible for any offers")
                st.metric("Monthly Charge remains at",value=monthly_charges)

        
else:
    st.warning("Please choose a specific optimization framework strategy in the sidebar to see prediction results")