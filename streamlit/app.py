import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_models():
    try:
        accuracy_model=joblib.load("ml_models/accuracy_model.pkl")
        aggressive_model=joblib.load("ml_models/aggressive_model.pkl")
        defensive_model=joblib.load("ml_models/defensive_model.pkl")
        
        scaler=joblib.load(r"ml_models\scaler.pkl")

        model_columns=joblib.load(r"ml_models\columns.pkl")

        return {"Accuracy": accuracy_model,"Aggressive":aggressive_model,"Defensive":defensive_model,"Scaler":scaler,"Columns":model_columns}
    except Exception as e:
        st.sidebar.error(f"Error Loading Models or Scaler :{e}")
        return None
    
models_dict=load_models()

if models_dict is not None:
    st.session_state['models']=models_dict
    st.session_state['scaler']=models_dict["Scaler"]
    st.session_state['model_columns']=models_dict["Columns"]






def preprocessing_data(raw_df):
    processed_df=raw_df.copy()

    model_columns=st.session_state.get("model_columns")
    

    binary_cols=["Partner","Dependents","PhoneService","PaperlessBilling","SeniorCitizen"]
    for col in binary_cols:
        if col in processed_df.columns:
            processed_df[col]=processed_df[col].map({"Yes":1,"No":0}).fillna(0)
        
    if 'gender' in processed_df.columns:
        processed_df["gender"]=processed_df["gender"].map({"Male":1,"Female":0}).fillna(0)

    if 'customerID' in processed_df.columns:
        processed_df=processed_df.set_index('customerID')
        
    processed_df=pd.get_dummies(processed_df,dtype=int)

    for col in model_columns:
        if col not in processed_df.columns:
            processed_df[col]=0

    processed_df=processed_df[model_columns]

    scaler=st.session_state.get('scaler')
    if scaler is not None:
        try:
            scaled_features = scaler.transform(processed_df)
            processed_df = pd.DataFrame(scaled_features, columns=model_columns)
        except Exception as scale_err:
            st.sidebar.error(f"Scaling Transformation Failed: {scale_err}")

    return processed_df



st.session_state['processed_df']=preprocessing_data

    


single_user_page=st.Page("single_user_page.py",title="Single User Interface")
multi_user_page=st.Page("multi_user_page.py",title="Multi User Interface",default=True)

pg=st.navigation([multi_user_page,single_user_page],position="top")

strategy=st.sidebar.selectbox("Optimization Framework",options=["Select Strategy to provide offers and stop churning","Accuracy - Uses the most accurate model","Aggressive - Uses the model that catches most Churners","Defensive - Uses the model that reduces false positives"],key="Strategy Selection")

if models_dict is not None:
    if strategy.startswith("Accuracy"):
        st.info("Running Random Forest Regressor model which is best for balanced approach with an F1 score of 0.64")
        st.session_state['model']=models_dict['Accuracy']
        st.session_state['precision']=0.52
        st.session_state['recall']=0.80
    elif strategy.startswith("Aggressive"):
        st.info("Running Logistic Regressor model with balanced class weight and the maximum iteration limit set to 10000 which is tested to catch more than 80% of Churners ")
        st.session_state['model']=models_dict['Aggressive']
        st.session_state['precision']=0.51     
        st.session_state['recall']=0.78      
    elif strategy.startswith("Defensive"):
        st.info("Running Random Forest Regressor which reduces false positives with a precision of around 70%")
        st.session_state['model']=models_dict['Defensive']
        st.session_state['precision']=0.67    
        st.session_state['recall']=0.47      
    else:
        st.session_state['model']=None
        st.session_state['precision']=None
        st.session_state['recall']=None


if "offer_count" not in st.session_state:
    st.session_state.offer_count=1
    st.sidebar.markdown("<div style='padding-top: 40px;'>Offers</div>", unsafe_allow_html=True)

seen_thresholds = set()

for i in range(st.session_state.offer_count):
    try:
        chosen_prob = st.sidebar.number_input(label=f"Select min probability of churning for offer {i+1}",min_value=0,max_value=100,step=10,value=10+i*10,key=f"prob_offer {i+1}")
        
        if chosen_prob in seen_thresholds:
            raise ValueError(f"Offer {i+1} selection matches a previously configured threshold layout.")
        seen_thresholds.add(chosen_prob)
        
        st.sidebar.number_input(label=f"Select Offer Percentage for offer {i+1}",min_value=0,max_value=100,step=1,value=10,key=f"percent_offer {st.session_state.get(f'prob_offer {i+1}')}")
    except ValueError as err:
        st.sidebar.error("Multiple offers cannot target the exact same Churn Threshold percentage. Please check your configurations.")
        st.stop()

def add_more_offers():
    st.session_state.offer_count+=1
def remove_offers():
    st.session_state.offer_count-=1

empty,button=st.sidebar.columns([1,2])
with button:
    if st.session_state.offer_count>1:
        st.button("Remove Offer",on_click=remove_offers)
    st.button("Add Offer",on_click=add_more_offers)



pg.run()