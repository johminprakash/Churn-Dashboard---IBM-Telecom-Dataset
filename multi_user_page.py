import streamlit as st
import pandas as pd
import math

st.set_page_config(layout="wide")
st.title("Customer Churn Dashboard")

sidebar_threshold = st.sidebar.number_input(label="Set Churn Threshold", min_value=0, max_value=100, step=1, value=50)

active_model = st.session_state.get('model')
preprocess_fn = st.session_state.get('processed_df')
strategy_selection = st.session_state.get('Strategy Selection')
precision = st.session_state.get('precision', 1.0)
recall = st.session_state.get('recall', 1.0)

uploaded_file = st.file_uploader("Choose an Excel File containing the dataset of customers", type=['xlsx', 'xls'])

if uploaded_file is not None:
    if "predicted_df" not in st.session_state or st.session_state.get("cached_strategy") != strategy_selection:
        try:
            df = pd.read_excel(uploaded_file)
            if preprocess_fn is not None:
                if active_model is not None:
                    with st.spinner("Processing data and generating predictions..."):
                        model_ready_df = preprocess_fn(df)
                        df['Churn_probability'] = active_model.predict_proba(model_ready_df)[:, 1] * 100
                        
                        st.session_state["predicted_df"] = df
                        st.session_state["cached_strategy"] = strategy_selection
                        st.success("Predictions generated successfully!")
                else:
                    st.warning("Please choose a specific Optimization Framework Strategy in the sidebar.")
            else:
                st.error("Preprocessing engine function is missing from application memory.")
        except Exception as e:
            st.error(f"Error processing pipeline: {e}")
else:
    if "predicted_df" in st.session_state:
        del st.session_state["predicted_df"]
    if "cached_strategy" in st.session_state:
        del st.session_state["cached_strategy"]
    st.info("Upload the customer dataset to predict Churning")

if "predicted_df" in st.session_state and active_model is not None:
    df = st.session_state["predicted_df"]
    
    offer_map = {}
    offer_count = st.session_state.get("offer_count", 1)
    
    for i in range(offer_count):
        prob_trigger = st.session_state.get(f"prob_offer {i+1}")
        if prob_trigger is not None:
            pct_value = st.session_state.get(f"percent_offer {prob_trigger}")
            if pct_value is not None:
                offer_map[prob_trigger] = pct_value

    base_boundaries = list(range(0, 101, 10))
    custom_boundaries = list(offer_map.keys())
    
    boundary_set = set(base_boundaries + custom_boundaries)
    boundary_set.add(sidebar_threshold)
    all_boundaries = sorted(list(boundary_set))

    table = []
    for i in range(len(all_boundaries) - 1):
        lower_bound = all_boundaries[i]
        upper_bound = all_boundaries[i + 1] - 1
        
        if lower_bound == 100:
            continue
        if upper_bound > 99:
            upper_bound = 100
            
        count = df['Churn_probability'].between(lower_bound, upper_bound).sum()
        total_monthly_premium = df.loc[df['Churn_probability'].between(lower_bound, upper_bound), 'MonthlyCharges'].sum()
        
        applicable_offer = 0
        for thresh in sorted(offer_map.keys(), reverse=True):
            if lower_bound >= thresh:
                applicable_offer = offer_map[thresh]
                break
        
        expected_true_caught = count * precision
        total_offer_deduction = total_monthly_premium * applicable_offer / 100
        premium_after_offer = total_monthly_premium - total_offer_deduction
        wasted_money = ((count - count * precision) / count * total_offer_deduction) if count > 0 else 0
        
        acceptance_prob = 1 / (1 + math.exp(-0.25 * (applicable_offer - 20))) if lower_bound >= sidebar_threshold else 1.0
        
        if lower_bound >= sidebar_threshold:
            additional_revenue = acceptance_prob * premium_after_offer
            expected_profit = additional_revenue - wasted_money
        else:
            additional_revenue = 0
            expected_profit = -total_offer_deduction
            
        missed_ppl = ((count / recall) - count) if count > 0 else 0
        
        if count > 0 and lower_bound >= sidebar_threshold:
            missed_rev = missed_ppl / count * total_monthly_premium
        else:
            missed_rev = 0

        table.append({
            "Churn Probability": f"{lower_bound}-{upper_bound}%", 
            "Count": count, 
            "Expected True People in the range caught": expected_true_caught,
            "Offer": f"{applicable_offer}%",
            "Total Monthly Premium": total_monthly_premium,
            "Total Offer Deduction on Monthly Premium": total_offer_deduction,
            "Total Monthly Premium after Offer": premium_after_offer,
            "Expected Offer Money given to people actually not in range": wasted_money,
            "Probability of Acceptance": f"{acceptance_prob * 100:.2f}%",
            "Additional Revenue gained in next month by these offers(Using S curve model)": additional_revenue,
            "Expected Total Profit": expected_profit,
            "Expected People in the range Missed": missed_ppl,
            "Expected loss in revenue due to missed people": missed_rev
        })

    table.reverse()
    table_df = pd.DataFrame(table)

    summary_row = {'Churn Probability': "0-100 %"}
    sum_cols = [
        'Count', 'Expected True People in the range caught', 'Total Monthly Premium', 
        'Total Offer Deduction on Monthly Premium', 'Total Monthly Premium after Offer', 
        'Expected Offer Money given to people actually not in range', 
        'Additional Revenue gained in next month by these offers(Using S curve model)', 
        'Expected Total Profit', 'Expected People in the range Missed', 
        'Expected loss in revenue due to missed people'
    ]
    avg_cols = ['Probability of Acceptance']

    summary_row['Offer'] = f"{table_df['Offer'].str.rstrip('%').astype(float).min()}% - {table_df['Offer'].str.rstrip('%').astype(float).max()}%" 

    for col in sum_cols:
        summary_row[col] = table_df[col].sum()
    for col in avg_cols:
        summary_row[col] = f"{table_df[col].str.rstrip('%').astype(float).mean():.2f}%"

    table_df.loc['Summary'] = summary_row 

    st.subheader("Customer Risk Segmentation Summary")
    st.dataframe(table_df, use_container_width=True)