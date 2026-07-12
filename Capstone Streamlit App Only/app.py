import streamlit as st
from transformers import pipeline
from news_functions import *


model1_df1, model1_df2, model2_df1, model2_df2, model3_df1, model3_df2 = output_format()

if "changed_model" not in st.session_state:
    st.session_state.changed_model = False
if "button_selected" not in st.session_state:
    st.session_state.button_selected = False

#NEED TO IMPLEMENT CACHING @st.cache_data
st.set_page_config(page_title="Sentiment Analysis on News Headlines", layout="wide")

col1, col2 = st.columns(spec=[1,40], vertical_alignment="center")
with col1:
    st.image(image="imgs/gnews.jpg", width=50)
with col2:
    st.header(f"Financial Headings from today: {news_date()}")

with st.container(border=True):
    tab1, tab2, tab3 = st.tabs(["Kip's Self-Trained Financial Sentiment Model", "DistilRoberta Financial News Sentiment Model", "Prosus AI Finbert Financial Sentiment Model"])
    with tab1:
        st.header("Kip's")
        st.dataframe(model1_df1, hide_index = True)
        st.dataframe(model1_df2, hide_index = True)
    with tab2:
        st.header("DistilRoberta")
        st.dataframe(model2_df1, hide_index = True)
        st.dataframe(model2_df2, hide_index = True)
    with tab3:
        st.header("Prosus AI Finbert")
        st.dataframe(model3_df1, hide_index = True)
        st.dataframe(model3_df2, hide_index = True)


st.header("Try Financial Sentiment Predictions Yourself")
with st.container(border=True):
    left_column, right_column = st.columns(2)

    def predict():
        st.session_state.changed_model = True
        if model == "Kip's Self-Trained Model":
            sentiment_model = pipeline(model="kevinwlip/financial-sentiment-model-5000-samples")
        elif model == "DistilRoberta":
            sentiment_model = pipeline("sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
        elif model == "Prosus AI Finbert":
            sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        elif model == "Fine-Tuned DistilRoberta":
            sentiment_model = pipeline("sentiment-analysis", model="kevinwlip/distilroberta-financial-sentiment-model-5000-samples-fine-tune")
        elif model == "Fine-Tuned Prosus AI Finbert":
            sentiment_model = pipeline("sentiment-analysis", model="kevinwlip/ProsusAI-finbert-5000-samples-fine-tune")

        vals = [result.values() for result in sentiment_model([input])]
        output = [f"{x.capitalize()}, Probability: {y}" for x,y in vals][0]
        output = output.replace("Label_0", "Negative").replace("Label_1", "Positive").replace("Label_2", "Neutral")

        return output

    with left_column:
        st.header("Select Your Model:")
        model = st.selectbox('Model', ["Kip's Self-Trained Model", "DistilRoberta", "Prosus AI Finbert", "Fine-Tuned DistilRoberta", "Fine-Tuned Prosus AI Finbert"])
        st.caption("Trained using 4K Training Data and 1K Test Data")

    with right_column:
        st.header("Input Your Financial Text:")
        input = st.text_input(label = "Financial Text", value = "Sales have risen in the export markets")

        if st.button("Predict Sentiment"):
            st.session_state.button_selected = True
            output = predict()
            st.success(output, icon="🎉")

        st.caption("Label 0: Negative, Label 1: Positive, Label 2: Neutral")
