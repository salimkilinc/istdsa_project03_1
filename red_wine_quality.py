import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Red Wine Quality Classifier",
    page_icon="https://account.citibikenyc.com/favicon.ico",
    menu_items={
        "Get help": "mailto:salimkilinc@yahoo.com",
        "About": "For More Information\n" + "https://github.com/salimkilinc"
    }
)

st.title("Red Wine Quality Classifier")

st.markdown("An upscale restaurant wants to decide whether the quality of a red wine is **:red[Low]** or **:red[High]** based on some wine data from a database.")

st.image("https://w0.peakpx.com/wallpaper/577/271/HD-wallpaper-red-wine-glass-of-wine-bottle-of-wine-grapes.jpg")

st.markdown("Following recent advancements in the artificial intelligence sector, they anticipate us to create a **machine learning model** that aligns with their requirements and supports their research endeavors.")
st.markdown("They also want us to develop a product that, after receiving information about a new type of red wine, can predict whether the quality of this red wine is low or high using the information provided.")
st.markdown("*Let's lend our assistance to them!*")

st.image("https://assets.architecturaldigest.in/photos/6008342bb3d78db39997cec9/16:9/w_2240,c_limit/Let-Awakening-inspire-your-wine-selection-1366x768.jpg")

st.markdown("- **quality**: whether a red wine's quality is low or high (0 = low, 1 = high)")
st.markdown("- **fixed_acidity**: most acids involved with wine or fixed or nonvolatile (do not evaporate readily)")
st.markdown("- **volatile_acidity**: the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste")
st.markdown("- **citric_acid**: found in small quantities, citric acid can add 'freshness' and flavor to wines")
st.markdown("- **residual_sugar**: the amount of sugar remaining after fermentation stops, it's rare to find wines with less than 1 gram/liter")
st.markdown("- **chlorides**: the amount of salt in the wine")
st.markdown("- **free_sulfur_dioxide**: the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion")
st.markdown("- **total_sulfur_dioxide**: amount of free and bound forms of S02; in low concentrations")
st.markdown("- **density**: the density of water is close to that of water depending on the percent alcohol and sugar content")
st.markdown("- **ph**: describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic)")
st.markdown("- **sulphates**: a wine additive which can contribute to sulfur dioxide gas (S02) levels")
st.markdown("- **alcohol**: alcohol content of wine by volume")

df = pd.read_csv("red_wine_quality.csv")
sample_df = df

st.table(sample_df.sample(5, random_state=33))


st.sidebar.markdown("**Select** the features from the options below to view the outcome!")

name = st.sidebar.text_input("Name", help="Please ensure that the initial letter of your name is capitalized.")
surname = st.sidebar.text_input("Surname", help="Please ensure that the initial letter of your surname is capitalized.")
fixed_acidity = st.sidebar.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, step=0.1, format="%.1f")
volatile_acidity = st.sidebar.number_input("Volatile Acidity", min_value=0.00, max_value=2.00, step=0.01)
citric_acid = st.sidebar.number_input("Citric Acid", min_value=0.00, max_value=1.00, step=0.01)
residual_sugar = st.sidebar.number_input("Residual Sugar", min_value=0.0, max_value=20.0, step=0.1, format="%.1f")
chlorides = st.sidebar.number_input("Chlorides", min_value=0.000, max_value=1.000, step=0.001, format="%.3f")
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", min_value=0, max_value=200, step=1)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", min_value=0, max_value=200, step=1)
density = st.sidebar.number_input("Density", min_value=0.99000, max_value=1.00320, step=0.00001, format="%.5f")
ph = st.sidebar.number_input("pH", min_value=2.00, max_value=4.00, step=0.01)
sulphates = st.sidebar.number_input("Sulphates", min_value=0.00, max_value=2.00, step=0.01)
alcohol = st.sidebar.number_input("Alcohol", min_value=8.0, max_value=15.0, step=0.1, format="%.1f")


from joblib import load

rf_model = load('rf_model.pkl')

input_df = pd.DataFrame({
    'fixed_acidity': [fixed_acidity],
    'volatile_acidity': [volatile_acidity],
    'citric_acid': [citric_acid],
    'residual_sugar': [residual_sugar],
    'chlorides': [chlorides],
    'free_sulfur_dioxide': [free_sulfur_dioxide],
    'total_sulfur_dioxide': [total_sulfur_dioxide],
    'density': [density],
    'ph': [ph],
    'sulphates': [sulphates],
    'alcohol': [alcohol]
})

pred = rf_model.predict(input_df)
pred_probability = np.round(rf_model.predict_proba(input_df), 2)


st.header("Outcome")

if st.sidebar.button("Submit"):

    st.info("The outcome is located beneath.")

    from datetime import date, datetime

    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")

    results_df = pd.DataFrame({
    'Name': [name],
    'Surname': [surname],
    'Date': [today],
    'Time': [time],
    'fixed_acidity': [fixed_acidity],
    'volatile_acidity': [volatile_acidity],
    'citric_acid': [citric_acid],
    'residual_sugar': [residual_sugar],
    'chlorides': [chlorides],
    'free_sulfur_dioxide': [free_sulfur_dioxide],
    'total_sulfur_dioxide': [total_sulfur_dioxide],
    'density': [density],
    'ph': [ph],
    'sulphates': [sulphates],
    'alcohol': [alcohol],
    'Prediction': [pred]
    })

    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("0","Low Quality"))
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("1","High Quality"))

    st.table(results_df)

    if pred == 0:
        st.image("https://w.forfun.com/fetch/4d/4d2e2094c306b0c6245d136e286f6eaa.jpeg")
    else:
        st.image("https://w.forfun.com/fetch/20/203c306b2f5c0c2b619ad2f25c9f4244.jpeg")
else:
    st.markdown("Please click on **Submit** button!")