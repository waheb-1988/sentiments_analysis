import sys
sys.path.append('../')
from models.TF_IDF import TFIDF
import streamlit as st
import PIL
from pathlib import Path
import os
from PIL import Image
from itables.streamlit import interactive_table
# Local Modules
import settings
import requests
from streamlit_lottie import st_lottie
# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="NLP Feature Exctraction and Sentiment Analyser", page_icon=":tada:", layout="wide")

# ---- LOAD ASSETS ----
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

# ---- WHAT I DO ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header(" Sentiment Analyzer NLP Feature Extrcation" )
        st.write("##")
        st.write(
            """
            - TFIDF
            - BAGOFW
            - CBOW
            - COUNTV
            - NGRAM
            - Machine Learning
            - HyperTuning
            - Balnced and Unbalanced
            
            """
        )
        
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")

### read data
instance = TFIDF("df_contact","jumia_reviews_df_multi_page")
data = instance.read_data()
#print(data.info())

# ###
product= data["product"].unique()

# Sidebar
option = st.sidebar.selectbox(
    'Select Kaggle Database',
    (product),
    )

st.sidebar.write('You selected:', option)

st.sidebar.header("Feature Extraction")

# Model Options
model_type = st.sidebar.radio(
    "Select option", [None,'TFIDF', 'BAGOFW' , 'CBOW' , 'COUNTV' , 'NGRAM'])


col1, col2 = st.columns(2)  
col3 = st.columns(1)[0]
col4, col5 = st.columns(2)
if model_type == None:
    # Load TFIDF image   
    
    with col1:
                st.subheader("Default Rating Distribution")  # Add a title
                default_image_1 = Image.open(settings.IMAGES_DIR_RATING)
                st.image(default_image_1, caption="Default Rating Dis", use_column_width=True)

    with col2:
                st.subheader("Default Sentiment Distribution")
                default_image_2 = Image.open(settings.IMAGES_DIR_SENTIMENT)
                st.image(default_image_2, caption="Default Sentiment Dis", use_column_width=True)
     
    with col3:
                st.subheader("Default WordMap Distribution")
                default_image_3 = Image.open(settings.IMAGES_DIR_Wordmap)
                st.image(default_image_3, caption="Default WordMap ", use_column_width=True)
                
else:
    st.sidebar.header("ML models")
    models = TFIDF.models()  # Assuming this returns a list of models
    result = instance.train_model()  # Assuming this trains the model and returns a DataFrame
    result1 = instance.hyper_tun()
    option = st.sidebar.selectbox(
        'ML models',
        models,
    )

    st.sidebar.write('You selected:', option)
    
    with col4:
        st.subheader(f"Intial ML models")
        st.dataframe(result.style.highlight_max(axis=0))
        
    with col5:
        st.subheader(f"HyperTunning ML models")
        st.dataframe(result1.style.highlight_max(axis=0))
    if model_type == 'TFIDF':
        with col1:
            st.subheader(f"Rating Distribution {model_type}")
            tfidf_image_1 = Image.open(settings.IMAGES_DIR_RATINGN)
            st.image(tfidf_image_1, caption="Rating Dis", use_column_width=True)

        with col2:
            st.subheader(f"Sentiment Distribution {model_type}")
            tfidf_image_2 = Image.open(settings.IMAGES_DIR_SENTIMENTN)
            st.image(tfidf_image_2, caption="Sentiment Dis", use_column_width=True)

        with col3:
            st.subheader(f"WordMap Distribution {model_type}")
            tfidf_image_3 = Image.open(settings.IMAGES_DIR_WordmapN)
            st.image(tfidf_image_3, caption="WordMap Dis", use_column_width=True)