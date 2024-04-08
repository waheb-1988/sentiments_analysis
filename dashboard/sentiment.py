import sys
sys.path.append('../')
from models.TF_IDF import TFIDF
import streamlit as st
import PIL
from pathlib import Path
import os
from PIL import Image

# Local Modules
import settings

### read data
instance = TFIDF("df_contact","jumia_reviews_df_multi_page")
data = instance.read_data()
#print(data.info())

# ###
product= data["product"].unique()

########################### App
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Sentiment Analysis")

# Sidebar
option = st.sidebar.selectbox(
    'Select Kaggle Database',
    (product),
    )

st.sidebar.write('You selected:', option)

st.sidebar.header("Feature Extraction")

# Model Options
model_type = st.sidebar.radio(
    "Select option", [None,'TFIDF', 'BagOfWord'])


col1, col2 = st.columns(2)  
col3 = st.columns(1)[0]
col4 = st.columns(1)[0]
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
                
if model_type == 'TFIDF':
    # Load TFIDF image      
    with col1:
                st.subheader("Rating Distribution {}".format(model_type))
                default_image_11 = Image.open(settings.IMAGES_DIR_RATINGN)
                st.image(default_image_11, caption="Rating Dis", use_column_width=True)

    with col2:
                st.subheader("Sentiment Distribution {}".format(model_type))
                default_image_21 = Image.open(settings.IMAGES_DIR_SENTIMENTN)
                st.image(default_image_21, caption="Sentiment Dis", use_column_width=True)

    with col3: 
                st.subheader("WordMap Distribution {}".format(model_type))
                default_image_33 = Image.open(settings.IMAGES_DIR_WordmapN)
                st.image(default_image_33, caption="WordMap Dis", use_column_width=True)

st.sidebar.header("ML models")
models= TFIDF.models()
result = instance.train_model()
option = st.sidebar.selectbox(
    'ML models',
    (models),
    )

st.sidebar.write('You selected:', option)
with col4:
    st.dataframe(result.style.highlight_max(axis=0))