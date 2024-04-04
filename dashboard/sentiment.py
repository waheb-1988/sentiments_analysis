import sys
sys.path.append('../')
from models.TF_IDF import TFIDF
import streamlit as st


instance = TFIDF("df_contact","jumia_reviews_df_multi_page")
data = instance.read_data()
#print(data.info())
###
product= data["product"].unique()
option = st.selectbox(
    'Select Kaggle Database',
    (product),
    )

st.write('You selected:', option)

