import streamlit as st
import numpy as np
import pandas as pd
import re
from datetime import datetime
from scipy.sparse import load_npz
from scipy.spatial.distance import cosine

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

from my_functions import *
    

# title
st.sidebar.title('SNIFFLE')
num_top_results = st.sidebar.slider("Number of results", min_value=1, max_value=20, value=10, step=1)
st.sidebar.markdown("""Welcome to Sniffle &#151; a search engine for scientific literature related to the COVID-19 pandemic. It was created as part of my capstone project for the Data Science bootcamp at [Lighthouse Labs](https://www.lighthouselabs.ca/en/data-science-bootcamp).

This app uses the CORD-19 dataset available on [Kaggle](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge).

See the [repo](https://github.com/omlean/sniffle) on Github.

> Note: To focus on the most up-to-date information (and to save search time), this version returns documents dated in 2021.
""")

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# Load tf-idf search data
data = load_data()
index = data.cord_uid.values
with open('data/streamlit_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
tdm = load_npz('data/streamlit_tdm.npz') # load term-document matrix

# Load LDA objects
model = LdaModel.load('data/lda_100_0.01_0.1_model')
dictionary = Dictionary.load('data/dictionary.dict')
corpus = MyCorpus(data.cord_uid.tolist(), dictionary=dictionary)


# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

query = st.text_area("Enter search query here:", "Evidence for effectiveness of masks in preventing COVID-19 transmission.")
search_method = st.selectbox('Select search method', ['TF-IDF','LDA Topics'], index=0)
start_search = st.button("Search")

if start_search:
    if search_method == 'TF-IDF':
        results_table = execute_search(query, vectorizer, tdm, index, num_top_results, data)
        write_results(results_table)
    if search_method == 'LDA Topics':
        results_table = lda_search(query, model, corpus, dictionary, data, num_top_results)
        write_results(results_table)

