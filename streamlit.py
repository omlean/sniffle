# import libraries
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.sparse import load_npz
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import pickle
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

# import custom functions
from my_functions import *
    

# title
st.sidebar.title('SNIFFLE')
num_top_results = st.sidebar.slider("Number of results", min_value=1, max_value=20, value=10, step=1)
st.sidebar.markdown("""Welcome to Sniffle &#151; a search engine for scientific literature related to the COVID-19 pandemic. It's an ongoing project initially created as part of my capstone project for the Data Science bootcamp at [Lighthouse Labs](https://www.lighthouselabs.ca/en/data-science-bootcamp).

This app uses the CORD-19 dataset available on [Kaggle](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge). See the [repo](https://github.com/omlean/sniffle) on Github.

> Note: To focus on the most up-to-date information (and to save search time), this version returns documents dated in 2021.

Sniffle allows you to search the corpus using either of two methods: 

1. _TF-IDF vectorization_. This method emphasises the words that are most characteristic of the document relative to the entire corpus, i.e. words that are common in the document and/or rare across the corpus as a whole. 
2. _LDA Topic vectorization_. This method vectorizes the document/query by representing its predicted topic profile. The topics in question are based on an [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) topic model pre-trained on the corpus.

Each method has different advantages: TF-IDF uses direct string-to-string matching, so it works well for finding documents about specific, named entities like a particular drug or gene (provided that drug or gene is given the same exact name in the document as in the query).

On the other hand, LDA-based searching abstracts from direct word matches and focuses instead on collections of words that tend to be mentioned together in the same documents, and are hence assumed to be semantically related. This makes it less able to pinpoint specific narrow search interests, but is able to find documents which are topically connected in a more general sense. For example, a search for the "effectiveness of remdesivir for treating COVID-19" may return documents about _other_ antiviral treatments &#151; treatments that the user may not be aware of but may nevertheless be interested to know about. This makes it useful for more open-ended, exploratory searching.

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
corpus = MyCorpus(data.search_text.tolist(), dictionary=dictionary)

# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

# text box for user to input query
query = st.text_area("Enter search query here:", "Evidence for effectiveness of masks in preventing COVID-19 transmission.")

# choice of vectorization method
search_method = st.selectbox('Select search method.', ['TF-IDF','LDA Topics'], index=0)

# "Search" button
start_search = st.button("Search")

if start_search:
    if search_method == 'TF-IDF':
        results_table = execute_search(query, vectorizer, tdm, index, num_top_results, data)
        write_results(results_table)
    if search_method == 'LDA Topics':
        results_table = lda_search(query, dictionary, model, corpus, data, num_top_results=num_top_results)
#         for i in range(len(results_table)):
#             print(results_table.iloc[i].title)
        write_results(results_table)

