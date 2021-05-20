import streamlit as st
import numpy as np
import pandas as pd
import re
from datetime import datetime
from scipy.sparse import load_npz
from scipy.spatial.distance import cosine
from gensim.matutils import cossim

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

##########################################################################################

# @st.cache
def load_data():
    df = pd.read_csv('data/data_streamlit.csv.gz', sep='\t')
    df.date = pd.to_datetime(df.date)
    df.abstract.fillna('nan', inplace=True)
    df.url.fillna('None', inplace=True)
    return df

def clean_text(s, stem=False, lemmatize=True, stopword_list=stopwords.words('english')):
    """Removes punctuation, lowercases, removes stopwords, 
    removes digit-only words, stems (optional) and lemmatizes (optional).
    Note: to remove no stopwords, pass [] as stopword_list."""
    s = s.lower() # lowercase
    s = " ".join([w for w in word_tokenize(s) if w not in stopword_list]) # remove stopwords
    s = s.lower() # lowercase
    s = re.sub(r'[-–]', ' ', s) # replace hyphens with spaces
    s = re.sub(r'[^a-z|0-9|\s]', '', s) # remove anything that isn't alphanumeric or whitespace
    s = re.sub(r'\s\d+\s', ' ', s) # remove digit-only words
    
    if stem:
        porter = PorterStemmer()
        stemmed_words = []
        for word in word_tokenize(s):
            stemmed_words.append(porter.stem(word))
        s = ' '.join(stemmed_words)
        
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = []
        for word in word_tokenize(s):
            lemmatized_words.append(lemmatizer.lemmatize(word, pos='v'))
        s = ' '.join(lemmatized_words)
    
    # Drops words containing more than 4 digits, or beginning with digits then letters
    s = re.sub(r'\d[a-z]*\d[a-z]*\d[a-z]*\d[a-z]*\d[\d\w]*', '', s)
    s = re.sub(r'\d+[a-z]+\W*', '', s)
    s = " ".join(word.strip() for word in s.split())
    
    return s

# function for cleaning and vectorizing input query
def clean_vectorize_query(query_string, vectorizer):
    """Input: query string (raw), vectorizer fit to search corpus.
    Output: Query vector."""
    clean = clean_text(query_string)
    v = vectorizer.transform([clean]).toarray()
    return v

def search(query, vectorizer, term_document_matrix, index, num_top_results=5):
    """Input:
    query: raw query string
    term_document_matrix: term-document matrix transformed with same vectorizer as the query vector
    index: lookup index for document IDs, e.g. a list or Series. Must return the relevant cord_uid for vector i using `index[i]`
    Output: array of cord_uids for top results."""
    
    v = clean_vectorize_query(query, vectorizer)
    print('Vectorized search query')
    
    num_documents = term_document_matrix.shape[0]
    scores = np.ones(num_documents)
     
    print('Computing document similarity...')
    progress = st.progress(0)
    for i in range(num_documents):
        scores[i] = cosine(v, term_document_matrix[i,:].toarray())
        progress.progress(i/num_documents)
    progress.progress(100)
    top_results = np.argsort(scores)[:num_top_results]
    top_results = np.array(top_results)
    
    uids = [index[i] for i in top_results]
    print(f'Returned top {num_top_results} results.')
    
    return uids


def execute_search(query, vectorizer, tdm, index, num_top_results, data):
    status = st.text("Searching...")
    uids = search(query, vectorizer, tdm, index, num_top_results)
    status.text("Top results found")
    results_table = data[data.cord_uid.apply(lambda x: x in uids)]
    return results_table

def write_results(results_table):
    for i in range(len(results_table)):
        row = results_table.iloc[i]
        st.markdown(f'## {row.title}')
        if row.abstract != 'None':
            st.markdown("Abstract: \n" + row.abstract + "\n\n")
        st.markdown(f'Authors: _{row.authors}_')
        st.markdown(f"Journal: **{row.journal}**, {datetime.strftime(row.date, '%Y-%m-%d')}")
        for url in row.url.split(';'):
            if url.strip() != 'None':
                st.markdown(f'[{url.strip()}]({url.strip()})')
            else:
                st.markdown('No URL available')

######################################################################################################
# LDA functions


class MyCorpus():
    
    def __init__(self, search_texts, dictionary=None):
        self.docs = search_texts
        self.dictionary = dictionary
        if self.dictionary is not None:
            self.id2word = self.dictionary.id2token
    
    def __len__(self):
        return len(self.docs)
        
    def __iter__(self):
        for doc in self.docs:
            yield self.dictionary.doc2bow(doc.split())

def lda_search(query, dictionary, model, corpus, metadata, num_top_results=5):
    query = clean_text(query)
    query_bow = dictionary.doc2bow(query.split())
    query_vec = model[query_bow]
    distances = []
    progress = st.progress(0)
    n = 0
    for doc in corpus:
        distances.append(cossim(query_vec, model[doc]))
        n += 1
        progress.progress(n/len(corpus))
    distances = np.array(distances)
    top_results = np.argsort(distances)[-num_top_results:][::-1] # find n closest documents
    results_df = metadata.iloc[top_results]
    return results_df
##########################################################################################
