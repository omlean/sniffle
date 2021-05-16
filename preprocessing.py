# import libraries
import pandas as pd
import json
import nltk
import re
import string
from tqdm import tqdm
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def parse_questions(filepath):
    """Parses the JSON files containing consumer and expert questions from EPIC-QA.
    Returns DataFrame"""
    with open(filepath, 'r') as file:
        j = json.load(file)
    
    question_id, question, query, background = [], [], [], []
    
    for item in j:
        question_id.append(item['question_id'])
        question.append(item['question'])
        query.append(item['query'])
        background.append(item['background'])
        
    df = pd.DataFrame(data={'question_id': question_id,
                        'question': question,
                        'query': query,
                        'background': background})
    
    return df

##############################################################################

def drop_emptier_duplicates(df, col):
    """For all sets of rows with the same value of duplicate_column, keep only the one with the fewest NaNs"""
    duplicates_df = df[df[col].duplicated(keep=False)]
    duplicates_df['nans'] = duplicates_df.apply(lambda x: x.isnull().sum(), axis=1)
    droplist = []
    print("Choosing rows to drop")
    for value in tqdm(duplicates_df[col].unique()):
        sets = duplicates_df[duplicates_df[col] == value]
        for i in sets.sort_values('nans', ascending=False).iloc[1:].index:
            droplist.append(i)
    return df.drop(index=droplist)

##############################################################################

def clean_metadata(filepath):
    """Loads and preprocesses CORD-19 metadata table at the specified location.
    Returns DataFrame"""
    df = pd.read_csv(filepath, low_memory=False)
    print('CSV file loaded successfully')
    
    # drop unwanted columns
#     drop_columns = ['sha', 'license']
#     df = df.drop(columns=drop_columns)

    # drop rows with no title (these appear to be non-English articles)
    df = df[df.title.notnull()]
    
    # convert publish_time column to datetime format
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    
    # drop rows with identical pdf and pmcs
    c1 = df.pdf_json_files.notnull()
    c2 = df.pmc_json_files.notnull()
    df_has_file = df[c1 | c2]
    duplicates = df_has_file[df_has_file[['pdf_json_files','pmc_json_files']].duplicated(keep='first')]
    drops = duplicates.index
    df = df.drop(index=drops)
    drop_index = df[df.pdf_json_files.duplicated(keep='first') & df.pdf_json_files.notnull()].index
    df = df.drop(index=drop_index)
    
    # remove duplicated titles more than 10 words
    c1 = df.title.apply(lambda x: len(x.split()) > 10)
    c2 = df.title.duplicated(keep='first')
    drop_index = df[c1 & c2].index
    df = df.drop(index=drop_index)
    
    # remove duplicate abstracts more than 50 words
    c1 = df.abstract.apply(lambda x: len(str(x).split()) > 50)
    c2 = df.abstract.duplicated(keep='first')
    drop_index = df[c1 & c2].index
    df = df.drop(index=drop_index)
    
    print('Dropping duplicate uids')
    df = drop_emptier_duplicates(df, 'cord_uid')

    
    print('Metadata cleaning complete')
    
    return df

##############################################################################

def clean_metadata_for_lda(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    print('CSV file loaded successfully')
    print(f'{len(df)} rows')

    print("Dropping rows with no title")
    df = df[df.title.notnull()]
    print(f'{len(df)} rows')

    print("Converting publish_time column to datetime format")
    df['publish_time'] = pd.to_datetime(df['publish_time'])

    print("Dropping rows with duplicate pdf files")
    drop_index = df[df.pdf_json_files.notnull() & df.pdf_json_files.duplicated(keep='first')].index
    df = df.drop(index=drop_index)
    print(f'{len(df)} rows')

    print("Removing duplicated titles more than 10 words")
    c1 = df.title.apply(lambda x: len(x.split()) > 10)
    c2 = df.title.duplicated(keep='first')
    drop_index = df[c1 & c2].index
    df = df.drop(index=drop_index)
    print(f'{len(df)} rows')

    print("Removing duplicate abstracts more than 50 words")
    c1 = df.abstract.apply(lambda x: len(str(x).split()) > 50)
    c2 = df.abstract.duplicated(keep='first')
    drop_index = df[c1 & c2].index
    df = df.drop(index=drop_index)
    print(f'{len(df)} rows')

    print('Dropping duplicate uids')
    duplicates_df = df[df['cord_uid'].duplicated(keep=False)]
    duplicates_df['nans'] = duplicates_df.apply(lambda x: x.isnull().sum(), axis=1)
    droplist = []
    print("Choosing rows to drop")
    for value in tqdm(duplicates_df['cord_uid'].unique()):
        sets = duplicates_df[duplicates_df['cord_uid'] == value]
        for i in sets.sort_values('nans', ascending=False).iloc[1:].index:
            droplist.append(i)
    df = df.drop(index=droplist)
    print(f'{len(df)} rows')
    print('Metadata cleaning complete')
    return df

##############################################################################

def load_cleaned_metadata(path):
    # load dataframe
    df = pd.read_csv(path, index_col=0, sep='\t', low_memory=False)
    
    # convert publish_time to datetime format
    df.publish_time = pd.to_datetime(df.publish_time)
    
#     # fill nans with "none"
#     none_fill = ['pmcid', 'pubmed_id', 'mag_id', 'who_covidence_id', 'arxiv_id', 's2_id',
#             'pdf_json_files', 'pmc_json_files']
#     for col in none_fill:
#         df[col] = df[col].fillna('none')
    
#     # fill nans with "unknown"
#     unknown_fill = ['doi', 'publish_time', 'authors', 'journal', 'url']
#     for col in unknown_fill:
#         df[col] = df[col].fillna('unknown')
    
    # fill empty abstracts with empty string
    empty_fill = ['abstract']
    for col in empty_fill:
        df[col] = df[col].fillna('')
        
    return df        

##############################################################################

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

##############################################################################

def make_search_documents(df, stem=False, lemmatize=True, stopword_list=stopwords.words('english')):
    """Input: dataframe whose titles and abstracts are to be merged into clean documents for vectorization.
        Must contain columns ['cord_uid', 'title', 'abstract']
    Output: List of cleaned strings consisting of titles and abstracts"""
    l = []
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        s = row.title + " " + row.abstract
        cleaned_text = clean_text(s, stem=stem, lemmatize=lemmatize, stopword_list=stopword_list)
        l.append(cleaned_text)

    return l



##############################################################################

def clean_text_lda(s, stem=False, lemmatize=True, stopword_list=stopwords.words('english')):
    s = s.lower() # lowercase
    s = re.sub(r'[-–]', '', s) # delete hyphens
    s = re.sub(r'[^a-z|0-9|\s]', '', s) # remove anything that isn't alphanumeric or whitespace
    s = " ".join([w for w in word_tokenize(s) if w not in stopword_list+['et', 'al']]) # remove stopwords and 'et al'
    
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
    
    # Drop numbers, words containing more than 4 digits, or beginning with digits then letters

    s = re.sub(r'\w+\d[a-z]*\d[a-z]*\d[a-z]*\d[a-z]*\d[\d\w]*', '', s)
    s = re.sub(r' \d+[a-z]+\W*', '', s)
    s = " ".join(word.strip() for word in s.split())
    for i in range(10):
        s = re.sub(r' \d+', ' ', s) # remove numbers
    s = re.sub(r'  +', ' ', s)
    
    return s

##############################################################################

from data_access import get_txt
# get text, clean, and write to new file
def get_clean_write(uid, dest_directory, source_directory='data/cord-19/body_text/lda_raw/'):
    text = get_txt(uid, directory=source_directory)
    text = clean_text_lda(text)
    dest_path = dest_directory + uid + '_clean.txt'
    with open(dest_path, 'w') as file:
        file.write(text)