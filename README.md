# Sniffle
Sniffle is an app for searching Kaggle's CORD-19 corpus of COVID-19 literature. It offers the option to search the corpus using different vectorization methods. The current version offers the option to match query to documents by TF-IDF vectorization, or by vectorization based on LDA topic predictions. Currently, both options use cosine distance to find the nearest documents to the query.

Stay tuned for updates that offer more features, including search by word or sentence embeddings and citation network-based recommendation.

View the app deployed on [Streamlit](https://share.streamlit.io/omlean/sniffle/main/streamlit.py).