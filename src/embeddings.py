# src/embeddings.py

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def embed_and_save(X_train, X_val, X_test, embedding_types=["tfidf", "sbert"], repo_root=None):
    """
    Embeds text data using TF-IDF and SBERT and saves numpy arrays.

    Parameters:
    - X_train, X_val, X_test: arrays of text
    - embedding_types: list, choose from ['tfidf', 'sbert']
    - repo_root: optional, absolute path to repo root (auto-detected if None)
    """
    
    # Detect repo root if not provided
    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # go up from notebooks folder

    embedded_base = os.path.join(repo_root, "data", "embedded")
    
    # Ensure base folder exists
    os.makedirs(embedded_base, exist_ok=True)

    if "tfidf" in embedding_types:
        # --- TF-IDF embeddings ---
        tfidf_folder = os.path.join(embedded_base, "tfidf")
        os.makedirs(tfidf_folder, exist_ok=True)

        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
        X_val_tfidf = vectorizer.transform(X_val).toarray()
        X_test_tfidf = vectorizer.transform(X_test).toarray()

        # Save arrays
        np.save(os.path.join(tfidf_folder, "X_train.npy"), X_train_tfidf)
        np.save(os.path.join(tfidf_folder, "X_val.npy"), X_val_tfidf)
        np.save(os.path.join(tfidf_folder, "X_test.npy"), X_test_tfidf)

        print("TF-IDF embeddings saved in:", tfidf_folder)

    if "sbert" in embedding_types:
        # --- SBERT embeddings ---
        sbert_folder = os.path.join(embedded_base, "sbert")
        os.makedirs(sbert_folder, exist_ok=True)

        model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight SBERT model
        X_train_sbert = model.encode(X_train, show_progress_bar=True)
        X_val_sbert = model.encode(X_val, show_progress_bar=True)
        X_test_sbert = model.encode(X_test, show_progress_bar=True)

        # Save arrays
        np.save(os.path.join(sbert_folder, "X_train.npy"), X_train_sbert)
        np.save(os.path.join(sbert_folder, "X_val.npy"), X_val_sbert)
        np.save(os.path.join(sbert_folder, "X_test.npy"), X_test_sbert)

        print("SBERT embeddings saved in:", sbert_folder)

    print("All selected embeddings completed successfully!")
