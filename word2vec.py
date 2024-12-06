from bs4 import BeautifulSoup
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import numpy as np
from scipy.spatial.distance import cosine
import os
from rich.progress import Progress

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
             if token.isalnum() and token not in stop_words]
    return tokens

def build_corpus(path, progress, task):
    corpus = []
    for j in range(22):
        rem = j % 10
        mult = j // 10
        with open(f"{path}/reut2-0{mult}{rem}.sgm", 'r', encoding='iso-8859-1') as f:
            file_content = f.read()
        soup = BeautifulSoup(file_content, 'html.parser')
        texts = soup.find_all('text')
        for text in texts:
            body = text.find('body')
            if body:
                processed_text = preprocess_text(body.get_text())
                if processed_text:
                    corpus.append(processed_text)
        progress.update(task, advance=1)
    return corpus

def train_word2vec_model(corpus):
    model = Word2Vec(sentences=corpus, vector_size=300, window=10, min_count=5,
                    workers=4, sg=1, negative=15, epochs=20, alpha=0.025,
                    min_alpha=0.0001, sample=1e-5)
    model.corpus_count = len(corpus)
    return model

def get_document_embedding(doc_tokens, model, use_idf=True):
    if not doc_tokens:
        return None
    term_freq = Counter(doc_tokens)
    vectors = []
    weights = []
    
    for token in doc_tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
            if use_idf:
                idf = np.log(model.corpus_count / (model.wv.get_vecattr(token, "count") + 1))
                weights.append(term_freq[token] * idf)
            else:
                weights.append(term_freq[token])
    
    if vectors:
        weights = np.array(weights)
        weights = weights / weights.sum()
        doc_vector = np.average(vectors, weights=weights, axis=0)
        doc_vector = doc_vector / (np.linalg.norm(doc_vector) + 1e-8)
        return doc_vector
    return None

def compute_document_embeddings(corpus, model, progress, task):
    embeddings = []
    total_docs = len(corpus)
    for i, doc in enumerate(corpus):
        embedding = get_document_embedding(doc, model)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            embeddings.append(np.zeros(model.vector_size))
        progress.update(task, advance=100/total_docs)
    return np.array(embeddings)

def information_retrieval_word2vec(path, query, topN=5):
    with Progress() as progress:
        corpus_task = progress.add_task("[cyan]Building corpus...", total=22)
        train_task = progress.add_task("[cyan]Training model...", total=100, visible=False)
        embed_task = progress.add_task("[cyan]Computing embeddings...", total=100, visible=False)
        
        corpus = build_corpus(path, progress, corpus_task)
        
        progress.update(corpus_task, visible=False)
        progress.update(train_task, visible=True)
        model = train_word2vec_model(corpus)
        progress.update(train_task, advance=100)
        
        progress.update(train_task, visible=False)
        progress.update(embed_task, visible=True)
        doc_embeddings = compute_document_embeddings(corpus, model, progress, embed_task)
        
        doc_info_dict = build_doc_info(path)
        query_tokens = preprocess_text(query)
        
        if not query_tokens:
            return [], [], []
        
        query_embedding = get_document_embedding(query_tokens, model)
        if query_embedding is None:
            return [], [], []
            
        similarities = compute_similarity_scores(query_embedding, doc_embeddings)
        doc_info_list = [doc_info_dict[i+1] for i in range(len(similarities))]
        final_scores = rank_documents(similarities, doc_info_list, query_tokens)
        
        top_indices = np.argsort(final_scores)[-topN:][::-1]
        doc_IDs = top_indices + 1
        
        valid_indices = final_scores[top_indices] >= 0.3
        filtered_doc_IDs = doc_IDs[valid_indices]
        filtered_doc_info = [doc_info_dict[id] for id in filtered_doc_IDs]
        filtered_scores = final_scores[top_indices][valid_indices]
        
        return filtered_doc_IDs, filtered_doc_info, filtered_scores

def article(doc_ID, path):
    j = (doc_ID-1)//1000
    remainder = (doc_ID-1)%1000
    rem = j % 10
    mult = j // 10
    with open(f"{path}/reut2-0{mult}{rem}.sgm", 'r', encoding='iso-8859-1') as f:
        file_content = f.read()
    soup = BeautifulSoup(file_content, 'html.parser')
    texts = soup.find_all('text')
    return texts[remainder].get_text()

def compute_similarity_scores(query_embedding, doc_embeddings):
    similarities = []
    for doc_embedding in doc_embeddings:
        if np.all(doc_embedding == 0) or np.all(query_embedding == 0):
            similarities.append(0.0)
            continue
        similarity = 1 - cosine(query_embedding, doc_embedding)
        similarities.append(max(0.0, similarity))
    return np.array(similarities)

def build_doc_info(path):
    doc_info_dict = {}
    for j in range(22):
        with open(f"{path}/reut2-0{j//10}{j%10}.sgm", 'r', encoding='iso-8859-1') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            texts = soup.find_all('text')
            for i, text in enumerate(texts, 1):
                doc_id = j * 1000 + i
                title = text.find('title')
                dateline = text.find('dateline')
                title_text = title.get_text().strip() if title else ""
                date_text = dateline.get_text().strip() if dateline else ""
                topics = text.find('topics')
                topic_list = []
                if topics:
                    for topic in topics.find_all('d'):
                        topic_list.append(topic.get_text().strip())
                doc_info_dict[doc_id] = {
                    'title': title_text,
                    'date': date_text,
                    'topics': topic_list
                }
    return doc_info_dict

def rank_documents(similarities, doc_info_list, query_tokens, boost_factor=0.2):
    scores = similarities.copy()
    for i, doc_info in enumerate(doc_info_list):
        title_tokens = set(preprocess_text(doc_info['title']))
        title_match_ratio = len(set(query_tokens) & title_tokens) / len(query_tokens)
        scores[i] += boost_factor * title_match_ratio
        if any(topic in ' '.join(query_tokens) for topic in doc_info['topics']):
            scores[i] += boost_factor * 0.5
    return scores