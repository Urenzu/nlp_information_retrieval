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
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import re

def map_pos_tag(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) - {'in', 'at', 'on', 'during', 'before', 'after', 'to', 'from'}
    
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = word_tokenize(text.lower())
    
    pos_tags = pos_tag(tokens)
    processed_tokens = []
    
    for token, tag in pos_tags:
        if token not in stop_words and len(token) > 1:
            pos = map_pos_tag(tag)
            lemma = lemmatizer.lemmatize(token, pos=pos)
            processed_tokens.append(lemma)
    
    return processed_tokens

def extract_named_entities(text):
    try:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        named_entities = ne_chunk(pos_tags)
        entities = []
        for chunk in named_entities:
            if hasattr(chunk, 'label'):
                entities.append(' '.join(c[0] for c in chunk))
        return entities
    except Exception as e:
        print(f"Warning: Error in named entity extraction: {e}")
        return []

def build_corpus(path, progress, task):
    corpus = []
    entity_dict = {}
    doc_count = 0
    
    for j in range(22):
        try:
            rem = j % 10
            mult = j // 10
            with open(f"{path}/reut2-0{mult}{rem}.sgm", 'r', encoding='iso-8859-1') as f:
                file_content = f.read()
            soup = BeautifulSoup(file_content, 'html.parser')
            texts = soup.find_all('text')
            
            for text in texts:
                doc_count += 1
                body = text.find('body')
                if body:
                    content = body.get_text()
                    entities = extract_named_entities(content)
                    entity_dict[doc_count] = entities
                    processed_text = preprocess_text(content)
                    if processed_text:
                        corpus.append(processed_text)
            
            progress.update(task, advance=1)
        except Exception as e:
            print(f"Warning: Error processing file {j}: {e}")
            continue
    
    return corpus, entity_dict

def train_word2vec_model(corpus):
    model = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=2,
                    workers=4, sg=1, negative=10, epochs=30, alpha=0.025,
                    min_alpha=0.0001, sample=1e-5)
    model.corpus_count = len(corpus)
    return model

def get_document_embedding(doc_tokens, model, use_idf=True):
    if not doc_tokens:
        return None
        
    term_freq = Counter(doc_tokens)
    total_terms = sum(term_freq.values())
    vectors = []
    weights = []
    
    for token in doc_tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
            if use_idf:
                tf = term_freq[token] / total_terms
                idf = np.log(model.corpus_count / (model.wv.get_vecattr(token, "count") + 1))
                weights.append(tf * idf)
            else:
                weights.append(term_freq[token] / total_terms)
    
    if vectors:
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-8)
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
        
        corpus, entity_dict = build_corpus(path, progress, corpus_task)
        
        progress.update(corpus_task, visible=False)
        progress.update(train_task, visible=True)
        model = train_word2vec_model(corpus)
        progress.update(train_task, advance=100)
        
        progress.update(train_task, visible=False)
        progress.update(embed_task, visible=True)
        doc_embeddings = compute_document_embeddings(corpus, model, progress, embed_task)
        
        doc_info_dict = build_doc_info(path)
        query_tokens = preprocess_text(query)
        query_entities = extract_named_entities(query)
        
        if not query_tokens:
            return [], [], []
        
        query_embedding = get_document_embedding(query_tokens, model)
        if query_embedding is None:
            return [], [], []
            
        similarities = compute_similarity_scores(query_embedding, doc_embeddings)
        doc_info_list = [doc_info_dict[i+1] for i in range(len(similarities))]
        final_scores = rank_documents(similarities, doc_info_list, query_tokens, 
                                   entity_dict, query_entities)
        
        top_indices = np.argsort(final_scores)[-topN:][::-1]
        doc_IDs = top_indices + 1
        
        valid_indices = final_scores[top_indices] >= 0.4
        filtered_doc_IDs = doc_IDs[valid_indices]
        filtered_doc_info = [doc_info_dict[id] for id in filtered_doc_IDs]
        filtered_scores = final_scores[top_indices][valid_indices]
        
        return filtered_doc_IDs, filtered_doc_info, filtered_scores

def article(doc_ID, path):
    try:
        j = (doc_ID-1)//1000
        remainder = (doc_ID-1)%1000
        rem = j % 10
        mult = j // 10
        with open(f"{path}/reut2-0{mult}{rem}.sgm", 'r', encoding='iso-8859-1') as f:
            file_content = f.read()
        soup = BeautifulSoup(file_content, 'html.parser')
        texts = soup.find_all('text')
        return texts[remainder].get_text()
    except Exception as e:
        print(f"Warning: Error retrieving article {doc_ID}: {e}")
        return "Article could not be retrieved."

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
        try:
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
        except Exception as e:
            print(f"Warning: Error processing file {j} for doc info: {e}")
            continue
    return doc_info_dict

def rank_documents(similarities, doc_info_list, query_tokens, entity_dict, query_entities, boost_factor=0.3):
    scores = similarities.copy()
    
    for i, doc_info in enumerate(doc_info_list):
        title_tokens = set(preprocess_text(doc_info['title']))
        title_match_ratio = len(set(query_tokens) & title_tokens) / max(len(query_tokens), 1)
        scores[i] += boost_factor * title_match_ratio
        
        topic_boost = 0
        for topic in doc_info['topics']:
            if topic.lower() in ' '.join(query_tokens).lower():
                topic_boost += 0.4
            elif any(qt in topic.lower() for qt in query_tokens):
                topic_boost += 0.2
        scores[i] += topic_boost
        
        if i+1 in entity_dict:
            doc_entities = entity_dict[i+1]
            entity_matches = len(set(query_entities) & set(doc_entities))
            scores[i] += 0.3 * entity_matches

    return scores