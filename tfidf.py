from bs4 import BeautifulSoup
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize 
from sklearn.preprocessing import normalize
import numpy as np
import os
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn
from rich.console import Console
from rich.status import Status
import time

def doc_info(path):
    console = Console()
    doc_info = {}
    
    with Progress() as progress:
        file_task = progress.add_task("[cyan]Reading documents...", total=22)
        
        for j in range(22):
            rem = j % 10
            mult = j // 10
            
            filepath = os.path.join(path, f'reut2-0{mult}{rem}.sgm')
            with open(filepath, 'r', encoding='iso-8859-1') as f:
                file_content = f.read()
            
            soup = BeautifulSoup(file_content, 'html.parser')
            text = soup.find('text')
            
            for i in range(1, 1001):
                if text:
                    title = text.find('title')
                    dateline = text.find('dateline')
                
                    title_text = title.get_text().strip() if title else ""
                    date_text = dateline.get_text().strip() if dateline else ""
                    
                    doc_info[j*1000+i] = {'title': title_text, 'date': date_text}
                
                try:
                    text = text.find_next('text')
                except:
                    break
                    
            progress.update(file_task, advance=1)
    
    return doc_info

def count_all(path):
    console = Console()
    my_counter = Counter()
    
    with Progress() as progress:
        file_task = progress.add_task("[cyan]Building vocabulary...", total=22)
        
        for j in range(22):
            rem = j % 10
            mult = j // 10
            
            with open(path + '/reut2-0' + str(mult) + str(rem) + '.sgm', 'r', encoding='iso-8859-1') as f:
                file_content = f.read()
            
            soup = BeautifulSoup(file_content, 'html.parser')
            text = soup.find('text')
            
            while text:
                body = text.find('body')
                if body:
                    body_text = body.get_text()
                    my_counter.update(word_tokenize(body_text.strip()))
                text = text.find_next('text')
            
            progress.update(file_task, advance=1)
    
    return my_counter

def count_each_doc_chunk(full_counter, path, start_doc, end_doc):
    V = len(full_counter)
    chunk_size = end_doc - start_doc
    frequency_matrix = np.zeros((V, chunk_size))
    
    indexed_counter = index_Counter(full_counter)
    start_file = start_doc // 1000
    end_file = (end_doc - 1) // 1000
    
    for j in range(start_file, end_file + 1):
        rem = j % 10
        mult = j // 10
        
        with open(path+'/reut2-0'+str(mult)+str(rem)+'.sgm', 'r', encoding='iso-8859-1') as f:
            file_content = f.read()
        
        soup = BeautifulSoup(file_content, 'html.parser')
        text = soup.find('text')
        
        doc_in_file = 0
        while doc_in_file < (start_doc % 1000) and text:
            text = text.find_next('text')
            doc_in_file += 1
        
        while text and (j * 1000 + doc_in_file) < end_doc:
            if (j * 1000 + doc_in_file) >= start_doc:
                body = text.find('body')
                if body:
                    body_text = body.get_text()
                    my_counter = Counter()
                    my_counter.update(word_tokenize(body_text.strip()))
                    
                    chunk_idx = (j * 1000 + doc_in_file) - start_doc
                    frequency_matrix[:, chunk_idx] = helper2(indexed_counter, my_counter)
            
            text = text.find_next('text')
            doc_in_file += 1
    
    return frequency_matrix

def tfidf(full_counter, path):
    console = Console()
    V = len(full_counter)
    N = 21578
    chunk_size = 1000
    
    with Progress() as progress:
        df_task = progress.add_task(
            "[cyan]Calculating document frequencies...", 
            total=N
        )
        
        document_frequency = np.zeros(V)
        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            chunk_matrix = count_each_doc_chunk(full_counter, path, start_idx, end_idx)
            document_frequency += np.sum(chunk_matrix > 0, axis=1)
            progress.update(df_task, advance=chunk_size)
        
        with console.status("[cyan]Calculating IDF...") as status:
            idf = np.log10(N/document_frequency)
            idf[document_frequency == 0] = 0
        
        tfidf_task = progress.add_task(
            "[cyan]Calculating TF-IDF matrix...", 
            total=N
        )
        
        tf_idf = np.zeros((V, N))
        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            chunk_matrix = count_each_doc_chunk(full_counter, path, start_idx, end_idx)
            
            tf_chunk = 1 + np.log10(chunk_matrix)
            tf_chunk[chunk_matrix == 0] = 0
            tf_idf_chunk = tf_chunk * idf[:, np.newaxis]
            
            norms = np.sqrt(np.sum(tf_idf_chunk ** 2, axis=0))
            norms[norms == 0] = 1
            tf_idf_chunk = tf_idf_chunk / norms
            
            tf_idf[:, start_idx:end_idx] = tf_idf_chunk
            progress.update(tfidf_task, advance=chunk_size)
    
    console.print("[green]TF-IDF calculation complete![/green]")
    return tf_idf, idf

def tdidf_query(query, indexed_counter,idf):  
    my_counter = Counter()
    my_counter.update(word_tokenize(query.strip()))
    query_frequency_vector = helper2(indexed_counter, my_counter)
    tf_query = 1+np.log10(query_frequency_vector)
    tf_query[tf_query==-np.inf] = 0 
    
    tfidf_query = tf_query*idf
    tf_idf_query_normalized = tfidf_query/np.sqrt(np.sum(tfidf_query**2))
    

               
    return tf_idf_query_normalized

def score(tf_idf_query_normalized,tf_idf_normalized):
    return np.dot(tf_idf_normalized.T,tf_idf_query_normalized)

def retrieve_doc_Ids(vector_of_cosines,topN=10): # retrieves the topN ids for documents with highest cosine values e.g. topN = 10
    max_top_N_indices_unsorted = np.argpartition(vector_of_cosines, -topN)[-topN:]   # uses a partial sort see docs.
    max_top_N_indices_sorted_decreasing = max_top_N_indices_unsorted[np.argsort(vector_of_cosines[max_top_N_indices_unsorted])[::-1]] # sorts indices in increasing order by values, then reverses the order.
    doc_Ids = max_top_N_indices_sorted_decreasing + 1   # add one to account for numbering scheme of docs which starts from 1
    
    return doc_Ids

def doc_info_selected(doc_ID_arr,doc_info_dict):
    return [doc_info_dict[idd]for idd in doc_ID_arr]

def article(doc_ID,path):
    j = (doc_ID-1)//1000
    remainder = (doc_ID-1)%1000
    
    rem=j%10
    mult=j//10
    
    with open(path+'/reut2-0'+str(mult)+str(rem)+'.sgm','r',encoding='iso-8859-1') as f:
        file_content = f.read()

    soup = BeautifulSoup(file_content, 'html.parser')
    
    text = soup.find_all('text')

    return text[remainder].get_text()

def index_Counter(full_counter):
    indexed_counter = Counter({key: index for index, key in enumerate(full_counter.keys())}) 
    return indexed_counter

def helper2(indexed_counter, my_counter):
    V = len(indexed_counter)
    doc_word_frequency = np.zeros(V)
    for key, value in my_counter.items():
        if key in indexed_counter:
            index = indexed_counter[key]
            doc_word_frequency[index] = value
    return doc_word_frequency
    
def learn_tfidf(path):
    console = Console()
    
    with console.status("[cyan]Building vocabulary...") as status:
        full_counter = count_all(path)
        vocab_size = len(full_counter)
    
    with console.status(f"[cyan]Computing TF-IDF matrix for {vocab_size} terms...") as status:
        tfidf_mat, idf_vec = tfidf(full_counter, path)
    
    return full_counter, tfidf_mat, idf_vec

def information_retrieval_tfidf(path, query, topN):
    console = Console()
    
    if not hasattr(information_retrieval_tfidf, 'full_counter'):
        with console.status("[cyan]Initializing TF-IDF model...") as status:
            information_retrieval_tfidf.doc_info_dict = doc_info(path)
            information_retrieval_tfidf.full_counter, information_retrieval_tfidf.tfidf_mat, information_retrieval_tfidf.idf_vec = learn_tfidf(path)
    
    with console.status("[cyan]Processing query...") as status:
        indices = index_Counter(information_retrieval_tfidf.full_counter)
        tf_idf_query_normalized = tdidf_query(query, indices, information_retrieval_tfidf.idf_vec)
        vector_of_cosines = score(tf_idf_query_normalized, information_retrieval_tfidf.tfidf_mat)
        doc_IDs = retrieve_doc_Ids(vector_of_cosines, topN)
    
    return doc_IDs, (doc_info_selected(doc_IDs, information_retrieval_tfidf.doc_info_dict))
    