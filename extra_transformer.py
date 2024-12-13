import nltk
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.status import Status
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.special import softmax
import re
from collections import Counter
import time

console = Console()

def download_nltk_data():
    with console.status("[bold blue]Downloading NLTK data...") as status:
        for resource in ['reuters', 'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'words']:
            status.update(f"[bold blue]Downloading {resource}...")
            nltk.download(resource, quiet=True)
            time.sleep(0.5)

class EnhancedRetrieval:
    def __init__(self):
        self.console = Console()
        with self.console.status("[bold blue]Initializing models...") as status:
            status.update("[bold blue]Loading lemmatizer and stopwords...")
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            status.update("[bold blue]Loading BERT tokenizer...")
            self.bert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            
            status.update("[bold blue]Loading BERT model...")
            self.bert_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            self.bert_model.eval()
            
            status.update("[bold blue]Loading English dictionary...")
            self.english_words = set(nltk.corpus.words.words())
            
        self.document_vectors = {}
        self.document_info = {}

    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        tokens = [self.lemmatizer.lemmatize(token, self._get_wordnet_pos(tag))
                 for token, tag in pos_tags
                 if token not in self.stop_words and len(token) > 2]
        return [t for t in tokens if t in self.english_words]

    def _get_wordnet_pos(self, treebank_tag):
        return {'J': nltk.corpus.wordnet.ADJ,
                'V': nltk.corpus.wordnet.VERB,
                'N': nltk.corpus.wordnet.NOUN,
                'R': nltk.corpus.wordnet.ADV}.get(treebank_tag[0], nltk.corpus.wordnet.NOUN)

    def _get_bert_embedding(self, text):
        inputs = self.bert_tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def train_models(self):
        file_ids = reuters.fileids()
        total_files = len(file_ids)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            
            self.console.print("\n[bold blue]Starting model training...[/bold blue]")
            
            process_task = progress.add_task(
                "[cyan]Processing documents...",
                total=total_files
            )
            
            for i, file_id in enumerate(file_ids):
                raw_text = reuters.raw(file_id)
                
                progress.update(process_task, 
                              description=f"[cyan]Processing document {i+1}/{total_files}: {file_id}")
                
                self.document_info[file_id] = {
                    'title': ' '.join(reuters.words(file_id))[:50] + '...',
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'keywords': self._extract_keywords(raw_text)
                }
                
                bert_vector = self._get_bert_embedding(raw_text)
                self.document_vectors[file_id] = bert_vector.flatten()
                
                progress.advance(process_task)
            
            self.console.print("\n[bold green]Model training completed successfully![/bold green]")
            self.console.print(f"[green]Processed {total_files} documents[/green]")

    def _extract_keywords(self, text):
        tokens = self.preprocess_text(text)
        return [word for word, _ in Counter(tokens).most_common(5)]

    def search(self, query, topN=5):
        with self.console.status("[bold blue]Processing search query...") as status:
            query_tokens = self.preprocess_text(query)
            if not query_tokens:
                return [], [], []

            status.update("[bold blue]Generating query embeddings...")
            query_vector = self._get_bert_embedding(query).flatten()
            
            status.update("[bold blue]Calculating document similarities...")
            similarities = {}
            for file_id in reuters.fileids():
                doc_vector = self.document_vectors[file_id]
                
                vector_sim = cosine_similarity(query_vector.reshape(1, -1),
                                            doc_vector.reshape(1, -1))[0][0]
                
                keyword_overlap = len(set(query_tokens) & 
                                   set(self.document_info[file_id]['keywords'])) / 5.0
                
                final_score = 0.7 * vector_sim + 0.3 * keyword_overlap
                similarities[file_id] = final_score

            sorted_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:topN]
            return ([doc_id for doc_id, _ in sorted_docs],
                    [self.document_info[doc_id] for doc_id, _ in sorted_docs],
                    [score for _, score in sorted_docs])

def display_results(query, doc_IDs, doc_info_list, similarity_scores):
    console.print(f"\n[bold blue]Enhanced Neural Search Results[/bold blue]")
    console.print(f"Query: {query}\n")
    
    table = Table(show_header=True)
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Title", style="blue")
    table.add_column("Keywords", style="magenta")
    
    for i, (score, info) in enumerate(zip(similarity_scores, doc_info_list), 1):
        table.add_row(
            f"{i}",
            f"{score:.3f}",
            info['title'],
            ', '.join(info['keywords'])
        )
    
    console.print(table)
    
    if doc_IDs:
        console.print("\n[bold]Top Article:[/bold]")
        console.print(reuters.raw(doc_IDs[0]))

def main():
    console.print("[bold blue]Initializing Enhanced Neural Search System[/bold blue]\n")
    
    download_nltk_data()
    
    with console.status("[bold blue]Setting up search system...") as status:
        retriever = EnhancedRetrieval()
    
    retriever.train_models()
    
    console.print("\n[bold green]System ready for queries![/bold green]")
    console.print("[cyan]Enter your search queries below, or type 'quit' to exit.[/cyan]\n")
    
    while True:
        query = input("\nEnter your search query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            console.print("\n[bold blue]Thank you for using Enhanced Neural Search![/bold blue]")
            break
        doc_IDs, doc_info_list, similarity_scores = retriever.search(query)
        display_results(query, doc_IDs, doc_info_list, similarity_scores)

if __name__ == "__main__":
    main()