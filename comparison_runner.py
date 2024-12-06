import tfidf
import word2vec
import os
from rich.console import Console
from rich.table import Table

def run_comparison(query, topN=5):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'reuters21578')
    
    console = Console()
    console.print("\n[bold blue]TF-IDF vs Word2Vec Comparison[/bold blue]")
    console.print(f"Query: {query}\n")
    
    tfidf_doc_IDs, tfidf_info = tfidf.information_retrieval_tfidf(path, query, topN=topN)
    w2v_doc_IDs, w2v_info, w2v_scores = word2vec.information_retrieval_word2vec(
        path, query, topN=topN
    )
    
    table = Table(show_header=True)
    table.add_column("Rank")
    table.add_column("TF-IDF Results")
    table.add_column("Word2Vec Results")
    table.add_column("W2V Score")
    
    for i in range(topN):
        tfidf_title = tfidf_info[i]['title'] if i < len(tfidf_info) else "-"
        w2v_title = w2v_info[i]['title'] if i < len(w2v_info) else "-"
        w2v_score = f"{w2v_scores[i]:.3f}" if i < len(w2v_scores) else "-"
        
        table.add_row(
            f"{i+1}",
            tfidf_title,
            w2v_title,
            w2v_score
        )
    
    console.print(table)
    
    overlap = len(set(tfidf_doc_IDs) & set(w2v_doc_IDs))
    console.print(f"\nResults Overlap: {overlap} documents")

if __name__ == "__main__":
    query = "Airplane in Berlin started in morning but landed in the evening."
    run_comparison(query)