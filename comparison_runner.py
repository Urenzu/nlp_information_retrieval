import tfidf
import word2vec
import os
from rich.console import Console
from rich.table import Table
from evaluation import evaluate_and_plot

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

def generate_relevance_labels(ranked_docs, query):
    """
    Manually label the relevance of the documents for the given query.
    This should be replaced with an automated process in a real-world setting.
    
    Args:
        ranked_docs: List of document titles or IDs in ranked order.
        query: Query string for information retrieval.
    
    Returns:
        relevance_labels: List of binary labels (1 for relevant, 0 for not relevant).
    """
    keywords = query.lower().split()

    relevance_labels = []
    for doc in ranked_docs:
        if any(keyword in doc.lower() for keyword in keywords):
            relevance_labels.append(1)  # Relevant
        else:
            relevance_labels.append(0)  # Not relevant

    return relevance_labels


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'reuters21578')

    query = "Airplane in Berlin started in morning but landed in the evening."
    run_comparison(query)
    
    tfidf_doc_IDs, tfidf_info = tfidf.information_retrieval_tfidf(path, query, topN=5)
    w2v_doc_IDs, w2v_info, w2v_scores = word2vec.information_retrieval_word2vec(path, query, topN=5)
    tfidf_results = [doc['title'] for doc in tfidf_info]
    word2vec_results = [doc['title'] for doc in w2v_info]

    # Generate relevance labels dynamically
    relevance_labels_tfidf = generate_relevance_labels(tfidf_results, query)
    relevance_labels_w2v = generate_relevance_labels(word2vec_results, query)

    # Evaluate and plot
    evaluate_and_plot(tfidf_results, word2vec_results, relevance_labels_tfidf, relevance_labels_w2v)

    #assigns binary relevance labels (1 for relevant, 0 for not relevant) to each document based on its content and the query.
    print("TF-IDF Results:", tfidf_results)
    print("TF-IDF Relevance Labels:", relevance_labels_tfidf)
    print("Word2Vec Results:", word2vec_results)
    print("Word2Vec Relevance Labels:", relevance_labels_w2v)
