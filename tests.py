import document_processing
import tfidf
import word2vec
import os
import pickle
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt # type: ignore
from rich.progress import Progress

console = Console()

def initial_tfidf(tfidf_path, save_path):
    """
    Train tfidf and save the module
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    console.print("[bold green]Training TF-IDF model...[/bold green]")
    full_counter, tfidf_matrix, idf_vector = tfidf.learn_tfidf(tfidf_path)
    
    tfidf_model = {
        "full_counter": full_counter,
        "tfidf_matrix": tfidf_matrix,
        "idf_vector": idf_vector
    }
    
    with open(os.path.join(save_path, "tfidf_model.pkl"), "wb") as f:
        pickle.dump(tfidf_model, f)
    
    console.print(f"[bold green]TF-IDF model has been saved to:[/bold green] [cyan]{tfidf_path}/tfidf_model.pkl[/cyan]")

def initial_w2v(w2v_path, save_path):
    """
    Train and save Word2Vec model.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    console.print("[bold green]Building corpus with progress tracking...[/bold green]")
    
    # Use Progress for progress tracking
    with Progress() as progress:
        corpus_task = progress.add_task("[cyan]Building corpus...", total=22)
        corpus = word2vec.build_corpus(w2v_path, progress=progress, task=corpus_task)
    
    console.print("[bold green]Training Word2Vec model...[/bold green]")
    w2v_model = word2vec.train_word2vec_model(corpus)

    # Save the model
    w2v_model_path = os.path.join(save_path, "w2v_model.pkl")
    with open(w2v_model_path, "wb") as f:
        pickle.dump(w2v_model, f)

    console.print(f"[bold green]Word2Vec model has been saved to:[/bold green] [cyan]{w2v_model_path}[/cyan]")

def tfidf_retrieval(model_path, datafile_path, query, topN):
    """
    Using the trained tfidf model to do search
    """
    with open(model_path, "rb") as f:
        tfidf_model = pickle.load(f)

    full_counter = tfidf_model["full_counter"]
    tfidf_matrix = tfidf_model["tfidf_matrix"]
    idf_vector = tfidf_model["idf_vector"]

    indexed_counter = tfidf.index_Counter(full_counter)
    tf_idf_query_normalized = tfidf.tdidf_query(query, indexed_counter, idf_vector)

    vector_of_cosines = tfidf.score(tf_idf_query_normalized, tfidf_matrix)

    doc_IDs = tfidf.retrieve_doc_Ids(vector_of_cosines, topN)

    doc_info_dict = tfidf.doc_info(datafile_path)
    results = [{"ID": doc_id, "title": doc_info_dict[doc_id]["title"]} for doc_id in doc_IDs]

    return results

def w2v_retrieval(model_path, datafile_path, query, topN):
    """
    Using the trained Word2Vec model to search.

    Args:
        model_path (str): Path to the saved Word2Vec model.
        datafile_path (str): Path to the data folder.
        query (str): User query.
        topN (int): Number of top results to return.

    Returns:
        list: List of top N results with article titles and IDs.
    """
    with open(model_path, "rb") as f:
        w2v_model = pickle.load(f)

    # Progress tracking for corpus and document embeddings
    with Progress() as progress:
        corpus_task = progress.add_task("[cyan]Building corpus...", total=22)
        corpus = word2vec.build_corpus(datafile_path, progress=progress, task=corpus_task)

        embeddings_task = progress.add_task("[cyan]Computing document embeddings...", total=len(corpus))
        doc_embeddings = word2vec.compute_document_embeddings(corpus, w2v_model, progress=progress, task=embeddings_task)

    # Process the query
    query_tokens = word2vec.preprocess_text(query)
    query_embedding = word2vec.get_document_embedding(query_tokens, w2v_model)

    if query_embedding is None:
        console.print("[bold red]Query embedding failed! Returning empty results.[/bold red]")
        return []

    # Compute similarities
    similarities = word2vec.compute_similarity_scores(query_embedding, doc_embeddings)

    # Get Top N document IDs
    top_indices = similarities.argsort()[-topN:][::-1]
    doc_IDs = top_indices + 1

    # Retrieve document titles and IDs
    doc_info_dict = word2vec.build_doc_info(datafile_path)
    results = [{"ID": doc_id, "title": doc_info_dict[doc_id]["title"]} for doc_id in doc_IDs]

    return results

def test(data_path, save_path, queries):
    """
    Test function to train models, query data, and visualize results
    """
    # Ensure result directory exists
    result_dir = os.path.join(save_path, "result_graph")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # Train and save models
    console.print("[bold green]Training TF-IDF and Word2Vec models...[/bold green]")
    initial_tfidf(data_path, save_path)
    initial_w2v(data_path, save_path)

    # Load models
    with open(os.path.join(save_path, "tfidf_model.pkl"), "rb") as f:
        tfidf_model = pickle.load(f)
    with open(os.path.join(save_path, "w2v_model.pkl"), "rb") as f:
        w2v_model = pickle.load(f)

    # Store AP values for final table
    ap_values = {"tfidf": [], "w2v": []}

    for query_name, query in queries:
        console.print(f"[bold green]Testing with query: {query}[/bold green]")
        
        # Retrieve results for TF-IDF and Word2Vec
        tfidf_results = tfidf_retrieval(
            model_path=os.path.join(save_path, "tfidf_model.pkl"),
            datafile_path=data_path,
            query=query,
            topN=5
        )
        w2v_results = w2v_retrieval(
            model_path=os.path.join(save_path, "w2v_model.pkl"),
            datafile_path=data_path,
            query=query,
            topN=5
        )

        # Create a table for displaying results
        def display_results(results, model_name):
            table = Table(title=f"{model_name} Results for Query: {query}")
            table.add_column("ID", justify="center")
            table.add_column("Title", justify="left")
            table.add_column("Content", justify="left")

            relevant_count = 0
            judged_relevant = []

            for result in results:
                article_content = document_processing.find_article(data_path, result["ID"])
                table.add_row(str(result["ID"]), result["title"], article_content[:50] + "...")

                print(article_content)  # Display full content
                console.print(f"[bold]Is this article relative with the query '{query}'? (y/n): [/bold]")
                response = input()
                judged_relevant.append(response.strip().lower() == "y")
                if judged_relevant[-1]:
                    relevant_count += 1

            console.print(table)
            return judged_relevant, relevant_count

        # Judge relevance and compute precision/recall for both models
        tfidf_judgments, tfidf_relevant_count = display_results(tfidf_results, "TF-IDF")
        w2v_judgments, w2v_relevant_count = display_results(w2v_results, "Word2Vec")

        # Compute recall's denominator
        total_relevant = document_processing.find_total_relatives(data_path, "gold")

        # Precision and recall for plotting
        def compute_precision_recall(judgments, total_relevant, relevant_count):
            if total_relevant == 0:
                console.print("[bold red]Warning: Total relevant documents is zero. Recall cannot be computed.[/bold red]")
                precision = [sum(judgments[:i + 1]) / (i + 1) for i in range(len(judgments))]
                recall = [0] * len(judgments)
                return precision, recall
            precision = [sum(judgments[:i + 1]) / (i + 1) for i in range(len(judgments))]
            recall = [sum(judgments[:i + 1]) / total_relevant for i in range(len(judgments))]
            return precision, recall

        tfidf_precision, tfidf_recall = compute_precision_recall(tfidf_judgments, total_relevant, tfidf_relevant_count)
        w2v_precision, w2v_recall = compute_precision_recall(w2v_judgments, total_relevant, w2v_relevant_count)

        # Compute AP
        def compute_ap(precision, judgments):
            return sum(precision[i] for i in range(len(judgments)) if judgments[i]) / len(judgments)

        tfidf_ap = compute_ap(tfidf_precision, tfidf_judgments)
        w2v_ap = compute_ap(w2v_precision, w2v_judgments)
        ap_values["tfidf"].append(tfidf_ap)
        ap_values["w2v"].append(w2v_ap)

        # Save plot
        plt.figure()
        plt.plot(tfidf_recall, tfidf_precision, label="TF-IDF", marker="o")
        plt.plot(w2v_recall, w2v_precision, label="Word2Vec", marker="x")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve ({query_name})")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(result_dir, f"{query_name}_pr_curve.jpg"))
        console.print(f"[bold blue]AP for TF-IDF: {tfidf_ap:.4f}[/bold blue]")
        console.print(f"[bold blue]AP for Word2Vec: {w2v_ap:.4f}[/bold blue]")

    # Compute MAP
    def compute_map(ap_values):
        return sum(ap_values) / len(ap_values)

    tfidf_map = compute_map(ap_values["tfidf"])
    w2v_map = compute_map(ap_values["w2v"])

    # Print summary table
    summary_table = Table(title="AP and MAP Summary")
    summary_table.add_column("Model", justify="center")
    summary_table.add_column("Simple Query AP", justify="center")
    summary_table.add_column("Intermediate Query AP", justify="center")
    summary_table.add_column("Advanced Query AP", justify="center")
    summary_table.add_column("MAP", justify="center")

    summary_table.add_row(
        "TF-IDF",
        f"{ap_values['tfidf'][0]:.4f}",
        f"{ap_values['tfidf'][1]:.4f}",
        f"{ap_values['tfidf'][2]:.4f}",
        f"{tfidf_map:.4f}"
    )
    summary_table.add_row(
        "Word2Vec",
        f"{ap_values['w2v'][0]:.4f}",
        f"{ap_values['w2v'][1]:.4f}",
        f"{ap_values['w2v'][2]:.4f}",
        f"{w2v_map:.4f}"
    )

    console.print(summary_table)

if __name__ == "__main__":
    # Queries for testing
    queries = [
        ("simple_query", "gold"),
        ("intermediate_query", "money coconut"),
        ("advanced_query", "money coconut nasdaq"),
    ]
    data_path = "reuters21578"
    model_path = "models"
    test(data_path, model_path, queries)
