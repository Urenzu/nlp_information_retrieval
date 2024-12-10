import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def calculate_precision_recall(ranked_docs, relevance_labels):
    """
    Calculate precision, recall, and interpolated precision.

    Args:
        ranked_docs: List of document IDs in the ranked order.
        relevance_labels: List of binary labels indicating relevance (1 for relevant, 0 for not).

    Returns:
        precision: List of precision values.
        recall: List of recall values.
        interpolated_precision: Interpolated precision for precision-recall curve.
    """
    relevant_count = np.cumsum(relevance_labels)  # Cumulative relevant docs
    precision = relevant_count / np.arange(1, len(ranked_docs) + 1)
    recall = relevant_count / sum(relevance_labels)

    # Interpolate precision
    interpolated_precision = np.maximum.accumulate(precision[::-1])[::-1]
    return precision, recall, interpolated_precision

def plot_combined_precision_recall_curve(tfidf_precision, tfidf_recall, w2v_precision, w2v_recall):
    """
    Plot combined precision-recall curve for TF-IDF and Word2Vec on the same axes.

    Args:
        tfidf_precision: List of precision values for TF-IDF.
        tfidf_recall: List of recall values for TF-IDF.
        w2v_precision: List of precision values for Word2Vec.
        w2v_recall: List of recall values for Word2Vec.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(tfidf_recall, tfidf_precision, marker="o", label="TF-IDF")
    plt.plot(w2v_recall, w2v_precision, marker="x", label="Word2Vec")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve: TF-IDF vs Word2Vec")
    plt.legend()
    plt.grid()
    plt.show()

def calculate_map(precision, relevance_labels):
    """
    Calculate Mean Average Precision (MAP).

    Args:
        precision: List of precision values.
        relevance_labels: List of binary labels indicating relevance (1 for relevant, 0 for not).

    Returns:
        mean_average_precision: MAP score.
    """
    # Calculate average precision for relevant documents
    average_precision = [
        precision[i] for i in range(len(relevance_labels)) if relevance_labels[i] == 1
    ]
    return np.mean(average_precision) if average_precision else 0.0

def evaluate_and_plot(
    tfidf_results, word2vec_results, relevance_labels_tfidf, relevance_labels_w2v, plot_title
):
    """
    Evaluate and compare TF-IDF and Word2Vec results and plot combined precision-recall curves.

    Args:
        tfidf_results: Ranked document IDs for TF-IDF.
        word2vec_results: Ranked document IDs for Word2Vec.
        relevance_labels_tfidf: Relevance labels for TF-IDF results.
        relevance_labels_w2v: Relevance labels for Word2Vec results.
        plot_title: Title for the precision-recall plot.
    """
    # TF-IDF evaluation
    tfidf_precision, tfidf_recall, _ = calculate_precision_recall(
        tfidf_results, relevance_labels_tfidf
    )
    tfidf_map = calculate_map(tfidf_precision, relevance_labels_tfidf)
    print(f"TF-IDF AP: {tfidf_map:.4f}")

    # Word2Vec evaluation
    w2v_precision, w2v_recall, _ = calculate_precision_recall(
        word2vec_results, relevance_labels_w2v
    )
    w2v_map = calculate_map(w2v_precision, relevance_labels_w2v)
    print(f"Word2Vec AP: {w2v_map:.4f}")

    # Plot combined precision-recall curves
    plt.figure(figsize=(8, 6))
    plt.plot(tfidf_recall, tfidf_precision, marker="o", label="TF-IDF")
    plt.plot(w2v_recall, w2v_precision, marker="x", label="Word2Vec")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(plot_title)
    plt.legend()
    plt.grid()
    plt.show()
    
# Example Usage
if __name__ == "__main__":


    evaluate_and_plot(tfidf_results, word2vec_results, relevance_labels_tfidf, relevance_labels_w2v)
