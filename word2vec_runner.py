import word2vec
import os
from rich.console import Console
from rich.table import Table

def run_word2vec_search(query, topN=5):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'reuters21578')
    
    console = Console()
    console.print("\n[bold blue]Word2Vec Search Results[/bold blue]")
    console.print(f"Query: {query}\n")
    
    doc_IDs, doc_info_list, similarity_scores = word2vec.information_retrieval_word2vec(
        path, query, topN=topN
    )
    
    table = Table(show_header=True)
    table.add_column("Rank")
    table.add_column("Score")
    table.add_column("Title")
    table.add_column("Date")
    
    for i, (score, info) in enumerate(zip(similarity_scores, doc_info_list), 1):
        table.add_row(
            f"{i}",
            f"{score:.3f}",
            info['title'],
            info['date']
        )
    
    console.print(table)
    
    if len(doc_IDs) > 0:
        top_article = word2vec.article(doc_IDs[0], path)
        console.print("\n[bold]Top Article:[/bold]")
        console.print(top_article)

if __name__ == "__main__":
    query = "Airplane in Berlin started in morning but landed in the evening."
    run_word2vec_search(query)