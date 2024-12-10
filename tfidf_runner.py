import tfidf
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def run_tfidf_search(query, topN=10):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'reuters21578')
    
    console = Console()
    console.print("\n[bold blue]TF-IDF Search Results[/bold blue]")
    console.print(f"Query: {query}\n")
    
    doc_IDs, doc_info_list = tfidf.information_retrieval_tfidf(path, query, topN=topN)
    
    table = Table(show_header=True)
    table.add_column("Rank")
    table.add_column("Title")
    table.add_column("Date")
    
    for i, info in enumerate(doc_info_list, 1):
        table.add_row(f"{i}", info['title'], info['date'])
    
    console.print(table)
    
    if len(doc_IDs) > 0:
        top_article = tfidf.article(doc_IDs[0], path)
        console.print("\n[bold]Top Article:[/bold]")
        console.print(top_article)

if __name__ == "__main__":
    #query = "Airplane in Berlin started in morning but landed in the evening."
    
    run_tfidf_search(query)