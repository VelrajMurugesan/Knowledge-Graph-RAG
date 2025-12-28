"""
Knowledge Graph vs Traditional RAG Demo

This script demonstrates the differences between Traditional RAG and Knowledge Graph-based RAG
using CloudStore, DriveSphere, and SkyVault API documentation as sample data.
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from traditional_rag import TraditionalRAG
from knowledge_graph import KnowledgeGraphRAG
from comparison import compare_systems, run_comparison_suite, plot_comparison_metrics, visualize_graph

console = Console()


# Sample questions that highlight KG advantages across all 3 API documents
DEMO_QUESTIONS = [
    # CloudStore API questions
    "How does the AuthenticationService relate to the UserManager in CloudStore?",
    "What services depend on the PermissionManager in CloudStore?",
    "Explain the file upload workflow in CloudStore and all the services involved.",
    # DriveSphere API questions
    "How does AssetPipeline relate to MediaProcessor and MetadataEnricher in DriveSphere?",
    "What is the relationship between ActivityService and RealtimeGateway in DriveSphere?",
    "How does RoleEngine handle permissions in DriveSphere?",
    # SkyVault API questions
    "How does DocumentService interact with StorageEngine and RetentionService in SkyVault?",
    "What is the relationship between MemberService and AccessControlService in SkyVault?",
    # Cross-API questions
    "Compare authentication methods across CloudStore, DriveSphere, and SkyVault APIs",
    "What are the differences in storage architecture between the three APIs?",
    "How do the three APIs handle document/file permissions differently?"
]


def setup_environment():
    """Load and validate environment variables."""
    load_dotenv()

    required_vars = [
        "OPENAI_API_KEY",
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD"
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        console.print(f"[bold red]Error: Missing required environment variables:[/bold red]")
        for var in missing_vars:
            console.print(f"  - {var}")
        console.print("\n[yellow]Please create a .env file based on .env.example[/yellow]")
        return False

    return True


async def initialize_systems():
    """Initialize both RAG systems."""
    console.print("\n[bold cyan]Initializing Systems...[/bold cyan]\n")

    # Load configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Initialize Traditional RAG
    console.print("[yellow]1. Initializing Traditional RAG...[/yellow]")
    rag_system = TraditionalRAG(
        openai_api_key=openai_api_key,
        model_name=model_name,
        embedding_model=embedding_model
    )

    # Load and index all documents from sample_data folder
    sample_data_path = Path("sample_data")
    document_files = [
        "api_documentation.txt",  # CloudStore API
        "drivesphere_platform_api.txt",  # DriveSphere API
        "skyvault_api_documentation.txt"  # SkyVault API
    ]
    
    all_documents = []
    loaded_files = []
    for doc_file in document_files:
        doc_path = sample_data_path / doc_file
        if not doc_path.exists():
            console.print(f"[yellow]Warning: Sample data not found at {doc_path}[/yellow]")
            continue
        console.print(f"[yellow]Loading {doc_file}...[/yellow]")
        docs = rag_system.load_documents(str(doc_path))
        all_documents.extend(docs)
        loaded_files.append(doc_file)
    
    if not all_documents:
        console.print(f"[bold red]Error: No documents found in sample_data folder[/bold red]")
        return None, None
    
    console.print(f"[yellow]Building index with {len(all_documents)} total chunks...[/yellow]")
    rag_system.build_index(all_documents)
    console.print(f"[green][OK] Traditional RAG initialized with {len(loaded_files)} document(s)[/green]\n")

    # Initialize Knowledge Graph RAG
    console.print("[yellow]2. Initializing Knowledge Graph RAG...[/yellow]")
    kg_system = KnowledgeGraphRAG(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_username,
        neo4j_password=neo4j_password,
        openai_api_key=openai_api_key,
        model_name=model_name
    )

    # Build required Neo4j indexes and constraints
    await kg_system.graphiti.build_indices_and_constraints()

    # Check if we should rebuild the graph
    stats = kg_system.get_graph_statistics()
    if stats['total_nodes'] > 0:
        console.print(f"[yellow]Found existing graph with {stats['total_nodes']} nodes[/yellow]")
        rebuild = Confirm.ask("Do you want to rebuild the knowledge graph?", default=False)
        if rebuild:
            kg_system.clear_graph()
            stats = kg_system.get_graph_statistics()

    # Build knowledge graph if needed
    if stats['total_nodes'] == 0:
        console.print("[yellow]Building knowledge graph (this may take a few minutes)...[/yellow]")
        
        # Load all documents for KG with proper source tags
        document_sources = {
            "api_documentation.txt": "cloudstore_api",
            "drivesphere_platform_api.txt": "drivesphere_api",
            "skyvault_api_documentation.txt": "skyvault_api"
        }
        
        loaded_sources = 0
        # Split documents by source for KG - reuse text splitter from rag_system
        for doc_file, source_name in document_sources.items():
            doc_path = sample_data_path / doc_file
            if not doc_path.exists():
                console.print(f"[yellow]Warning: {doc_file} not found, skipping...[/yellow]")
                continue
            
            console.print(f"[yellow]Adding {doc_file} to knowledge graph...[/yellow]")
            # Load document content
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks for KG processing using the same text splitter
            doc_chunks = rag_system.text_splitter.split_text(content)
            
            # Add each chunk to the graph
            await kg_system.add_documents_to_graph(doc_chunks, source=source_name)
            loaded_sources += 1
        
        stats = kg_system.get_graph_statistics()
        console.print(f"[green][OK] Knowledge Graph initialized with {loaded_sources} document source(s)[/green]")
        console.print(f"  - Nodes: {stats['total_nodes']}")
        console.print(f"  - Relationships: {stats['total_relationships']}")
        console.print(f"  - Entities: {stats['num_entities']}")
        console.print(f"  - Episodes: {stats['num_episodes']}\n")
    else:
        console.print(f"[green][OK] Using existing Knowledge Graph[/green]")
        console.print(f"  - Nodes: {stats['total_nodes']}")
        console.print(f"  - Relationships: {stats['total_relationships']}")
        console.print(f"  - Entities: {stats['num_entities']}")
        console.print(f"  - Episodes: {stats['num_episodes']}\n")

    return rag_system, kg_system


async def run_single_comparison(rag_system, kg_system):
    """Run a single question comparison."""
    console.print("\n[bold cyan]Single Question Comparison[/bold cyan]\n")

    # Show available demo questions
    console.print("[yellow]Suggested questions:[/yellow]")
    for i, q in enumerate(DEMO_QUESTIONS, 1):
        console.print(f"  {i}. {q}")

    console.print(f"\n[yellow]Enter a question number (1-{len(DEMO_QUESTIONS)}) or type your own question:[/yellow]")
    user_input = Prompt.ask("Question")

    # Parse input
    try:
        question_num = int(user_input)
        if 1 <= question_num <= len(DEMO_QUESTIONS):
            question = DEMO_QUESTIONS[question_num - 1]
        else:
            question = user_input
    except ValueError:
        question = user_input

    # Run comparison
    await compare_systems(rag_system, kg_system, question, verbose=True)


async def run_full_comparison_suite(rag_system, kg_system):
    """Run the full comparison suite with all demo questions."""
    console.print("\n[bold cyan]Running Full Comparison Suite[/bold cyan]\n")
    console.print(f"This will test both systems with {len(DEMO_QUESTIONS)} predefined questions.\n")

    confirm = Confirm.ask("Continue?", default=True)
    if not confirm:
        return

    # Run suite
    results = await run_comparison_suite(rag_system, kg_system, DEMO_QUESTIONS)

    # Generate visualizations
    console.print("\n[yellow]Generating comparison visualizations...[/yellow]")
    plot_comparison_metrics(results, "comparison_metrics.png")
    console.print("[green][OK] Metrics plot saved to: comparison_metrics.png[/green]")


def visualize_knowledge_graph(kg_system):
    """Generate knowledge graph visualization."""
    console.print("\n[bold cyan]Generating Knowledge Graph Visualization[/bold cyan]\n")

    visualize_graph(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USERNAME"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        output_file="knowledge_graph.html",
        max_nodes=100
    )

    console.print("[green][OK] Visualization saved to: knowledge_graph.html[/green]")
    console.print("[yellow]Open this file in a web browser to explore the graph interactively[/yellow]")


async def interactive_mode(rag_system, kg_system):
    """Run interactive question-answering mode."""
    console.print("\n[bold cyan]Interactive Mode[/bold cyan]\n")
    console.print("[yellow]Ask questions about CloudStore, DriveSphere, or SkyVault APIs (type 'exit' to quit)[/yellow]\n")

    while True:
        question = Prompt.ask("\n[bold]Your question[/bold]")

        if question.lower() in ['exit', 'quit', 'q']:
            break

        await compare_systems(rag_system, kg_system, question, verbose=True)


async def main():
    """Main demo function."""
    console.print(Panel.fit(
        "[bold green]Knowledge Graph vs Traditional RAG Demo[/bold green]\n"
        "Demonstrating the advantages of Knowledge Graph-based RAG over Traditional RAG\n"
        "Using CloudStore, DriveSphere, and SkyVault API documentation",
        border_style="green"
    ))

    # Setup
    if not setup_environment():
        return

    # Initialize systems
    rag_system, kg_system = await initialize_systems()
    if not rag_system or not kg_system:
        return

    # Main menu
    while True:
        console.print("\n" + "=" * 80)
        console.print("[bold cyan]Demo Menu[/bold cyan]")
        console.print("=" * 80)
        console.print("1. Run single question comparison")
        console.print("2. Run full comparison suite (all demo questions)")
        console.print("3. Visualize knowledge graph")
        console.print("4. Interactive mode (ask your own questions)")
        console.print("5. View graph statistics")
        console.print("6. Exit")

        choice = Prompt.ask("\n[bold]Select an option[/bold]", choices=["1", "2", "3", "4", "5", "6"])

        if choice == "1":
            await run_single_comparison(rag_system, kg_system)
        elif choice == "2":
            await run_full_comparison_suite(rag_system, kg_system)
        elif choice == "3":
            visualize_knowledge_graph(kg_system)
        elif choice == "4":
            await interactive_mode(rag_system, kg_system)
        elif choice == "5":
            stats = kg_system.get_graph_statistics()
            console.print("\n[bold cyan]Knowledge Graph Statistics:[/bold cyan]")
            console.print(f"  - Total Nodes: {stats['total_nodes']}")
            console.print(f"  - Total Relationships: {stats['total_relationships']}")
            console.print(f"  - Entities: {stats['num_entities']}")
            console.print(f"  - Episodes: {stats['num_episodes']}")
        elif choice == "6":
            console.print("\n[bold green]Thank you for using the demo![/bold green]")
            kg_system.close()
            break


if __name__ == "__main__":
    asyncio.run(main())
