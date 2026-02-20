"""Download ArXiv papers for the RAG Lab dataset."""

import argparse
from pathlib import Path

import arxiv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def download_papers(
    query: str,
    max_results: int,
    output_dir: Path = Path("data/raw"),
) -> None:
    """
    Download papers from ArXiv based on a search query.

    Args:
        query: Search query for ArXiv 
        max_results: Maximum number of papers to download
        output_dir: Directory to save downloaded PDFs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold blue]Searching ArXiv for:[/] {query}")
    console.print(f"[bold blue]Max results:[/] {max_results}")
    console.print(f"[bold blue]Output directory:[/] {output_dir}\n")

    # Create ArXiv search client
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    downloaded = 0
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading papers...", total=max_results)

        for result in search.results():
            # Create safe filename from paper ID
            paper_id = result.entry_id.split("/")[-1]
            filename = f"{paper_id}.pdf"
            filepath = output_dir / filename

            # Skip if already downloaded
            if filepath.exists():
                progress.console.print(
                    f"[yellow]Skipping[/] {paper_id} (already exists)"
                )
                skipped += 1
                progress.advance(task)
                continue

            try:
                # Download PDF
                result.download_pdf(dirpath=str(output_dir), filename=filename)
                progress.console.print(
                    f"[green]Downloaded[/] {paper_id}: {result.title[:60]}..."
                )
                downloaded += 1
            except Exception as e:
                progress.console.print(f"[red]Error downloading {paper_id}:[/] {e}")

            progress.advance(task)

    console.print(f"\n[bold green]✓ Download complete![/]")
    console.print(f"  Downloaded: {downloaded}")
    console.print(f"  Skipped: {skipped}")
    console.print(f"  Total files: {len(list(output_dir.glob('*.pdf')))}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download ArXiv papers for RAG Lab dataset"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="ArXiv search query (e.g., 'retrieval augmented generation')",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        required=True,
        help="Maximum number of papers to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for downloaded PDFs",
    )

    args = parser.parse_args()

    download_papers(
        query=args.query,
        max_results=args.max_results,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
