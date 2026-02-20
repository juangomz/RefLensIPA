"""Evaluator for running metrics on RAG test sets."""

import os
import statistics
import time
from datetime import datetime

from langfuse import get_client
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from src.rag.config import settings
from src.rag.evaluation.metrics import Metric

console = Console()


class EvaluationExample(BaseModel):
    """Single example for evaluation."""

    query: str
    answer: str
    contexts: list[str]
    trace_id: str | None = None


class MetricStatistics(BaseModel):
    """Statistical summary for a metric."""

    mean: float
    std_dev: float
    min: float
    max: float
    median: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile
    count_below_50: int  # Count of scores < 0.5
    count_50_to_70: int  # Count of scores 0.5-0.7
    count_above_70: int  # Count of scores > 0.7
    percent_below_50: float
    percent_50_to_70: float
    percent_above_70: float


class MetricReport(BaseModel):
    """Report for a single metric across all examples."""

    metric_name: str
    average_score: float
    statistics: MetricStatistics
    individual_scores: list[float]
    individual_reasoning: list[str]
    duration_seconds: float  # Time taken to evaluate this metric


class QueryLevelResult(BaseModel):
    """Results for a single query across all metrics."""

    query_index: int
    query: str
    metric_scores: dict[str, float]  # metric_name -> score
    overall_score: float  # Average across all metrics


class EvaluationReport(BaseModel):
    """Complete evaluation report across all metrics."""

    timestamp: str
    configuration: dict[str, str]  # All environment variables as key-value pairs
    num_examples: int
    total_duration_seconds: float  # Total time taken for all evaluations
    metric_reports: list[MetricReport]
    query_level_results: list[QueryLevelResult]


class RAGEvaluator:
    """Evaluator for RAG systems using multiple metrics."""

    def __init__(self, metrics: list[Metric]):
        """
        Initialize RAG evaluator.

        Args:
            metrics: List of metrics to evaluate
        """
        self.metrics = metrics

    def evaluate(
        self,
        evaluation_examples: list[EvaluationExample],
    ) -> EvaluationReport:
        """
        Evaluate RAG system on evaluation examples.

        Args:
            evaluation_examples: List of examples to evaluate

        Returns:
            Evaluation report with results for all metrics
        """
        start_time = time.time()

        console.print(f"\n[bold blue]Evaluating {len(evaluation_examples)} examples...[/]\n")

        metric_reports = []
        all_scores = {}  # metric_name -> list of scores

        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            metric_start_time = time.time()

            console.print(f"[yellow]Running {metric_name}...[/]")

            individual_scores = []
            individual_reasoning = []

            # Use tqdm for progress bar
            for example in tqdm(
                evaluation_examples,
                desc=f"  {metric_name}",
                unit="example",
                leave=True,
            ):
                result = metric.evaluate(
                    query=example.query,
                    answer=example.answer,
                    contexts=example.contexts,
                )

                individual_scores.append(result.score)
                individual_reasoning.append(result.reasoning)

                # Send score to LangFuse if trace_id is available
                if example.trace_id:
                    get_client().create_score(
                        name=metric_name,
                        value=result.score,
                        trace_id=example.trace_id,
                        data_type="NUMERIC",
                        comment=result.reasoning,
                    )

            metric_duration = time.time() - metric_start_time

            # Calculate statistics
            stats = self._calculate_statistics(individual_scores)
            all_scores[metric_name] = individual_scores

            avg_score = sum(individual_scores) / len(individual_scores)

            metric_reports.append(
                MetricReport(
                    metric_name=metric_name,
                    average_score=avg_score,
                    statistics=stats,
                    individual_scores=individual_scores,
                    individual_reasoning=individual_reasoning,
                    duration_seconds=metric_duration,
                )
            )

            console.print(
                f"[bold green]Average {metric_name}: {avg_score:.3f} "
                f"(completed in {metric_duration:.1f}s)[/]\n"
            )

        # Create query-level results
        query_level_results = self._create_query_level_results(
            evaluation_examples, all_scores
        )

        # Get configuration from environment
        configuration = self._get_configuration()

        total_duration = time.time() - start_time

        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            configuration=configuration,
            num_examples=len(evaluation_examples),
            total_duration_seconds=total_duration,
            metric_reports=metric_reports,
            query_level_results=query_level_results,
        )

        console.print(
            f"[bold cyan]Total evaluation time: {total_duration:.1f}s "
            f"({total_duration/60:.1f} minutes)[/]\n"
        )

        self._print_summary(report)
        self._print_query_results(report)

        return report

    def _calculate_statistics(self, scores: list[float]) -> MetricStatistics:
        """
        Calculate statistical summary for a list of scores.

        Args:
            scores: List of metric scores

        Returns:
            Statistical summary
        """
        n = len(scores)
        sorted_scores = sorted(scores)

        # Count scores in different ranges
        count_below_50 = sum(1 for s in scores if s < 0.5)
        count_50_to_70 = sum(1 for s in scores if 0.5 <= s <= 0.7)
        count_above_70 = sum(1 for s in scores if s > 0.7)

        return MetricStatistics(
            mean=statistics.mean(scores),
            std_dev=statistics.stdev(scores) if n > 1 else 0.0,
            min=min(scores),
            max=max(scores),
            median=statistics.median(scores),
            q25=statistics.quantiles(scores, n=4)[0] if n > 1 else scores[0],
            q75=statistics.quantiles(scores, n=4)[2] if n > 1 else scores[0],
            count_below_50=count_below_50,
            count_50_to_70=count_50_to_70,
            count_above_70=count_above_70,
            percent_below_50=round(count_below_50 / n * 100, 2),
            percent_50_to_70=round(count_50_to_70 / n * 100, 2),
            percent_above_70=round(count_above_70 / n * 100, 2),
        )

    def _create_query_level_results(
        self,
        evaluation_examples: list[EvaluationExample],
        all_scores: dict[str, list[float]],
    ) -> list[QueryLevelResult]:
        """
        Create per-query results across all metrics.

        Args:
            evaluation_examples: List of evaluation examples
            all_scores: Dictionary mapping metric names to score lists

        Returns:
            List of query-level results
        """
        query_results = []

        for i, example in enumerate(evaluation_examples):
            metric_scores = {
                metric_name: scores[i] for metric_name, scores in all_scores.items()
            }
            overall_score = sum(metric_scores.values()) / len(metric_scores)

            query_results.append(
                QueryLevelResult(
                    query_index=i,
                    query=example.query,
                    metric_scores=metric_scores,
                    overall_score=overall_score,
                )
            )

        return query_results

    def _get_configuration(self) -> dict[str, str]:
        """
        Get configuration from settings.

        Returns:
            Dictionary of all settings as key-value pairs
        """
        # Convert settings to dictionary
        config = {}
        for field_name in settings.model_fields.keys():
            value = getattr(settings, field_name)
            config[field_name] = str(value)

        return config

    def _print_summary(self, report: EvaluationReport):
        """Print summary table of evaluation results."""
        table = Table(title="Evaluation Summary", show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", justify="right", style="green")
        table.add_column("Std Dev", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Median", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("<0.5", justify="right", style="red")
        table.add_column("0.5-0.7", justify="right", style="yellow")
        table.add_column(">0.7", justify="right", style="green")
        table.add_column("Duration", justify="right", style="magenta")

        for metric_report in report.metric_reports:
            stats = metric_report.statistics

            table.add_row(
                metric_report.metric_name,
                f"{stats.mean:.3f}",
                f"{stats.std_dev:.3f}",
                f"{stats.min:.3f}",
                f"{stats.median:.3f}",
                f"{stats.max:.3f}",
                f"{stats.count_below_50} ({stats.percent_below_50}%)",
                f"{stats.count_50_to_70} ({stats.percent_50_to_70}%)",
                f"{stats.count_above_70} ({stats.percent_above_70}%)",
                f"{metric_report.duration_seconds:.1f}s",
            )

        console.print("\n")
        console.print(table)
        console.print("\n")

    def _print_query_results(self, report: EvaluationReport):
        """Print query-by-query evaluation results."""
        table = Table(
            title="Query-Level Results",
            show_header=True,
            header_style="bold",
            show_lines=True,
        )
        table.add_column("#", justify="right", style="dim")
        table.add_column("Query", style="cyan", max_width=60)

        # Add columns for each metric
        metric_names = [mr.metric_name for mr in report.metric_reports]
        for metric_name in metric_names:
            table.add_column(metric_name, justify="right")

        table.add_column("Overall", justify="right", style="bold green")

        # Add rows for each query
        for result in report.query_level_results:
            # Truncate query if too long
            query_display = result.query
            if len(query_display) > 60:
                query_display = query_display[:57] + "..."

            # Build row with metric scores
            row = [
                str(result.query_index + 1),
                query_display,
            ]

            # Add score for each metric
            for metric_name in metric_names:
                score = result.metric_scores.get(metric_name, 0.0)
                # Color code the score
                if score < 0.5:
                    row.append(f"[red]{score:.3f}[/red]")
                elif score < 0.7:
                    row.append(f"[yellow]{score:.3f}[/yellow]")
                else:
                    row.append(f"[green]{score:.3f}[/green]")

            # Add overall score
            overall = result.overall_score
            if overall < 0.5:
                row.append(f"[red]{overall:.3f}[/red]")
            elif overall < 0.7:
                row.append(f"[yellow]{overall:.3f}[/yellow]")
            else:
                row.append(f"[green]{overall:.3f}[/green]")

            table.add_row(*row)

        console.print("\n")
        console.print(table)
        console.print("\n")
