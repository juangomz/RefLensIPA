"""Evaluation metrics for RAG systems."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from src.rag.evaluation.llm_judge import LLMJudge


class EvaluationResult(BaseModel):
    """Result of a single metric evaluation."""

    metric_name: str
    score: float
    reasoning: str


class Metric(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> EvaluationResult:
        """
        Evaluate a RAG response.

        Args:
            query: User query
            answer: Generated answer
            contexts: Retrieved context documents

        Returns:
            Evaluation result with score and reasoning
        """
        pass


# Faithfulness metric models


class ClaimVerdict(BaseModel):
    """A claim with its verification verdict."""

    claim: str = Field(description="The factual claim from the answer")
    is_supported: bool = Field(
        description="Whether this claim is supported by the contexts"
    )


class FaithfulnessResponse(BaseModel):
    """Response format for faithfulness evaluation."""

    reasoning: str = Field(description="Explanation of the evaluation")
    claims: list[ClaimVerdict] = Field(
        description="List of claims with their verification verdicts"
    )


class FaithfulnessMetric(Metric):
    """
    Evaluates whether the answer is faithful to the retrieved contexts.

    Checks if all claims in the answer can be verified by the contexts,
    ensuring no hallucinations or unsupported statements.

    This metric:
    1. Extracts claims from the answer using LLM
    2. Verifies each claim against the contexts
    3. Calculates the ratio of supported claims

    Score: (number of supported claims) / (total number of claims)
    """

    def __init__(self, llm_judge: LLMJudge):
        """
        Initialize faithfulness metric.

        Args:
            llm_judge: LLM judge instance for evaluation
        """
        self.llm_judge = llm_judge

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> EvaluationResult:
        """
        Evaluate faithfulness of answer to contexts.

        Args:
            query: User query
            answer: Generated answer
            contexts: Retrieved context documents

        Returns:
            Evaluation result with faithfulness score (0-1)
        """
        system_prompt = """You are an expert evaluator assessing the faithfulness of answers to their source contexts.

Your task is to:
1. Extract all factual claims made in the answer
2. For each claim, determine if it can be verified by the provided contexts
3. A claim is supported ONLY if it can be directly inferred from the contexts
4. Claims that contradict the contexts or add new information should be marked as not supported

Be strict in your evaluation - if a claim cannot be clearly verified, mark it as not supported."""

        contexts_text = "\n\n".join(
            [f"<context id=\"{i+1}\">\n{ctx}\n</context>" for i, ctx in enumerate(contexts)]
        )

        user_prompt = f"""<question>
{query}
</question>

<answer>
{answer}
</answer>

<retrieved_contexts>
{contexts_text}
</retrieved_contexts>

Extract all claims from the answer and verify each against the contexts.
For each claim, return:
- claim: the text of the claim
- is_supported: boolean indicating if the claim is supported by the contexts

"""

        response = self.llm_judge.evaluate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=FaithfulnessResponse,
        )

        # Calculate faithfulness score
        if len(response.claims) == 0:
            score = 0.0
        else:
            score = sum(1 for c in response.claims if c.is_supported) / len(
                response.claims
            )

        return EvaluationResult(
            metric_name="faithfulness",
            score=score,
            reasoning=response.reasoning,
        )


# Answer relevance metric models


class RelevanceResponse(BaseModel):
    """Response format for relevance evaluation."""

    reasoning: str = Field(description="Explanation of the score")
    score: int = Field(
        ge=1, le=5,
        description="Relevance score from 1 (not relevant) to 5 (highly relevant)"
    )


class AnswerRelevanceMetric(Metric):
    """
    Evaluates whether the answer is relevant to the query.

    Checks if the answer actually addresses the question asked,
    regardless of whether it's correct or complete.

    This metric uses an LLM judge to assess how well the answer
    addresses the specific question in the query.

    Score: Normalized from 1-5 scale to 0-1 scale
    """

    def __init__(self, llm_judge: LLMJudge):
        """
        Initialize answer relevance metric.

        Args:
            llm_judge: LLM judge instance for evaluation
        """
        self.llm_judge = llm_judge

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> EvaluationResult:
        """
        Evaluate relevance of answer to query.

        Args:
            query: User query
            answer: Generated answer
            contexts: Not used for this metric

        Returns:
            Evaluation result with relevance score (0-1)
        """
        system_prompt = """You are an expert evaluator assessing whether answers are relevant to the questions asked.

Your task is to evaluate how well the answer addresses the specific question, regardless of correctness.

Scoring criteria:
- 5: Answer directly and completely addresses the question
- 4: Answer addresses the question but may miss some aspects
- 3: Answer partially addresses the question
- 2: Answer is tangentially related to the question
- 1: Answer is not relevant to the question

Focus on relevance, not accuracy or completeness."""

        user_prompt = f"""<question>
{query}
</question>

<answer>
{answer}
</answer>

Rate the relevance of this answer to the question on a scale from 1 to 5.
"""

        response = self.llm_judge.evaluate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=RelevanceResponse,
        )

        # Normalize score from 1-5 to 0-1
        score = (response.score - 1) / 4

        return EvaluationResult(
            metric_name="answer_relevance",
            score=score,
            reasoning=response.reasoning,
        )


# Context relevance metric models


class ContextRelevanceResponse(BaseModel):
    """Response format for context relevance evaluation."""

    reasoning: str = Field(description="Explanation of the evaluation")
    relevance_scores: list[int] = Field(
        description="For each context, a score from 1 (not relevant) to 5 (highly relevant)"
    )


class ContextRelevanceMetric(Metric):
    """
    Evaluates whether the retrieved contexts are relevant to the query.

    Measures the quality of retrieval by checking if the retrieved
    documents contain information useful for answering the query.

    This metric uses an LLM judge to assess each context's relevance
    and calculate the average relevance score.

    Score: Average relevance across all contexts, normalized to 0-1
    """

    def __init__(self, llm_judge: LLMJudge):
        """
        Initialize context relevance metric.

        Args:
            llm_judge: LLM judge instance for evaluation
        """
        self.llm_judge = llm_judge

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> EvaluationResult:
        """
        Evaluate relevance of contexts to query.

        Args:
            query: User query
            answer: Not used for this metric
            contexts: Retrieved context documents

        Returns:
            Evaluation result with context relevance score (0-1)
        """
        system_prompt = """You are an expert evaluator assessing whether retrieved contexts are relevant to answering a question.

Your task is to evaluate each context independently and rate how useful it would be for answering the given question.

Scoring criteria for each context:
- 5: Highly relevant - directly contains information needed to answer the question
- 4: Relevant - contains useful information related to the question
- 3: Somewhat relevant - tangentially related to the question
- 2: Barely relevant - mentions related concepts but not useful for answering
- 1: Not relevant - unrelated to the question

Evaluate each context independently."""

        contexts_text = "\n\n".join(
            [f"<context id=\"{i+1}\">\n{ctx}\n</context>" for i, ctx in enumerate(contexts)]
        )

        user_prompt = f"""<question>
{query}
</question>

<retrieved_contexts>
{contexts_text}
</retrieved_contexts>

Rate the relevance of each context to answering this question on a scale from 1 to 5.
Provide one score per context in order.
"""

        response = self.llm_judge.evaluate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=ContextRelevanceResponse,
        )

        # Calculate average relevance score and normalize to 0-1
        if len(response.relevance_scores) == 0:
            score = 0.0
        else:
            avg_score = sum(response.relevance_scores) / len(response.relevance_scores)
            score = (avg_score - 1) / 4  # Normalize from 1-5 to 0-1

        return EvaluationResult(
            metric_name="context_relevance",
            score=score,
            reasoning=response.reasoning,
        )
