"""LLM-as-a-Judge implementation for RAG evaluation."""

import json

from langfuse.openai import OpenAI
from openai import RateLimitError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.rag.utils import is_reasoning_model


class LLMJudge:
    """LLM judge for evaluating RAG responses."""

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        temperature: float,
    ):
        """
        Initialize LLM judge.

        Args:
            client: LLM client (LangFuse-wrapped for tracing)
            model_name: Model name for evaluation
            temperature: Temperature for generation
        """
        self.client = client
        self.model_name = model_name
        self.temperature = temperature

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def evaluate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type[BaseModel],
    ) -> BaseModel:
        """
        Evaluate using LLM judge with structured output.

        Args:
            system_prompt: System prompt for the judge
            user_prompt: User prompt with evaluation task
            response_format: Pydantic model for structured output

        Returns:
            Structured response as Pydantic model instance
        """
        # Create a sample/template of the expected output using the full schema
        schema_dict = response_format.model_json_schema()
        example = json.dumps(schema_dict, indent=2)

        system_message = (
            f"{system_prompt}\n\n"
            "Return ONLY a valid JSON object. Do not include any extra text, "
            "explanations, or markdown formatting.\n\n"
            "Expected format:\n"
            f"{example}"
        )

        # Reasoning models (o1, o3, gpt-5) don't support temperature parameter
        call_params = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
        }

        if not is_reasoning_model(self.model_name):
            call_params["temperature"] = self.temperature

        response = self.client.chat.completions.create(**call_params)
        content = response.choices[0].message.content or ""

        return self._parse_response(content, response_format)

    @staticmethod
    def _parse_response(
        content: str,
        response_format: type[BaseModel],
    ) -> BaseModel:
        content = content.strip()
        if not content:
            raise ValueError("Empty response from LLM judge")

        try:
            return response_format.model_validate_json(content)
        except Exception:
            start = content.find("{")
            if start == -1:
                raise

            # Try to extract valid JSON by finding matching braces
            brace_count = 0
            for i in range(start, len(content)):
                if content[i] == "{":
                    brace_count += 1
                elif content[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_blob = content[start : i + 1]
                        return response_format.model_validate_json(json_blob)

            # If we can't find matching braces, raise the original error
            raise
