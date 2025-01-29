import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import fire
from loguru import logger
from ollama import ChatResponse, chat


@dataclass
class BenchmarkResult:
    """Data class to represent the results of a single benchmark run."""

    elapsed_time: float
    num_inference_tokens: int
    inference_duration: float
    load_duration: float
    model: str
    num_prompt_tokens: int
    prompt_duration: float
    prompt: str
    response: str
    total_duration: float
    prompt_token_throughput: float
    inference_token_throughput: float


def load_prompts(filename: str) -> list:
    """Loads prompts from a text file."""
    try:
        with open(filename, "r") as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        raise


def run_chat_api(prompt: str, model: str) -> BenchmarkResult:
    """Executes the chat API and returns the response with timing details."""
    start_time = time.monotonic()
    response: ChatResponse = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    end_time = time.monotonic()

    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time for prompt '{prompt}': {elapsed_time:.2f} seconds")

    return BenchmarkResult(
        elapsed_time=elapsed_time,
        num_inference_tokens=response.eval_count,
        inference_duration=response.eval_duration / 1e9,
        load_duration=response.load_duration / 1e9,
        model=response.model,
        num_prompt_tokens=response.prompt_eval_count,
        prompt_duration=response.prompt_eval_duration / 1e9,
        prompt=prompt,
        response=response.message.content,
        total_duration=response.total_duration / 1e9,
        prompt_token_throughput=(
            response.prompt_eval_count / response.prompt_eval_duration * 1e9
            if response.prompt_eval_duration > 0
            else 0
        ),
        inference_token_throughput=(
            response.eval_count / response.eval_duration * 1e9
            if response.eval_duration > 0
            else 0
        ),
    )


def benchmark(prompts: List[str], model: str) -> List[BenchmarkResult]:
    """Runs the benchmark for all prompts."""
    # Warm up the model with a throwaway prompt
    logger.info("Warming up the model...")
    run_chat_api("This is a throwaway prompt to warm up the model.", model)
    logger.info("Model warmed up.")

    results = []
    num_prompts = len(prompts)
    for i, prompt in enumerate(prompts):
        logger.info(f"Running prompt {i + 1}/{num_prompts}: {prompt}")
        result = run_chat_api(prompt, model)
        results.append(result)
    return results


def calculate_summary(results: List[BenchmarkResult]) -> dict:
    """Generates summary statistics for the benchmark results."""
    metrics = [
        field.name
        for field in BenchmarkResult.__dataclass_fields__.values()
        if field.type is not str
    ]

    summary = {}
    for metric in metrics:
        values = [getattr(result, metric) for result in results]
        summary[f"total_{metric}"] = sum(values) if values else 0
        summary[f"avg_{metric}"] = sum(values) / len(values) if values else 0
        summary[f"min_{metric}"] = min(values) if values else 0
        summary[f"max_{metric}"] = max(values) if values else 0

    return summary


def save_results(results: dict, model: str) -> None:
    """Saves the benchmark results and summary to a JSON file in a results folder."""

    # Create the results folder if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Generate a timestamped filename with the model name
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = results_dir / f"benchmark_{model}_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)
    logger.info(f"Results saved to {filename}")


class BenchmarkRunner:
    """
    Main class to manage the benchmark process.  Loads prompts and runs the benchmark for a given model.
    """

    def __init__(self, prompts_file: str = "prompts.txt"):
        self.prompts = load_prompts(prompts_file)

    def run(self, model: str) -> None:
        """Runs the benchmark and saves the results."""
        if not model:
            logger.error(
                "Model name is required. See https://ollama.com/search for available models."
            )
            raise ValueError("Model name cannot be empty")

        logger.info(f"Loaded {len(self.prompts)} prompts.")

        logger.info(f"Running benchmark for model: {model}")
        results = benchmark(self.prompts, model)
        logger.info("Benchmark completed.")

        logger.info("Calculating summary statistics...")
        summary = calculate_summary(results)

        # Only log important summary metrics
        logger.info(f"Total elapsed time: {summary['total_elapsed_time']:.2f} seconds")
        logger.info(
            f"Average prompt token throughput: {summary['avg_prompt_token_throughput']:.2f} tokens/second"
        )
        logger.info(
            f"Average inference token throughput: {summary['avg_inference_token_throughput']:.2f} tokens/second"
        )

        logger.info("Saving results...")
        final_results = {"results": [asdict(r) for r in results], "summary": summary}
        save_results(final_results, model)

        logger.info("Benchmark completed successfully")


def main(
    model: str,
    prompts_file: str = "prompts.txt",
) -> None:
    """Entry point for the benchmark script."""
    try:
        runner = BenchmarkRunner(prompts_file)
        runner.run(model)
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    logger.add("logs/{time}.log")
    fire.Fire(main)
