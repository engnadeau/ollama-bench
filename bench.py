import json
import time
import fire
from loguru import logger
from ollama import ChatResponse, chat


def load_prompts(filename: str) -> list:
    """Loads prompts from a text file."""
    try:
        with open(filename, "r") as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        raise


def run_chat_api(prompt: str, model: str) -> dict:
    """Executes the chat API and returns the response with timing details."""
    start_time = time.monotonic()
    response: ChatResponse = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    end_time = time.monotonic()

    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time for prompt '{prompt}': {elapsed_time:.2f} seconds")

    return {
        "elapsed_time": elapsed_time,
        "num_inference_tokens": response.eval_count,
        "inference_duration": response.eval_duration / 1e9,
        "load_duration": response.load_duration / 1e9,
        "model": response.model,
        "num_prompt_tokens": response.prompt_eval_count,
        "prompt_duration": response.prompt_eval_duration / 1e9,
        "prompt": prompt,
        "response": response.message.content,
        "total_duration": response.total_duration / 1e9,
    }


def calculate_token_throughput(results: dict) -> dict:
    """Adds token throughput metrics to the results."""
    results["prompt_token_throughput"] = (
        results["num_prompt_tokens"] / results["prompt_duration"]
        if results["prompt_duration"] > 0
        else 0
    )
    results["inference_token_throughput"] = (
        results["num_inference_tokens"] / results["inference_duration"]
        if results["inference_duration"] > 0
        else 0
    )
    return results


def benchmark(prompts: list, model: str) -> list:
    """Runs the benchmark for all prompts."""
    results = []
    for prompt in prompts:
        result = run_chat_api(prompt, model)
        results.append(calculate_token_throughput(result))
    return results


def calculate_summary(results: list) -> dict:
    """Generates summary statistics for the benchmark results."""
    metrics = [
        "num_prompt_tokens",
        "num_inference_tokens",
        "elapsed_time",
        "inference_duration",
        "prompt_duration",
        "load_duration",
        "total_duration",
        "prompt_token_throughput",
        "inference_token_throughput",
    ]

    summary = {}
    for metric in metrics:
        values = [result[metric] for result in results]
        summary[f"total_{metric}"] = sum(values)
        summary[f"avg_{metric}"] = sum(values) / len(values) if values else 0
        summary[f"min_{metric}"] = min(values, default=0)
        summary[f"max_{metric}"] = max(values, default=0)

    return summary


def save_results(results: list, filename: str) -> None:
    """Saves the benchmark results and summary to a JSON file."""
    summary = calculate_summary(results)
    final_results = {"results": results, "summary": summary}
    with open(filename, "w") as f:
        json.dump(final_results, f, indent=4, sort_keys=True)
    logger.info(f"Results saved to {filename}")


def run_benchmark(
    model: str,
    prompts_file: str = "prompts.txt",
    output_file: str = "benchmark_results.json",
) -> None:
    """Runs the benchmark and saves the results."""
    if not model:
        logger.error(
            "Model name is required. See https://ollama.com/search for available models."
        )
        raise ValueError("Model name cannot be empty")

    logger.info(f"Loading prompts from: {prompts_file}")
    prompts = load_prompts(prompts_file)
    logger.info(f"Loaded {len(prompts)} prompts.")

    logger.info(f"Running benchmark for model: {model}")
    results = benchmark(prompts, model)
    save_results(results, output_file)
    logger.info("Benchmark completed successfully")


if __name__ == "__main__":
    logger.add("logs/{time}.log")
    fire.Fire(run_benchmark)
