import json
import time

import fire
from loguru import logger
from ollama import ChatResponse, chat


def benchmark(prompt: str, model: str, num_repetitions: int = 1) -> list:
    results = []
    for _ in range(num_repetitions):
        # Run the chat API
        start_time = time.monotonic()
        response: ChatResponse = chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        end_time = time.monotonic()

        # Log the elapsed time
        elapsed_time = end_time - start_time
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

        # Append the results
        # Also convert Ollama response duration nanoseconds to seconds
        results.append(
            {
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
        )
    return results


def save_results(results: list, filename: str) -> None:
    # Calculate the summary
    times = [result["elapsed_time"] for result in results]
    summary = {
        "num_repetitions": len(results),
        "min_time": min(times),
        "max_time": max(times),
        "mean_time": sum(times) / len(times),
    }

    # Log the summary
    logger.info(f"Number of repetitions: {summary['num_repetitions']}")
    logger.info(f"Min time: {summary['min_time']:.2f} seconds")
    logger.info(f"Max time: {summary['max_time']:.2f} seconds")
    logger.info(f"Mean time: {summary['mean_time']:.2f} seconds")

    # Save the results
    final_results = {"results": results, "summary": summary}
    with open(filename, "w") as f:
        json.dump(final_results, f, indent=4, sort_keys=True)


def run_benchmark(
    prompt: str,
    model: str,
    num_repetitions: int = 3,
    output_file: str = "benchmark_results.json",
) -> None:
    # Validate the input
    if not model:
        logger.error(
            "Model name is required. Please visit https://ollama.com/search to find the models you want to bench."
        )
        raise ValueError("Model name cannot be empty")

    # Log the benchmark configuration
    logger.info(f"Running benchmark for model: {model}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Number of repetitions: {num_repetitions}")

    # Run the benchmark
    results = benchmark(prompt, model, num_repetitions)

    # Save the results
    save_results(results, output_file)
    logger.info(f"Benchmark results saved to {output_file}")
    logger.info("Benchmark completed successfully")


if __name__ == "__main__":
    logger.add("logs/{time}.log")
    fire.Fire(run_benchmark)
