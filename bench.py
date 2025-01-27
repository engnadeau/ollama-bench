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
        result = {
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
        result["prompt_token_throughput"] = (
            result["num_prompt_tokens"] / result["prompt_duration"]
        )
        result["inference_token_throughput"] = (
            result["num_inference_tokens"] / result["inference_duration"]
        )
        results.append(result)
    return results


def save_results(results: list, filename: str) -> None:
    summary = {}

    # Calculate the summary
    num_prompt_tokens = [result["num_prompt_tokens"] for result in results]
    total_prompt_tokens = sum(num_prompt_tokens)
    min_prompt_tokens = min(num_prompt_tokens)
    max_prompt_tokens = max(num_prompt_tokens)
    avg_prompt_tokens = total_prompt_tokens / len(results)

    num_inference_tokens = [result["num_inference_tokens"] for result in results]
    total_inference_tokens = sum(num_inference_tokens)
    min_inference_tokens = min(num_inference_tokens)
    max_inference_tokens = max(num_inference_tokens)
    avg_inference_tokens = total_inference_tokens / len(results)

    elapsed_times = [result["elapsed_time"] for result in results]
    min_elapsed_time = min(elapsed_times)
    max_elapsed_time = max(elapsed_times)
    avg_elapsed_time = sum(elapsed_times) / len(results)

    inference_durations = [result["inference_duration"] for result in results]
    min_inference_duration = min(inference_durations)
    max_inference_duration = max(inference_durations)
    avg_inference_duration = sum(inference_durations) / len(results)

    prompt_durations = [result["prompt_duration"] for result in results]
    min_prompt_duration = min(prompt_durations)
    max_prompt_duration = max(prompt_durations)
    avg_prompt_duration = sum(prompt_durations) / len(results)

    load_durations = [result["load_duration"] for result in results]
    min_load_duration = min(load_durations)
    max_load_duration = max(load_durations)
    avg_load_duration = sum(load_durations) / len(results)

    total_durations = [result["total_duration"] for result in results]
    min_total_duration = min(total_durations)
    max_total_duration = max(total_durations)
    avg_total_duration = sum(total_durations) / len(results)

    prompt_token_throughputs = [result["prompt_token_throughput"] for result in results]
    min_prompt_token_throughput = min(prompt_token_throughputs)
    max_prompt_token_throughput = max(prompt_token_throughputs)
    avg_prompt_token_throughput = sum(prompt_token_throughputs) / len(results)

    inference_token_throughputs = [
        result["inference_token_throughput"] for result in results
    ]
    min_inference_token_throughput = min(inference_token_throughputs)
    max_inference_token_throughput = max(inference_token_throughputs)
    avg_inference_token_throughput = sum(inference_token_throughputs) / len(results)

    # Log the summary

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
