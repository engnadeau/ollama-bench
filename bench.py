import json
from loguru import logger
import fire
from ollama import OllamaClient


def benchmark(prompt, num_repetitions=1):
    model = OllamaClient("default_model")
    results = []
    for _ in range(num_repetitions):
        start_time = logger.time()
        response = model.query(prompt)
        end_time = logger.time()
        elapsed_time = end_time - start_time
        results.append(
            {"prompt": prompt, "response": response, "elapsed_time": elapsed_time}
        )
    return results


def save_results(results, filename="benchmark_results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)


def run_benchmark(prompt, num_repetitions=1, output_file="benchmark_results.json"):
    logger.info(f"Running benchmark for model: default_model")
    try:
        results = benchmark(prompt, num_repetitions)
        save_results(results, output_file)
        logger.info(f"Benchmark results saved to {output_file}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    fire.Fire(run_benchmark)
