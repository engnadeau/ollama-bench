import json
from loguru import logger
import fire
from ollama import OllamaClient


def benchmark(prompt: str, model_name: str, num_repetitions: int = 1) -> list:
    model = OllamaClient(model_name)
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


def save_results(results: list, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)


def run_benchmark(
    prompt: str,
    model_name: str,
    num_repetitions: int = 3,
    output_file: str = "benchmark_results.json",
) -> None:
    # Validate the input
    if not model_name:
        logger.error(
            "Model name is required. Please visit https://ollama.com/search to find the models you want to bench."
        )
        raise ValueError("Model name cannot be empty")

    # Log the benchmark configuration
    logger.info(f"Running benchmark for model: {model_name}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Number of repetitions: {num_repetitions}")

    # Run the benchmark
    results = benchmark(prompt, model_name, num_repetitions)
    save_results(results, output_file)
    logger.info(f"Benchmark results saved to {output_file}")
    logger.info("Benchmark completed successfully")


if __name__ == "__main__":
    fire.Fire(run_benchmark)
