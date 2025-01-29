# ollama-bench

This project benchmarks the performance of different LLMs using the Ollama API. It measures various metrics, including inference time, token throughput, and resource utilization. Results are saved to a JSON file for easy analysis.

## Setup

1. **Install Dependencies:** This project uses `uv` for dependency and virtual environment management. Install the dependencies:

   ```bash
   uv install
   ```

2. **Install Ollama:** Make sure you have Ollama installed and configured correctly. You'll need to have at least one LLM installed and accessible via the Ollama API. See the [Ollama documentation](https://ollama.com/) for instructions.

## Usage

1. **(Optional) Add Prompts:** The `prompts.txt` file already contains sample prompts. You can add or modify these as needed, one prompt per line.

2. **Run the Benchmark:** Execute the benchmark script using `uv`, specifying the Ollama model name:

   ```bash
   uv run python bench.py --model <model_name>
   ```

   Replace `<model_name>` with your Ollama model name (e.g., `qwen2.5-coder`). The script will:

   * Load prompts from `prompts.txt`.
   * Run a warm-up prompt to prepare the model.
   * Execute each prompt and record performance metrics.
   * Create a `results` directory (if it doesn't exist) and save the results to a JSON file within it. The filename will include a timestamp and the model name.

   **Error Handling:** The script includes comprehensive error handling and logging. Check the logs (`logs/*.log`) for details if any issues occur.

3. **Analyze Results:** The JSON file (e.g., `results/benchmark_gpt-3.5-turbo_2024-10-27_10-30-00.json`) contains detailed results for each prompt and a summary.

## Metrics

The benchmark collects the following metrics:

* `elapsed_time`: Total time to process the chat request from Python (seconds).
* `num_inference_tokens`: Number of tokens generated by the model.
* `inference_duration`: Time spent on inference (seconds).
* `load_duration`: Time spent loading the model (seconds).
* `model`: The Ollama model name used.
* `num_prompt_tokens`: Number of tokens in the prompt.
* `prompt_duration`: Time spent processing the prompt (seconds).
* `prompt`: The prompt text.
* `response`: The model's response.
* `total_duration`: Total time, including loading and processing (seconds).
* `prompt_token_throughput`: Prompt tokens processed per second.
* `inference_token_throughput`: Inference tokens generated per second.

## Development

This project uses a `Makefile` for common development tasks:

* `make format`: Formats the code using `black`, `isort`, and `ruff`.
* `make lint`: Runs code linting using `black`, `isort`, and `ruff`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
