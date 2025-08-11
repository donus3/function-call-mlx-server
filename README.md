# Fucntion Calling MLX Server

A lightweight HTTP server for running Qwen and GPT_OSS language models with MLX (Metal for Mac) acceleration. This server provides an OpenAI-compatible API interface for serving Qwen and OSS models locally.

## Features

- **OpenAI-compatible API**: Supports `/v1/chat/completions` and `/v1/completions` endpoints
- **MLX acceleration**: Leverages Metal for Mac (MLX) for fast inference on Apple Silicon
- **Speculative decoding**: Supports draft models for faster generation
- **Prompt caching**: Efficiently reuses common prompt prefixes
- **Tool calling support**: Native support for function calling with custom formats
- **Streaming responses**: Real-time token streaming support
- **Model adapters**: Support for fine-tuned model adapters

## Installation

```bash
# Install the package
uv sync
```

## Usage

Start the server with a Qwen model:

```bash
# Basic usage
uv run main.py --type qwen --model <path-to-qwen-model>

# With custom host and port
uv run main.py --type qwen --host 0.0.0.0 --port 8080 --model <path-to-qwen-model>

# With draft model for speculative decoding
uv run main.py --type qwen --model <path-to-qwen-model> --draft-model <path-to-draft-model>
```

## API Endpoints

### Chat Completions
```bash
POST /v1/chat/completions
```

Example request:
```json
{
  "model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false
}
```

### Text Completions
```bash
POST /v1/completions
```

Example request:
```json
{
  "model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ",
  "prompt": "Hello, my name is",
  "stream": false
}
```

### Health Check
```bash
GET /health
```

### Model List
```bash
GET /v1/models
```

## Configuration Options

- `--model`: Path to the MLX model weights, tokenizer, and config
- `--adapter-path`: Optional path for trained adapter weights and config
- `--host`: Host for the HTTP server (default: 127.0.0.1)
- `--port`: Port for the HTTP server (default: 8080)
- `--draft-model`: Model to be used for speculative decoding
- `--num-draft-tokens`: Number of tokens to draft when using speculative decoding
- `--trust-remote-code`: Enable trusting remote code for tokenizer
- `--log-level`: Set the logging level (default: INFO)
- `--chat-template`: Specify a chat template for the tokenizer
- `--use-default-chat-template`: Use the default chat template
- `--temp`: Default sampling temperature (default: 0.0)
- `--top-p`: Default nucleus sampling top-p (default: 1.0)
- `--top-k`: Default top-k sampling (default: 0, disables top-k)
- `--min-p`: Default min-p sampling (default: 0.0, disables min-p)
- `--max-tokens`: Default maximum number of tokens to generate (default: 512)

## Example Usage

### Using curl to test the server:

```bash
# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-7B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "stream": false
  }'
```

### Streaming response:
```bash
# Streaming chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write a short story about a robot learning to paint."}
    ],
    "stream": true
  }'
```

### Using with sst/opencode

Example configuration
```json
{
  "$schema": "https://opencode.ai/config.json",
  "share": "disabled",
  "provider": {
    "mlx-lm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "mlx-lm (local)",
      "options": {
        "baseURL": "http://127.0.0.1:28100/v1"
      },
      "models": {
        "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ": {
          "name": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ",
          "options": {
            "max_tokens": 128000,
          },
          "tools": true
        }
      }
    }
  }
}
```

## Development

### Running Tests

```bash
# Run the server in development mode
uv run main.py --model <path-to-model>
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project is built on top of:
- [MLX](https://github.com/ml-explore/mlx) - Metal for Mac
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - MLX Language Model Inference
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
