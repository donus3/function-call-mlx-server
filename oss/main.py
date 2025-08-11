#!/usr/bin/env python3

from http.server import HTTPServer
from typing_extensions import override
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
import argparse
import json
import logging
import os
import socket
import sys
import threading

from .cache import PromptCache
from .model import ModelProvider
from .api_handler import APIHandler


class ChangeHandler(FileSystemEventHandler):
    @override
    def on_any_event(self, event: FileSystemEvent):
        if event.is_directory:
            return
        print(f"Detected change in {event.src_path}, restarting...")
        os.execv(sys.executable, [sys.executable] + sys.argv)


def run(
    host: str,
    port: int,
    model_provider: ModelProvider,
    server_class: type[HTTPServer]=HTTPServer,
    handler_class: type[APIHandler]=APIHandler,
):
    server_address = (host, port)
    prompt_cache = PromptCache()
    infos = socket.getaddrinfo(
        *server_address, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )
    server_class.address_family, _, _, _, server_address = next(iter(infos))
    httpd = server_class(
        server_address,
        lambda *args, **kwargs: handler_class(
            model_provider,
            prompt_cache=prompt_cache,
            *args,
            **kwargs,
        ),
    )
    logging.info(f"Starting httpd at {host} on port {port}...")
    httpd.serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser(description="MLX Http Server.")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the MLX model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        help="A model to be used for speculative decoding.",
        default=None,
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative decoding.",
        default=3,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help="Specify a chat template for the tokenizer",
        required=False,
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Default sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Default nucleus sampling top-p (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Default top-k sampling (default: 0, disables top-k)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Default min-p sampling (default: 0.0, disables min-p)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Default maximum number of tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        help="""A JSON formatted string of arguments for the tokenizer's apply_chat_template, e.g. '{"enable_thinking":false}'""",
        default="{}",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    threading.Thread(target=run, args=(args.host, args.port,
                     ModelProvider(args)), daemon=True).start()
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=True)
    observer.start()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
