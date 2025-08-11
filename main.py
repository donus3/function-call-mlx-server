#!/usr/bin/env python3

import argparse
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'oss'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qwen'))

from oss.main import main as oss_main
from qwen.main import main as qwen_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLX Http Server.")
    parser.add_argument(
        "--type",
        type=str,
        help="qwen or oss",
    )
    args, unknown = parser.parse_known_args()
    if args.type == "qwen":
        qwen_main(unknown)
    elif args.type == "oss":
        oss_main(unknown)
