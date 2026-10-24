"""Command-line document ingestion entry point."""

import argparse
import json
from pathlib import Path

from src.ingestion.chunking import chunk_directory, chunk_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and chunk documents.")
    parser.add_argument("path", type=Path, help="A supported document or directory")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()
    chunks = (
        chunk_directory(args.path, args.chunk_size, args.overlap)
        if args.path.is_dir()
        else chunk_file(args.path, args.chunk_size, args.overlap)
    )
    print(json.dumps([chunk.model_dump() for chunk in chunks], indent=2))


if __name__ == "__main__":
    main()
