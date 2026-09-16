# -*- coding: utf-8 -*-
"""
Convenience entry point: starts the IEC RAG chat server with sensible
defaults, so day-to-day use is just:

    python run.py

Equivalent to:

    python api_server.py --db-path ./chroma_db --port 5001

Override any default with the same flags api_server.py accepts.
"""

import argparse

import api_server


def main():
    parser = argparse.ArgumentParser(description="Run the IEC Arabic RAG chat server.")
    parser.add_argument("--db-path", default="./chroma_db")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    api_server.init_app(args.db_path, k=args.k)
    print(f"\nOpen http://{args.host}:{args.port}/ in a browser.\n")
    api_server.app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
