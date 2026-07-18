"""Command-line entry point for the FastAPI Web server."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Coder Agent Web API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    try:
        import uvicorn
    except ModuleNotFoundError:
        print("FastAPI server dependencies are missing.")
        print("Install them with: pip install fastapi uvicorn")
        return 1

    if args.reload:
        uvicorn.run(
            "server.app:app",
            host=args.host,
            port=args.port,
            reload=True,
        )
        return 0

    from server.app import app
    from server.service import format_token_usage_summary

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=False,
        )
    finally:
        service = getattr(getattr(app, "state", None), "agent_service", None)
        if service is not None and hasattr(service, "get_token_usage_summary"):
            print(format_token_usage_summary(service.get_token_usage_summary()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
