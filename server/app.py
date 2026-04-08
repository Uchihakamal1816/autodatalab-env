"""Root FastAPI app shim for validators expecting `server.app:app`.

This module exposes the real OpenEnv FastAPI application implemented under
`data_cleaning_env.server.app`, while also providing a callable `main()` and a
direct-execution entrypoint for validator compatibility.
"""

from data_cleaning_env.server.app import app as app
from data_cleaning_env.server.app import main as _real_main


def main(host: str = "0.0.0.0", port: int = 7860):
    """Validator-friendly entrypoint that delegates to the real app launcher."""
    return _real_main(host=host, port=port)


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
