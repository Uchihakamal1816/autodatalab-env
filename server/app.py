"""Root FastAPI app shim for validators expecting `server.app:app`.

This module re-exports the real OpenEnv FastAPI application implemented under
`data_cleaning_env.server.app`.
"""

from data_cleaning_env.server.app import app, main

__all__ = ["app", "main"]
