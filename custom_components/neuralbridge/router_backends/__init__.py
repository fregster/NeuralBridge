"""Router backends for NeuralBridge — re-exports convenience package.

Import the concrete back-end classes directly from this package:

    from .router_backends import OllamaRouterBackend, IntegratedRouterBackend
"""

from __future__ import annotations

from .integrated_router_backend import IntegratedRouterBackend
from .ollama_router_backend import OllamaRouterBackend

__all__ = ["IntegratedRouterBackend", "OllamaRouterBackend"]
