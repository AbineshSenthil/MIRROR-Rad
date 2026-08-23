"""MIRROR-Rad evidence-grounded radiology summary integrity auditing."""

__version__ = "1.0.0"

try:
    from .verifier import RadTraceVerifier
except ImportError:  # Allows data-only utilities before optional ML dependencies are installed.
    RadTraceVerifier = None  # type: ignore[assignment]

__all__ = ["RadTraceVerifier", "__version__"]
