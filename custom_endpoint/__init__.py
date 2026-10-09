"""Custom Endpoint API nodes for ComfyUI.

Provider node owns endpoint + key + model; call nodes call the provider
directly, accept a batch of prompts and run the calls concurrently.
"""

__all__ = ["nodes", "routes"]
