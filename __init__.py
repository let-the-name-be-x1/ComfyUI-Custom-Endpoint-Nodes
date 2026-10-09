"""ComfyUI entry point: node registrations, the web UI extension, the guard route."""

if __package__:
    from .ce_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .custom_endpoint.routes import register_routes
else:  # loaded as a plain module (tests, tooling): guarantee our own files win
    import os as _os
    import sys as _sys

    _ROOT = _os.path.dirname(_os.path.abspath(__file__))
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)

    from ce_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from custom_endpoint.routes import register_routes

WEB_DIRECTORY = "./custom_endpoint/web"
register_routes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
