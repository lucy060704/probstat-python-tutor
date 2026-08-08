"""Platform-independent ASGI contract for later HTTPS deployment."""

from probstat_tutor.api.app import create_api_app

__all__ = ["create_api_app"]
