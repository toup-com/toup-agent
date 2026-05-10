"""Third-party connectors package.

Each subdirectory is one connector with two files:
  manifest.yaml   - declarative config (scopes, tools, channel policy, ...)
  provider.py     - Python module with a BaseConnectorProvider subclass

The connector registry (`app.services.connector_registry`) discovers,
validates, and indexes all subdirectories at platform boot.
"""
