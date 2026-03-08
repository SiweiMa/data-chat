"""Snowflake client wrapper.

This mirrors the role of DataHubClient in datahub-agent-context.
DataHub wraps DataHubGraph; we wrap snowflake.connector.Connection.

The client is the thing stored in contextvars — tools never create
connections themselves, they retrieve the client from context.
"""

import logging
import os
from typing import Optional

import snowflake.connector

logger = logging.getLogger(__name__)


class SnowflakeClient:
    """Thin wrapper around a Snowflake connection.

    Why a wrapper instead of using Connection directly?
    - Encapsulates connection creation logic (env vars, auth methods)
    - Provides a `from_env()` factory (same pattern as DataHubClient.from_env())
    - Single place to add connection pooling or retries later
    """

    def __init__(
        self,
        account: str,
        user: str,
        warehouse: str,
        database: str,
        schema: str,
        role: Optional[str] = None,
        authenticator: Optional[str] = None,
        private_key_path: Optional[str] = None,
        password: Optional[str] = None,
    ):
        connect_kwargs: dict = {
            "account": account,
            "user": user,
            "warehouse": warehouse,
            "database": database,
            "schema": schema,
        }
        if role:
            connect_kwargs["role"] = role

        # Auth: key pair > externalbrowser > password
        if private_key_path:
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import serialization

            with open(private_key_path, "rb") as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(), password=None, backend=default_backend()
                )
            connect_kwargs["private_key"] = private_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        elif authenticator:
            connect_kwargs["authenticator"] = authenticator
        elif password:
            connect_kwargs["password"] = password

        self._conn = snowflake.connector.connect(**connect_kwargs)
        logger.info("Connected to Snowflake: %s.%s.%s", account, database, schema)

    @classmethod
    def from_env(cls) -> "SnowflakeClient":
        """Create a client from environment variables.

        Same pattern as DataHubClient.from_env() — lets you configure
        the connection via env vars without touching code.

        Required env vars:
            SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_WAREHOUSE,
            SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA

        Optional env vars:
            SNOWFLAKE_ROLE, SNOWFLAKE_AUTHENTICATOR, SNOWFLAKE_PRIVATE_KEY_PATH,
            SNOWFLAKE_PASSWORD
        """
        return cls(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
            database=os.environ["SNOWFLAKE_DATABASE"],
            schema=os.environ["SNOWFLAKE_SCHEMA"],
            role=os.environ.get("SNOWFLAKE_ROLE"),
            authenticator=os.environ.get("SNOWFLAKE_AUTHENTICATOR"),
            private_key_path=os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH"),
            password=os.environ.get("SNOWFLAKE_PASSWORD"),
        )

    @property
    def connection(self) -> snowflake.connector.SnowflakeConnection:
        """The underlying Snowflake connection.

        Tools call get_connection() (from context.py), which returns this.
        Mirrors: DataHubClient._graph (DataHubGraph instance).
        """
        return self._conn

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()
