"""Snowflake client wrapper.

This mirrors the role of DataHubClient in datahub-agent-context.
DataHub wraps DataHubGraph; we wrap snowflake.connector.Connection.

The client is the thing stored in contextvars — tools never create
connections themselves, they retrieve the client from context.

Authentication methods (in priority order):
1. Key pair — SNOWFLAKE_PRIVATE_KEY_PATH env var, no browser pop-ups
2. External browser — SNOWFLAKE_AUTHENTICATOR=externalbrowser, opens Okta SSO
3. Password — SNOWFLAKE_PASSWORD env var (least common)

For SoFi/Galileo, externalbrowser is the default. Key pair is preferred
for automation and avoiding repeated browser prompts (see dbt dev_key_pair target).
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

        # Auth priority: key pair > externalbrowser > password
        #
        # Key pair: no browser, no password. Best for automation.
        #   Generate RSA key per Snowflake docs, register via stored proc.
        #   Set SNOWFLAKE_PRIVATE_KEY_PATH=/path/to/rsa_key.p8
        #
        # External browser: opens Okta SSO in browser. Default for SoFi.
        #   Set SNOWFLAKE_AUTHENTICATOR=externalbrowser
        #   Each new connection opens a browser tab for login.
        #
        # Password: basic auth. Least common at SoFi.
        #   Set SNOWFLAKE_PASSWORD=...
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
            auth_method = "key_pair"
        elif authenticator:
            connect_kwargs["authenticator"] = authenticator
            auth_method = authenticator
        elif password:
            connect_kwargs["password"] = password
            auth_method = "password"
        else:
            # Default to externalbrowser if no auth method specified.
            # This matches SoFi's standard Snowflake access pattern
            # (Okta SSO via browser).
            connect_kwargs["authenticator"] = "externalbrowser"
            auth_method = "externalbrowser (default)"

        logger.info(
            "Connecting to Snowflake: %s.%s.%s (auth: %s)",
            account, database, schema, auth_method,
        )
        self._conn = snowflake.connector.connect(**connect_kwargs)
        logger.info("Connected successfully")

    @classmethod
    def from_env(cls) -> "SnowflakeClient":
        """Create a client from environment variables.

        Same pattern as DataHubClient.from_env() — lets you configure
        the connection via env vars without touching code.

        Required env vars:
            SNOWFLAKE_ACCOUNT    — e.g. gna62195.us-east-1 (ST) or mha08645.us-east-1 (PROD)
            SNOWFLAKE_USER       — your Snowflake username (e.g. sma)
            SNOWFLAKE_WAREHOUSE  — compute warehouse name
            SNOWFLAKE_DATABASE   — database to connect to
            SNOWFLAKE_SCHEMA     — schema to connect to

        Optional env vars:
            SNOWFLAKE_ROLE              — Snowflake role (e.g. SNOWFLAKE_GALILEO_PROD_DATASCIENCE)
            SNOWFLAKE_AUTHENTICATOR     — "externalbrowser" for Okta SSO (default if no auth specified)
            SNOWFLAKE_PRIVATE_KEY_PATH  — path to RSA private key (.p8) for key pair auth
            SNOWFLAKE_PASSWORD          — password (least preferred)

        Auth auto-detection:
            1. If SNOWFLAKE_PRIVATE_KEY_PATH is set → key pair (no browser)
            2. If SNOWFLAKE_AUTHENTICATOR is set → use that (e.g. externalbrowser)
            3. If SNOWFLAKE_PASSWORD is set → password auth
            4. If none set → defaults to externalbrowser (opens Okta in browser)

        Example .zshrc setup (externalbrowser):
            export SNOWFLAKE_ACCOUNT=gna62195.us-east-1
            export SNOWFLAKE_USER=sma
            export SNOWFLAKE_WAREHOUSE=GALILEO_WH
            export SNOWFLAKE_DATABASE=AI_DATAMART
            export SNOWFLAKE_SCHEMA=PUBLIC
            export SNOWFLAKE_ROLE=SNOWFLAKE_GALILEO_ST_DATASCIENCE

        Example .zshrc setup (key pair — no browser pop-ups):
            export SNOWFLAKE_ACCOUNT=gna62195.us-east-1
            export SNOWFLAKE_USER=sma
            export SNOWFLAKE_WAREHOUSE=GALILEO_WH
            export SNOWFLAKE_DATABASE=AI_DATAMART
            export SNOWFLAKE_SCHEMA=PUBLIC
            export SNOWFLAKE_ROLE=SNOWFLAKE_GALILEO_ST_DATASCIENCE
            export SNOWFLAKE_PRIVATE_KEY_PATH=~/.ssh/snowflake_rsa_key.p8
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
