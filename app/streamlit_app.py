"""Streamlit web app for data-chat.

A conversational interface over Snowflake, powered by the shared agent module.

Run with:
    pip install -e '.[streamlit]'
    streamlit run app/streamlit_app.py
"""

import streamlit as st

from data_chat.agent import NoOpCallbacks, run_agent
from data_chat.client import SnowflakeClient
from data_chat.exceptions import (
    LLMAPIError,
    LLMConnectionError,
    LLMRateLimitError,
    SnowflakeConnectionError,
    SnowflakeSessionExpiredError,
)
from data_chat.llm import create_anthropic_client
from data_chat.tools import (
    get_lineage,
    get_tables,
    list_databases,
    list_schemas,
    run_query,
    search,
)

ALL_TOOLS = [list_databases, list_schemas, search, get_tables, get_lineage, run_query]

st.set_page_config(page_title="Data Chat", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat display messages
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = None  # API message history
if "sf_client" not in st.session_state:
    st.session_state.sf_client = None
if "llm_client" not in st.session_state:
    st.session_state.llm_client = None
if "llm_model" not in st.session_state:
    st.session_state.llm_model = None


# ---------------------------------------------------------------------------
# Callbacks for Streamlit UI
# ---------------------------------------------------------------------------


class StreamlitCallbacks(NoOpCallbacks):
    """Updates st.status() widget during agent execution."""

    def __init__(self, status_widget):
        self._status = status_widget
        self._tool_count = 0

    def on_tool_start(self, name: str, input: dict) -> None:
        self._tool_count += 1
        self._status.update(label=f"Calling {name}...", state="running")
        self._status.write(f"**Tool {self._tool_count}:** `{name}`")

    def on_tool_result(self, name: str, result: str) -> None:
        preview = result[:200] + "..." if len(result) > 200 else result
        self._status.write(f"Result: `{preview}`")

    def on_tool_error(self, name: str, error: Exception) -> None:
        self._status.write(f"Error in {name}: {error}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📊 Data Chat")

    # --- Snowflake connection ---
    st.subheader("Snowflake")
    if st.session_state.sf_client:
        st.success("Connected", icon="✅")
        if st.button("Disconnect Snowflake"):
            st.session_state.sf_client.close()
            st.session_state.sf_client = None
            st.rerun()
    else:
        st.info("Not connected")
        if st.button("Connect to Snowflake"):
            with st.spinner("Connecting (browser auth)..."):
                try:
                    st.session_state.sf_client = SnowflakeClient.from_env()
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {e}")

    st.divider()

    # --- LLM connection ---
    st.subheader("LLM")
    if st.session_state.llm_client:
        st.success(f"Model: {st.session_state.llm_model}", icon="🤖")
        if st.button("Disconnect LLM"):
            st.session_state.llm_client = None
            st.session_state.llm_model = None
            st.rerun()
    else:
        st.info("Not connected")
        if st.button("Connect to LLM"):
            with st.spinner("Connecting..."):
                try:
                    client, model = create_anthropic_client()
                    st.session_state.llm_client = client
                    st.session_state.llm_model = model
                    st.rerun()
                except Exception as e:
                    st.error(f"LLM connection failed: {e}")

    st.divider()

    # --- Conversation controls ---
    if st.button("New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_messages = None
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.header("Data Chat")

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about your data..."):
    # Check connections
    if not st.session_state.sf_client:
        st.error("Please connect to Snowflake first (sidebar).")
        st.stop()
    if not st.session_state.llm_client:
        st.error("Please connect to LLM first (sidebar).")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=False)
        callbacks = StreamlitCallbacks(status)

        try:
            answer, updated_messages = run_agent(
                st.session_state.llm_client,
                st.session_state.llm_model,
                st.session_state.sf_client,
                ALL_TOOLS,
                prompt,
                messages=st.session_state.agent_messages,
                callbacks=callbacks,
            )
            st.session_state.agent_messages = updated_messages
            status.update(label="Done", state="complete")

        except SnowflakeSessionExpiredError:
            answer = "Snowflake session expired. Please reconnect using the sidebar."
            st.session_state.sf_client = None
            status.update(label="Session expired", state="error")

        except SnowflakeConnectionError:
            answer = "Lost connection to Snowflake. Please reconnect using the sidebar."
            st.session_state.sf_client = None
            status.update(label="Connection lost", state="error")

        except LLMRateLimitError as e:
            retry_msg = f" Try again in {e.retry_after:.0f}s." if e.retry_after else ""
            answer = f"Rate limited by the LLM API.{retry_msg} Please wait and try again."
            status.update(label="Rate limited", state="error")

        except LLMConnectionError:
            answer = "Cannot reach the LLM API. Please check your network connection."
            status.update(label="LLM unreachable", state="error")

        except LLMAPIError as e:
            status_msg = f" (status {e.status_code})" if e.status_code else ""
            answer = f"LLM API error{status_msg}. Please try again."
            status.update(label="LLM error", state="error")

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
