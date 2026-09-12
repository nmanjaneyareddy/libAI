"""LibAI: a Streamlit chat interface for the Ollama Cloud API."""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st


OLLAMA_API_URL = "https://ollama.com/api/chat"
DEFAULT_MODEL = "gpt-oss:120b"
REQUEST_TIMEOUT_SECONDS = 300
MAX_CONVERSATION_MESSAGES = 20


SYSTEM_PROMPT = """
You are LibAI, the IIMB Library Reference Assistant.

Your role is to assist users with questions related to:
- library services and resources;
- electronic databases;
- research support and reference assistance;
- scholarly communication; and
- general academic information.

Be concise, professional, friendly, and helpful.

Do not invent IIMB Library policies, subscriptions, opening hours, rules,
contacts, or services. If reliable IIMB-specific information is not supplied
in the conversation, clearly say that you cannot confirm it and advise the
user to consult the official IIMB Library website or contact the Library.
""".strip()


st.set_page_config(
    page_title="LibAI",
    page_icon="📚",
    layout="centered",
)


def read_configuration() -> tuple[str, str]:
    """Read the API key and model name from Streamlit Secrets."""
    api_key = str(st.secrets.get("OLLAMA_API_KEY", "")).strip()
    model = str(st.secrets.get("OLLAMA_MODEL", DEFAULT_MODEL)).strip()

    if not api_key:
        st.error(
            "LibAI is not configured yet. Add OLLAMA_API_KEY to "
            "the app's Streamlit Secrets."
        )
        st.stop()

    if not model:
        model = DEFAULT_MODEL

    return api_key, model


def ask_ollama(
    messages: list[dict[str, str]],
    api_key: str,
    model: str,
) -> str:
    """Send the conversation to Ollama Cloud and return the reply."""
    response = requests.post(
        OLLAMA_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code == 401:
        raise RuntimeError("Ollama authentication failed. Check the API key.")

    if response.status_code == 404:
        raise RuntimeError(
            f"The Ollama model '{model}' is unavailable. "
            "Choose a model listed for your Ollama Cloud account."
        )

    if response.status_code == 429:
        raise RuntimeError(
            "Ollama has temporarily limited requests or the account has "
            "reached its usage allowance. Please try again later."
        )

    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        raise RuntimeError(
            f"Ollama returned an HTTP {response.status_code} error."
        ) from error

    try:
        data: dict[str, Any] = response.json()
        answer = str(data["message"]["content"]).strip()
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError("Ollama returned an unexpected response.") from error

    if not answer:
        raise RuntimeError("Ollama returned an empty response.")

    return answer


def reset_conversation() -> None:
    """Restore the conversation to its initial system message."""
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]


api_key, model = read_configuration()

st.title("📚 LibAI")
st.caption("AI-powered IIMB Library Reference Assistant")

if "messages" not in st.session_state:
    reset_conversation()

with st.sidebar:
    st.header("About LibAI")
    st.write(
        "LibAI provides general library and research assistance through "
        "an Ollama Cloud model."
    )
    st.caption(f"Model: {model}")
    if st.button("Clear conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask LibAI a library or research question...")

if question:
    clean_question = question.strip()

    if clean_question:
        st.session_state.messages.append(
            {"role": "user", "content": clean_question}
        )

        with st.chat_message("user"):
            st.markdown(clean_question)

        with st.chat_message("assistant"):
            with st.spinner("LibAI is thinking..."):
                try:
                    system_message = st.session_state.messages[0]
                    recent_messages = st.session_state.messages[
                        -MAX_CONVERSATION_MESSAGES:
                    ]
                    api_messages = [system_message] + [
                        message
                        for message in recent_messages
                        if message["role"] != "system"
                    ]

                    answer = ask_ollama(api_messages, api_key, model)
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                except requests.Timeout:
                    st.error(
                        "The request timed out. Please wait a moment and try again."
                    )

                except requests.ConnectionError:
                    st.error(
                        "LibAI could not connect to Ollama Cloud. "
                        "Please check the service and try again."
                    )

                except RuntimeError as error:
                    st.error(str(error))

                except requests.RequestException:
                    st.error(
                        "The request to Ollama Cloud failed. Please try again."
                    )
