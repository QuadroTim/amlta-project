import argparse
import logging
from typing import Any, cast
from uuid import UUID

import httpx
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler
from langgraph.types import Command

st.set_page_config(page_title="PROBAS Copilot", layout="wide")

from langchain_core.messages import HumanMessage

from amlta.app.agent.graph import graph
from amlta.app.langgraph_callback import get_streamlit_cb

logging.basicConfig(level=logging.INFO)


st.title("PROBAS Copilot")

if "messages" not in st.session_state:
    st.session_state.messages = []


UNSET_ARGS = argparse.Namespace(model=None, base_url=None)


chat, side = st.columns([0.7, 0.3])
chat_history = chat.empty()
chat_input_container = chat.empty()

process_candidates_container = side.empty()


class ToolCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.process_search_tool_ids = set()

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        super().on_tool_start(
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            inputs=inputs,
            **kwargs,
        )
        if serialized["name"] == "search_process":
            self.process_search_tool_ids.add(run_id)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        super().on_tool_end(
            output, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )
        if run_id in self.process_search_tool_ids:
            self.process_search_tool_ids.remove(run_id)
            output = cast(Command, output)
            process_docs = output.update["candidate_processes"]
            n = len(process_docs)
            with process_candidates_container:
                with st.expander(f"Found {n} processes"):
                    tabs = st.tabs([f"Process {i + 1}" for i in range(n)])
                    for i, doc in enumerate(process_docs):
                        with tabs[i]:
                            st.markdown(doc.page_content)


def _drop_none(data: dict) -> dict:
    return {k: v for k, v in data.items() if v is not None}


def main(args: argparse.Namespace = UNSET_ARGS):
    config = {
        "ollama": _drop_none(
            {
                "model": args.model,
                "base_url": args.base_url,
            }
        )
    }

    if args.base_url and "ngrok" in args.base_url:
        httpx.post(args.base_url, headers={"ngrok-skip-browser-warning": "skip"})

    g = graph.with_config(configurable=config)

    with chat_history:
        for msg in st.session_state.messages:
            with st.chat_message(msg.type):
                st.markdown(msg.content)

        if user_input := chat_input_container.chat_input("Type your message"):
            print(user_input)
            human_message = HumanMessage(content=user_input)
            st.session_state.messages.append(human_message)

            with chat_history.chat_message("user"):
                st.markdown(user_input)

            with chat_history.chat_message("assistant"):
                st_handler = StreamlitCallbackHandler(
                    st.empty(), expand_new_thoughts=False
                )

                resp = g.invoke(
                    {
                        "messages": st.session_state.messages,
                        "initial_question": user_input,
                    },
                    config={
                        "callbacks": [
                            get_streamlit_cb(st_handler),
                            ToolCallbackHandler(),
                        ]
                    },
                )

                message = resp["messages"][-1]

                st.session_state.messages.append(message)
                st.markdown(message.content)


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="llama3.2", help="Ollama model to use"
    )
    parser.add_argument("--base-url", type=str, default=None, help="Ollama base URL")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    launch()
