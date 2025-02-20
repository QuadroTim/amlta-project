import argparse  # noqa: I001

import streamlit as st

st.set_page_config(page_title="PROBAS Copilot")

from langchain_core.messages import HumanMessage

from amlta.app.agent.graph import graph
from amlta.app.langgraph_callback import get_streamlit_cb

st.title("PROBAS Copilot")

if "messages" not in st.session_state:
    st.session_state.messages = []


UNSET_ARGS = argparse.Namespace(model=None, base_url=None)


def main(args: argparse.Namespace = UNSET_ARGS):
    config = {
        "ollama_model": args.model,
        "ollama_base_url": args.base_url,
    }

    g = graph.with_config(
        configurable={key: value for key, value in config.items() if value is not None}
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

    if user_input := st.chat_input("Type your message"):
        human_message = HumanMessage(content=user_input)
        st.session_state.messages.append(human_message)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            resp = g.invoke(
                {
                    "messages": st.session_state.messages,
                    "initial_question": user_input,
                },
                config={"callbacks": [get_streamlit_cb(st.empty())]},
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
