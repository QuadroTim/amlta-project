import argparse
import logging
from typing import cast
from uuid import uuid4

import streamlit as st
from langchain.callbacks.tracers.logging import LoggingCallbackHandler
from langchain.globals import set_verbose
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from amlta.data_processing.tapas_flows import transform_flows_for_tapas
from amlta.probas.flows import extract_process_flows
from amlta.probas.processes import ProcessData

set_verbose(True)
st.set_page_config(page_title="PROBAS Copilot", layout="wide")


from amlta.app import config
from amlta.app.langgraph_callback import get_streamlit_cb

logging.basicConfig(level=logging.INFO)


st.title("PROBAS Copilot")

if "messages" not in st.session_state:
    st.session_state.messages = []


UNSET_ARGS = argparse.Namespace(model=None, base_url=None)


chat, side = st.columns([0.7, 0.3])
chat_history = chat.container()
chat_input_container = chat.container()

process_candidates_container = side.empty()


# class ToolCallbackHandler(BaseCallbackHandler):
#     def __init__(self):
#         super().__init__()
#         self.process_search_tool_ids = set()

#     def on_tool_start(
#         self,
#         serialized: dict[str, Any],
#         input_str: str,
#         *,
#         run_id: UUID,
#         parent_run_id: UUID | None = None,
#         tags: list[str] | None = None,
#         metadata: dict[str, Any] | None = None,
#         inputs: dict[str, Any] | None = None,
#         **kwargs: Any,
#     ) -> Any:
#         super().on_tool_start(
#             serialized,
#             input_str,
#             run_id=run_id,
#             parent_run_id=parent_run_id,
#             tags=tags,
#             metadata=metadata,
#             inputs=inputs,
#             **kwargs,
#         )
#         if serialized["name"] == "search_process":
#             self.process_search_tool_ids.add(run_id)

#     def on_tool_end(
#         self,
#         output: Any,
#         *,
#         run_id: UUID,
#         parent_run_id: UUID | None = None,
#         **kwargs: Any,
#     ) -> Any:
#         super().on_tool_end(
#             output, run_id=run_id, parent_run_id=parent_run_id, **kwargs
#         )
#         if run_id in self.process_search_tool_ids:
#             self.process_search_tool_ids.remove(run_id)
#             output = cast(Command, output)
#             process_docs = output.update["candidate_processes"]
#             n = len(process_docs)
#             with process_candidates_container:
#                 with st.expander(f"Found {n} processes"):
#                     tabs = st.tabs([f"Process {i + 1}" for i in range(n)])
#                     for i, doc in enumerate(process_docs):
#                         with tabs[i]:
#                             st.markdown(doc.page_content)


def main(args: argparse.Namespace = UNSET_ARGS):
    if args.model:
        config.ollama_model = args.model
    if args.base_url:
        config.ollama_base_url = args.base_url

    # late import to ensure config was loaded first

    from amlta.app.agent.graph import Output
    from amlta.app.agent.graph import main as graph

    with chat_history:
        # for msg in st.session_state.messages:
        #     with st.chat_message(msg.type):
        #         st.markdown(msg.content)

        if user_input := chat_input_container.chat_input("Type your message"):
            # human_message = HumanMessage(content=user_input)
            # st.session_state.messages.append(human_message)

            with chat_history.chat_message("human"):
                st.markdown(user_input)

            with chat_history.chat_message("assistant"):
                st_handler = StreamlitCallbackHandler(
                    st.empty(), expand_new_thoughts=False
                )

                resp = graph.invoke(
                    user_input,
                    config={
                        "callbacks": [
                            LoggingCallbackHandler(logging.getLogger("main")),
                            get_streamlit_cb(st_handler),
                            # ToolCallbackHandler(),
                        ],
                        "configurable": {"thread_id": uuid4().hex},
                    },
                )
                resp = cast(Output, resp)

                process = ProcessData.from_uuid(resp["selected_process_uuid"])
                flows_df = transform_flows_for_tapas(extract_process_flows(process))
                flows_df = (
                    flows_df.iloc[resp["flows_indices"]].copy().reset_index(drop=True)
                )
                st.write(
                    f"Rewritten process query: `{resp['rewritten_process_query']}`"
                )
                st.write(f"Rewritten flows query: `{resp['rewritten_flows_query']}`")
                st.write(
                    f"Selected process: `{process.processInformation.dataSetInformation.name.baseName.get()}`"
                )
                st.write(f"Aggregation: `{resp['aggregation']}`")
                st.write(flows_df)


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
