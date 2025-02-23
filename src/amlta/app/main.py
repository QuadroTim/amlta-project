import argparse
import logging
from typing import cast
from uuid import uuid4

import streamlit as st

st.set_page_config(page_title="PROBAS Copilot", layout="wide")
st.title("PROBAS Copilot")

from langchain.callbacks.tracers.logging import LoggingCallbackHandler
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from streamlit.elements.lib.mutable_status_container import StatusContainer

from amlta.app import config
from amlta.app.agent.core import (
    AgentEvent,
    AgentFinishedEvent,
    ProcessCandidatesFetchedEvent,
    RewritingFlowsQueriesEvent,
    RewritingProcessQueryEvent,
    RewrittenFlowsQueriesEvent,
    RewrittenProcessQueryEvent,
    SelectedProcessEvent,
)
from amlta.data_processing.tapas_flows import transform_flows_for_tapas
from amlta.probas.flows import extract_process_flows
from amlta.probas.processes import ProcessData

logging.basicConfig(level=logging.INFO)

if "messages" not in st.session_state:
    st.session_state.messages = []


UNSET_ARGS = argparse.Namespace(model=None, base_url=None)

chat, side = st.columns([0.7, 0.3])
chat_history = chat.container()
chat_input_container = chat.container()
agent_log_container = side.container()


def handle_event(
    event: AgentEvent,
    process_selection_container: StatusContainer,
    flows_selection_container: StatusContainer,
):
    ev = event.event
    # print(ev)

    match ev:
        case RewritingProcessQueryEvent():
            process_selection_container.update(label="Rewriting process query...")

        case RewrittenProcessQueryEvent(query=query):
            process_selection_container.update(label="Fetching process candidates...")
            process_selection_container.markdown(
                f"Rewritten process query: `{query.query}`"
            )

        case ProcessCandidatesFetchedEvent(candidates=candidates):
            process_selection_container.update(label="Selecting process...")
            process_selection_container.markdown(f"Found {len(candidates)} candidates")
            process_selection_container.markdown(
                "\n".join(
                    f"- `{process.processInformation.dataSetInformation.name.baseName.get()}`"
                    for process in candidates
                )
            )

        case SelectedProcessEvent(process_uuid=process_uuid):
            process = ProcessData.from_uuid(process_uuid)
            name = process.processInformation.dataSetInformation.name.baseName.get()
            process_selection_container.update(
                label=f"Selected process: `{name}`", state="complete"
            )
            process_selection_container.markdown(f"Selected process: `{name}`")

        case RewritingFlowsQueriesEvent():
            flows_selection_container.update(label="Rewriting flows queries...")

        case RewrittenFlowsQueriesEvent(rewritten_flows_queries=queries):
            flows_selection_container.update(
                label="Rewritten flows queries", state="complete"
            )
            flows_selection_container.markdown("Rewritten flows queries")
            flows_selection_container.markdown(
                "\n".join(f"- `{query.query}`" for query in queries.queries)
            )

        case AgentFinishedEvent(result=result):
            with chat_history.expander("Result", expanded=False):
                st.write(result)

            process = ProcessData.from_uuid(result["selected_process_uuid"])
            flows_df = transform_flows_for_tapas(extract_process_flows(process))
            flows_df = (
                flows_df.iloc[result["flows_indices"]].copy().reset_index(drop=True)
            )
            st.write(f"Aggregation: `{result['aggregation']}`")


async def main(args: argparse.Namespace = UNSET_ARGS):
    if args.model:
        config.ollama_model = args.model
    if args.base_url:
        config.ollama_base_url = args.base_url

    # late import to ensure config was loaded first
    from amlta.app.agent.graph import main as graph

    with chat_history:
        # for msg in st.session_state.messages:
        #     with st.chat_message(msg.type):
        #         st.markdown(msg.content)

        if user_input := chat_input_container.chat_input("Type your message"):
            # human_message = HumanMessage(content=user_input)
            # st.session_state.messages.append(human_message)

            process_selection_container = agent_log_container.status(
                "Process selection", expanded=True
            )
            flows_selection_container = agent_log_container.status(
                "Flows selection", expanded=True
            )

            with chat_history.chat_message("human"):
                st.markdown(user_input)

            with chat_history.chat_message("assistant"):
                st_handler = StreamlitCallbackHandler(
                    st.empty(), expand_new_thoughts=False
                )

                async for lg_event_type, event in graph.astream(
                    user_input,
                    config={
                        "callbacks": [
                            LoggingCallbackHandler(logging.getLogger("main")),
                            # get_streamlit_cb(st_handler),
                            # ToolCallbackHandler(side.container()),
                        ],
                        "configurable": {"thread_id": uuid4().hex},
                    },
                    stream_mode=["custom", "updates"],
                ):
                    if lg_event_type == "custom":
                        handle_event(
                            cast(AgentEvent, event),
                            process_selection_container=process_selection_container,
                            flows_selection_container=flows_selection_container,
                        )
                    else:
                        pass
                        # print(event)


async def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="llama3.2", help="Ollama model to use"
    )
    parser.add_argument("--base-url", type=str, default=None, help="Ollama base URL")

    args = parser.parse_args()
    await main(args)


if __name__ == "__main__":
    import asyncio

    asyncio.run(launch())
