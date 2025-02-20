from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from amlta.app.rag import uuids
from amlta.formatting.data import create_process_section
from amlta.formatting.markdown import format_as_markdown
from amlta.formatting.yaml import format_as_yaml
from amlta.probas.glossary import read_glossary
from amlta.probas.processes import ProcessData, read_uuids


class YamlProcessLoader(BaseLoader):
    def lazy_load(self):
        for uuid in read_uuids():
            process = ProcessData.from_uuid(uuid)
            process_sections = create_process_section(process, include_flows=False)

            yield Document(
                id=uuid,
                page_content=format_as_yaml(process_sections),
                metadata={"uuid": uuid, "type": "process", "format": "yaml"},
            )


class MarkdownProcessLoader(BaseLoader):
    def lazy_load(self):
        for uuid in read_uuids():
            process = ProcessData.from_uuid(uuid)
            process_sections = create_process_section(process, include_flows=False)

            yield Document(
                id=uuid,
                page_content=format_as_markdown(process_sections),
                metadata={"uuid": uuid, "type": "process", "format": "markdown"},
            )


class YamlGlossaryLoader(BaseLoader):
    def lazy_load(self):
        glossary = read_glossary()
        for term, definition in glossary.items():
            uuid = uuids.get_uuid(term).hex

            yield Document(
                id=uuid,
                page_content=format_as_yaml(
                    {"term": term, "definition": definition},
                    line_between_sections=False,
                ),
                metadata={"term": term, "type": "glossary", "format": "yaml"},
            )


class MarkdownGlossaryLoader(BaseLoader):
    def lazy_load(self):
        glossary = read_glossary()
        for term, definition in glossary.items():
            uuid = uuids.get_uuid(term).hex
            content = f"## {term}\n{definition}"

            yield Document(
                id=uuid,
                page_content=content,
                metadata={"term": term, "type": "glossary", "format": "markdown"},
            )
