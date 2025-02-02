from pathlib import Path
from typing import Any

from lxml import etree
from lxml.etree import _Element as Element
from lxml.etree import _ElementTree as ET

NSDict = dict[str | None, str]


def get_tree(xml_file: Any) -> tuple[ET, NSDict]:
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(xml_file, parser)
    root = tree.getroot()

    return tree, root.nsmap


def to_dict(xml_path: Path) -> dict:
    tree, namespaces = get_tree(xml_path)

    def elem_to_dict(element: Element) -> dict | str:
        children = list(element.getchildren())
        result: dict = {
            (k.split("}")[1] if "}" in k else k): v
            for k, v in element.attrib.items()
            if k not in namespaces.values()
        }

        if not children:
            if not result:
                return element.text

            result["value"] = element.text

        for child in children:
            key = child.tag.split("}")[1] if "}" in child.tag else child.tag
            value = elem_to_dict(child)

            if key in result:
                if not isinstance(result[key], list):
                    result[key] = [result[key]]

                result[key].append(value)
            else:
                result[key] = value

        return result

    result = elem_to_dict(tree.getroot())
    assert isinstance(result, dict)
    return result
