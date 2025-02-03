from os import PathLike

from amlta.config import config
from amlta.probas import xml


def read_glossary(path: PathLike | None = None) -> dict[str, str]:
    if path is None:
        path = config.data_dir / "glossary.html"

    with open(path, "r", encoding="utf-8") as file:
        tree, ns = xml.get_tree(file)

    glossary_data = {}
    letter_sections = tree.findall('.//div[@class="letter-section"]', ns)
    for section in letter_sections:
        heading = section.find('.//h2[@class="letter-heading"]', ns)
        if heading is not None:
            dts = section.findall('.//dt[@class="term"]', ns)
            dds = section.findall('.//dd[@class="definition"]', ns)
            for dt, dd in zip(dts, dds):
                term = "".join(dt.itertext()).strip()
                definition = "".join(dd.itertext()).strip()
                glossary_data[term] = definition

    return glossary_data


if __name__ == "__main__":
    glossary = read_glossary()
    print(glossary)
