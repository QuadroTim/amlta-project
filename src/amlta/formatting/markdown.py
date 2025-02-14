from amlta.formatting.data import SectionData, SectionValue

_headings = [
    "#",
    "##",
    "###",
    "####",
    "#####",
    "######",
]


def format_as_markdown(data: SectionData) -> str:
    level = 0

    def render_markdown_value(value: SectionValue, level=level) -> str:
        if isinstance(value, list):
            return "\n".join(render_markdown_value(item, level=level) for item in value)
        elif isinstance(value, dict):
            return "\n\n".join(
                f"{_headings[level]} {key}\n{render_markdown_value(val, level + 1)}"
                for key, val in value.items()
            )
        else:
            return str(value)

    return render_markdown_value(data, level)
