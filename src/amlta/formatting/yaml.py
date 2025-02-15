from typing import Mapping

import yaml


# https://github.com/yaml/pyyaml/issues/240
# https://github.com/RamenDR/ramen/blob/main/test/drenv/yaml.py
def _str_presenter(dumper, data):
    """
    Preserve multiline strings when dumping yaml.
    https://github.com/yaml/pyyaml/issues/240
    """
    if "\n" in data:
        # Remove trailing spaces messing out the output.
        block = "\n".join([line.rstrip() for line in data.splitlines()])
        block = block.rstrip() + "\n\n"
        return dumper.represent_scalar("tag:yaml.org,2002:str", block, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, _str_presenter)
yaml.representer.SafeRepresenter.add_representer(str, _str_presenter)


# https://github.com/yaml/pyyaml/issues/127
class SeparatedSectionsDumper(yaml.SafeDumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()


def format_as_yaml(data: Mapping, line_between_sections=True) -> str:
    return yaml.dump(
        data,
        Dumper=SeparatedSectionsDumper if line_between_sections else yaml.SafeDumper,
        sort_keys=False,
        allow_unicode=True,
        width=1000,
    ).strip()
