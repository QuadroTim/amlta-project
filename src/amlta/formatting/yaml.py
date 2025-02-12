import yaml

from amlta.formatting.data import SectionData


# https://github.com/yaml/pyyaml/issues/127
class SeparatedSectionsDumper(yaml.SafeDumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()


def format_as_yaml(data: SectionData) -> str:
    return yaml.dump(
        data,
        Dumper=SeparatedSectionsDumper,
        sort_keys=False,
        allow_unicode=True,
        width=1000,
    )
