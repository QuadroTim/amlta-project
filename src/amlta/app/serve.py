import sys
from pathlib import Path

from streamlit import runtime
from streamlit.web import cli


def main():
    if runtime.exists():
        from amlta.app import main  # noqa: F401
    else:
        main_file = Path(__file__).parent / "main.py"
        sys.argv = ["streamlit", "run", str(main_file), *sys.argv[1:]]
        sys.exit(cli.main())


if __name__ == "__main__":
    main()
