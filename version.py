from os import path

here = path.dirname(__file__)


def read_version():
    version_file = path.join(here, "src", "elisa", "__init__.py")
    with open(version_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find __version__ string.")
