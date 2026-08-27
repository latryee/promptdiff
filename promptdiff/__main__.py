"""Direct entrypoint for python -m promptdiff."""

import sys
from promptdiff.cli.app import main

if __name__ == "__main__":
    sys.exit(main())
