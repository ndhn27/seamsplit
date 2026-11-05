"""``python -m seamsplit ...`` is the same as the ``seamsplit`` command."""

import sys

from .cli import main

sys.exit(main())
