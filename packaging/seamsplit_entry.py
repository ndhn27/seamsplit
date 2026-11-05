"""What the frozen program runs: ``seamsplit`` with its command line (``split``, ``merge``,
``run``, ``status``, ``gui`` ...). PyInstaller starts here, so this is a script and not the
package's own ``main``."""

import multiprocessing

from seamsplit.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()  # a no-op unless something spawns a worker of this program
    raise SystemExit(main())
