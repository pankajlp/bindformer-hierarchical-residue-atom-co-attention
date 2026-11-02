"""Start a command in its own session so it outlives the shell that launched it.

Long training runs were being killed when the launching terminal or editor session
went away. Double-forking and calling setsid detaches the child from that session's
process group, and stdout/stderr go to `log_path`.

    python scripts/run_detached.py <workdir> <log_path> <command> [args...]

Keep both this file and anything it runs on durable storage: a script read from a
temporary directory can be deleted mid-run, which kills the job with no error.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    workdir, log_path, command = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.chdir(workdir)
    log = open(log_path, "ab", 0)
    if os.fork():
        os._exit(0)
    os.setsid()
    if os.fork():
        os._exit(0)
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
