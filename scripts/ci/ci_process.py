"""Run CI commands with live output and a complete local log."""

import codecs
from pathlib import Path
import shlex
import subprocess
import sys


def run(command, log, *, cwd=None, env=None, append=False):
    """Stream both output channels once, retaining their merged order in the log."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a" if append else "w", encoding="utf-8") as output:
        def emit(text):
            output.write(text)
            output.flush()
            sys.stdout.write(text)
            sys.stdout.flush()

        emit("+ " + shlex.join(map(str, command)) + "\n")
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, bufsize=0) as process:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            try:
                while chunk := process.stdout.read(65536):
                    emit(decoder.decode(chunk))
                emit(decoder.decode(b"", final=True))
                returncode = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if returncode:
            raise RuntimeError(f"Command failed ({returncode}); full log: {log}")
