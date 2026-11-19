"""Show CI command progress while retaining the same merged text in a log.

run.py and dependencies.py use run(command, log) instead of capturing output
until a tool exits. The command is an argument sequence, executed without a shell;
stdout and stderr share one pipe and are forwarded as the child emits them.
Children still control their own buffering, so nested Python helpers use -u.
"""

import codecs
from pathlib import Path
import shlex
import subprocess
import sys


def run(command, log, *, cwd=None, env=None, append=False):
    """Execute a command, logging its invocation and output as they become available.

    cwd/env follow subprocess semantics: None inherits the caller's settings;
    an explicit env is the complete child environment. The log is truncated by
    default, or extended with append=True. UTF-8 decoding replaces invalid bytes.
    Return after success; a nonzero exit raises RuntimeError naming the status
    and log. Launch errors propagate, and interrupted reads terminate the child.
    """
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
            # A pipe read can split a UTF-8 character across chunks.
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
