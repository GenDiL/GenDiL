"""Check streaming and diagnostics using real child processes."""

from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest

import ci_process


class ProcessTests(unittest.TestCase):
    def test_failure_keeps_both_streams_without_repeating_output(self):
        with tempfile.TemporaryDirectory() as directory, \
             contextlib.redirect_stdout(io.StringIO()) as console:
            log = Path(directory) / "failure.log"
            with self.assertRaisesRegex(RuntimeError, r"Command failed \(7\)"):
                ci_process.run([sys.executable, "-u", "-c",
                                "import sys; print('stdout message'); "
                                "print('stderr message', file=sys.stderr); sys.exit(7)"], log)
            self.assertEqual(console.getvalue(), log.read_text())
            lines = console.getvalue().splitlines()
            self.assertEqual(lines.count("stdout message"), 1)
            self.assertEqual(lines.count("stderr message"), 1)

    def test_output_is_visible_before_child_exits(self):
        ready = threading.Event()

        class Console(io.StringIO):
            def write(self, value):
                result = super().write(value)
                if "ready" in value.splitlines():
                    ready.set()
                return result

        with tempfile.TemporaryDirectory() as directory, \
             contextlib.redirect_stdout(Console()) as console, \
             ThreadPoolExecutor(max_workers=1) as executor:
            log = Path(directory) / "live.log"
            release = Path(directory) / "release"
            script = ("import pathlib, sys, time\n"
                      "print('ready', flush=True)\n"
                      "while not pathlib.Path(sys.argv[1]).exists(): time.sleep(0.01)\n"
                      "print('done')\n")
            future = executor.submit(ci_process.run,
                                     [sys.executable, "-u", "-c", script, str(release)], log)
            try:
                self.assertTrue(ready.wait(5), "Child output was buffered until exit")
                self.assertFalse(future.done())
                self.assertIn("ready", log.read_text().splitlines())
                self.assertNotIn("done", console.getvalue().splitlines())
            finally:
                release.touch()
                future.result(timeout=5)

    def test_large_output_does_not_fill_the_pipe(self):
        with tempfile.TemporaryDirectory() as directory, \
             contextlib.redirect_stdout(io.StringIO()) as console:
            log = Path(directory) / "large.log"
            ci_process.run([sys.executable, "-u", "-c",
                            "import sys; sys.stdout.write('x' * 262144); "
                            "sys.stderr.write('y' * 262144)"], log)
            self.assertEqual(console.getvalue(), log.read_text())
            self.assertTrue(log.read_text().endswith("x" * 262144 + "y" * 262144))

    def test_cwd_environment_append_and_default_truncation(self):
        with tempfile.TemporaryDirectory() as directory, \
             contextlib.redirect_stdout(io.StringIO()):
            log = Path(directory) / "commands.log"
            script = "import os; print(os.environ['GENDIL_TEST_VALUE']); print(os.getcwd())"
            ci_process.run([sys.executable, "-c", script], log, cwd=directory,
                           env=dict(os.environ, GENDIL_TEST_VALUE="first"))
            ci_process.run([sys.executable, "-c", "print('second')"], log, append=True)
            lines = log.read_text().splitlines()
            self.assertIn("first", lines)
            self.assertIn(str(Path(directory).resolve()), lines)
            self.assertIn("second", lines)
            self.assertEqual(sum(line.startswith("+ ") for line in lines), 2)
            ci_process.run([sys.executable, "-c", "print('replacement')"], log)
            self.assertNotIn("first", log.read_text())
            self.assertIn("replacement", log.read_text().splitlines())

    def test_nested_helper_streams_child_output_once(self):
        with tempfile.TemporaryDirectory() as directory, \
             contextlib.redirect_stdout(io.StringIO()) as console:
            outer_log = Path(directory) / "outer.log"
            inner_log = Path(directory) / "inner.log"
            script = ("import sys\n"
                      "sys.path.insert(0, sys.argv[1])\n"
                      "import ci_process\n"
                      "ci_process.run([sys.executable, '-u', '-c', "
                      "\"print('nested result')\"], sys.argv[2])\n")
            ci_process.run([sys.executable, "-u", "-c", script,
                            str(Path(ci_process.__file__).parent), str(inner_log)], outer_log)
            for text in (console.getvalue(), outer_log.read_text(), inner_log.read_text()):
                self.assertEqual(text.splitlines().count("nested result"), 1)


if __name__ == "__main__":
    unittest.main()
