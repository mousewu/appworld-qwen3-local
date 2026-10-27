"""Drain a training subprocess into the parent SDK console proxy and a full local log."""
import codecs
from pathlib import Path
import sys
import threading


class ProcessLogRelay:
    def __init__(self, pipe, path, console=None):
        self.pipe = pipe
        self.path = Path(path)
        # Bind AFTER swanlab.init so Python writes pass through its console proxy.
        self.console = sys.stdout if console is None else console
        self.error = None
        self.thread = threading.Thread(target=self._drain, name='training-log-relay', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _drain(self):
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        try:
            with self.path.open('w', encoding='utf-8', newline='') as log:
                while True:
                    block = self.pipe.read(8192)
                    if not block:
                        break
                    text = decoder.decode(block)
                    log.write(text)
                    log.flush()
                    try:
                        self.console.write(text)
                        self.console.flush()
                    except Exception:
                        # Do not stop draining (or lose local logs) if a console SDK proxy fails.
                        pass
                tail = decoder.decode(b'', final=True)
                if tail:
                    log.write(tail)
                    try:
                        self.console.write(tail)
                        self.console.flush()
                    except Exception:
                        pass
        except Exception as exc:
            self.error = exc
        finally:
            self.pipe.close()

    def finish(self, timeout=30):
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise RuntimeError('Training output pipe did not close; local log may still be draining')
        if self.error:
            raise RuntimeError(f'Cannot persist training log: {self.error}') from self.error
