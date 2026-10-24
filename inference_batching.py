"""One model owner serving concurrent episode threads with microbatches."""
from concurrent.futures import Future
import queue
import threading
import time


class BatchingBackend:
    def __init__(self, backend, max_batch_size=4, wait_ms=20):
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.wait_seconds = wait_ms / 1000
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.closed = False
        self.stats = dict(batches=0, requests=0, max_batch_size=0, service_seconds=0.)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def generate(self, messages):
        future = Future()
        with self.lock:
            if self.closed:
                raise RuntimeError('Inference batcher is closed')
            self.queue.put((messages, future))
        return future.result()

    def _serve(self):
        while True:
            first = self.queue.get()
            if first is None:
                return
            batch = [first]
            deadline = time.monotonic() + self.wait_seconds
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.queue.get(timeout=remaining))
                except queue.Empty:
                    break
            started = time.monotonic()
            try:
                outputs = self.backend.generate_batch([messages for messages, _ in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError('Inference result count differs from request count')
            except Exception as exc:
                outputs = [exc] * len(batch)
            with self.lock:
                self.stats['batches'] += 1
                self.stats['requests'] += len(batch)
                self.stats['max_batch_size'] = max(self.stats['max_batch_size'], len(batch))
                self.stats['service_seconds'] += time.monotonic() - started
            for (_, future), output in zip(batch, outputs):
                if isinstance(output, Exception):
                    future.set_exception(output)
                else:
                    future.set_result(output)

    def metrics(self):
        with self.lock:
            stats = dict(self.stats)
        stats['average_batch_size'] = stats['requests'] / stats['batches'] if stats['batches'] else 0.
        return stats

    def close(self):
        # Call only after all episode threads have finished; no pending futures.
        with self.lock:
            self.closed = True
        self.queue.put(None)
        self.thread.join()
