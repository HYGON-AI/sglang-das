"""Track source ownership across queueing, parking and transfer execution.

An unexpected worker exception poisons the room. This intentionally retains its
resources: an allocator or transfer API exception is not proof of quiescence.
"""

import threading


class SourceDrainTracker:
    def __init__(self):
        self.lock = threading.RLock()
        self._pending = {}
        self._poisoned = set()

    def publish(self, chunk, queue, accept):
        # The sender's abort uses the same lock. Register before publication so
        # a queued but not yet dequeued chunk prevents source reclamation.
        with self.lock:
            if not accept():
                return
            self._pending.setdefault(chunk.room, {})[id(chunk)] = chunk
            try:
                queue.put(chunk)
            except BaseException:
                # Publication might have succeeded before the exception.
                self._poisoned.add(chunk.room)
                raise

    def complete(self, chunk):
        with self.lock:
            pending = self._pending.get(chunk.room)
            if pending is not None:
                pending.pop(id(chunk), None)
                if not pending:
                    self._pending.pop(chunk.room, None)

    def poison(self, room):
        with self.lock:
            self._poisoned.add(room)

    def drained(self, room):
        with self.lock:
            return room not in self._poisoned and not self._pending.get(room)
