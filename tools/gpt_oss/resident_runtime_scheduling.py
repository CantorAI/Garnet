"""Explicit process-local scheduling experiment for serial resident requests."""
import sys
import threading

POLICY = 'GARNET_RESIDENT_RUNTIME_SWITCH_US'
VALUES = ('0', '100', '200', '1000')
_active = threading.Lock()


def parse_switch_us(value):
    if not isinstance(value, str) or value not in VALUES:
        raise ValueError(POLICY + ' must be 0, 100, 200 or 1000')
    return int(value)


class RuntimeSwitchScope:
    """Restore the caller's interval even when request execution fails.

    No background contexts may overlap an enabled scope in this process.
    This does not change the interval of any other inference process.
    """
    def __init__(self, microseconds, runtime=sys):
        self.microseconds = parse_switch_us(str(microseconds))
        self.runtime = runtime
        self.metadata = dict(requested_microseconds=self.microseconds,
            default_seconds=None, effective_seconds=None, restored_seconds=None,
            applied=False)

    def _restore(self):
        self.runtime.setswitchinterval(self.metadata['default_seconds'])
        restored = self.runtime.getswitchinterval()
        self.metadata['restored_seconds'] = restored
        if abs(restored - self.metadata['default_seconds']) > 1e-12:
            raise RuntimeError('Runtime scheduling interval was not restored')

    def __enter__(self):
        if not self.microseconds:
            return self
        if not _active.acquire(blocking=False):
            raise RuntimeError('Runtime scheduling scopes cannot overlap')
        attempted = False
        try:
            original = self.runtime.getswitchinterval()
            if not isinstance(original, (int, float)) or not 0 < original < 1:
                raise RuntimeError('Invalid original runtime scheduling interval')
            self.metadata['default_seconds'] = original
            attempted = True
            self.runtime.setswitchinterval(self.microseconds / 1000000.0)
            effective = self.runtime.getswitchinterval()
            self.metadata['effective_seconds'] = effective
            if abs(effective - self.microseconds / 1000000.0) > 1e-12:
                raise RuntimeError('Runtime scheduling interval was not applied')
            self.metadata['applied'] = True
            return self
        except BaseException:
            try:
                if attempted:
                    self._restore()
            finally:
                _active.release()
            raise

    def __exit__(self, kind, value, traceback):
        if self.microseconds:
            try:
                self._restore()
            finally:
                _active.release()
        return False
