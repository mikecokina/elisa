"""Centralized logging system for capturing terminal output in Gradio UI.

This module provides a mechanism to capture Python's logging and stdout/stderr
streams and make them accessible to Gradio components for display in a
terminal-like widget.
"""

from __future__ import annotations

import logging
import sys
from collections import deque
from io import StringIO


class CircularBuffer(StringIO):
    """A StringIO subclass that maintains a maximum size circular buffer.

    When the buffer exceeds max_lines, oldest lines are removed.
    This prevents unbounded memory growth.
    """

    def __init__(self, max_lines: int = 500) -> None:
        """Initialize the circular buffer.

        :param max_lines: Maximum number of lines to retain in buffer.
        :type max_lines: int
        :returns: ``None``
        :rtype: None
        """
        super().__init__()
        self.max_lines = max_lines
        self.lines = deque(maxlen=max_lines)

    def write(self, s: str) -> int:
        """Write string to buffer and maintain circular size.

        :param s: String to write.
        :type s: str
        :returns: Number of characters written.
        :rtype: int
        """
        if not s:
            return 0

        for line in s.splitlines(keepends=True):
            self.lines.append(line)

        return len(s)

    def getvalue(self) -> str:
        """Get current buffer contents.

        :returns: All lines in buffer as a single string.
        :rtype: str
        """
        return "".join(self.lines)

    def clear(self) -> None:
        """Clear all contents from buffer.

        :returns: ``None``
        :rtype: None
        """
        self.lines.clear()


class GradioLoggingHandler(logging.Handler):
    """Custom logging handler that writes to a circular buffer.

    This handler captures all log records and stores them in a
    CircularBuffer for display in Gradio components.
    """

    def __init__(self, buffer: CircularBuffer, *, include_timestamp: bool = True) -> None:
        """Initialize the handler.

        :param buffer: CircularBuffer instance to write logs to.
        :type buffer: CircularBuffer
        :param include_timestamp: Whether to include timestamp in log output.
        :type include_timestamp: bool
        :returns: ``None``
        :rtype: None
        """
        super().__init__()
        self.buffer = buffer
        self.include_timestamp = include_timestamp
        if include_timestamp:
            self.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S",
                ),
            )
        else:
            self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the buffer.

        :param record: The log record to emit.
        :type record: logging.LogRecord
        :returns: ``None``
        :rtype: None
        """
        try:
            msg = self.format(record)
            self.buffer.write(msg + "\n")
        except Exception:  # noqa: BLE001
            self.handleError(record)


class TerminalCapture:
    """Context manager to temporarily redirect stdout/stderr to a buffer."""

    def __init__(self, buffer: CircularBuffer) -> None:
        """Initialize the capture context.

        :param buffer: CircularBuffer instance to write output to.
        :type buffer: CircularBuffer
        :returns: ``None``
        :rtype: None
        """
        self.buffer = buffer
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self) -> TerminalCapture:  # noqa: PYI034
        """Enter context and redirect stdout/stderr.

        :returns: Self.
        :rtype: TerminalCapture
        """
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.buffer
        sys.stderr = self.buffer
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Exit context and restore stdout/stderr.

        :returns: ``None``
        :rtype: None
        """
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr


class UILogger:
    """Centralized logger for the Gradio UI.

    Manages a shared CircularBuffer and provides methods to configure logging
    and access the buffered output.
    """

    _buffer: CircularBuffer | None = None
    _handler: GradioLoggingHandler | None = None

    @classmethod
    def get_buffer(cls) -> CircularBuffer:
        """Get or create the shared buffer.

        :returns: The CircularBuffer instance.
        :rtype: CircularBuffer
        """
        if cls._buffer is None:
            cls._buffer = CircularBuffer(max_lines=1000)
        return cls._buffer

    @classmethod
    def get_output(cls) -> str:
        """Get current buffered output.

        :returns: Current contents of the buffer.
        :rtype: str
        """
        return cls.get_buffer().getvalue()

    @classmethod
    def clear_output(cls) -> None:
        """Clear all buffered output.

        :returns: ``None``
        :rtype: None
        """
        cls.get_buffer().clear()

    @classmethod
    def setup_logging(cls, *, include_timestamp: bool = True, level: int = logging.INFO) -> None:
        """Set up logging to capture to the buffer.

        Adds a GradioLoggingHandler to the root logger and all
        elisa.* loggers. Also redirects stdout and stderr to capture
        print() statements.

        :param include_timestamp: Whether to include timestamp in log output.
        :type include_timestamp: bool
        :param level: Logging level to capture (default: INFO).
        :type level: int
        :returns: ``None``
        :rtype: None
        """
        import sys  # noqa: PLC0415

        buffer = cls.get_buffer()

        # Create handler if needed
        if cls._handler is None:
            cls._handler = GradioLoggingHandler(buffer, include_timestamp=include_timestamp)
            cls._handler.setLevel(level)

        # Add to root logger
        root_logger = logging.getLogger()
        if cls._handler not in root_logger.handlers:
            root_logger.addHandler(cls._handler)

        # Add to elisa loggers
        elisa_logger = logging.getLogger("elisa")
        if cls._handler not in elisa_logger.handlers:
            elisa_logger.addHandler(cls._handler)

        # Redirect stdout and stderr to capture print statements
        if not isinstance(sys.stdout, CircularBuffer):
            sys.stdout = buffer  # type: ignore[assignment]
        if not isinstance(sys.stderr, CircularBuffer):
            sys.stderr = buffer  # type: ignore[assignment]

    @classmethod
    def get_capture_context(cls) -> TerminalCapture:
        """Get a context manager to capture stdout/stderr.

        :returns: TerminalCapture context manager.
        :rtype: TerminalCapture
        """
        return TerminalCapture(cls.get_buffer())

    @classmethod
    def set_max_lines(cls, max_lines: int) -> None:
        """Set the maximum number of lines to retain in buffer.

        :param max_lines: Maximum lines to retain.
        :type max_lines: int
        :returns: ``None``
        :rtype: None
        """
        buffer = cls.get_buffer()
        if max_lines != buffer.max_lines:
            # Create new buffer with new max_lines
            old_lines = buffer.lines.copy()
            buffer.lines = deque(old_lines, maxlen=max_lines)
            buffer.max_lines = max_lines

