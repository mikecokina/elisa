"""UTC compatibility shim for Python 3.10+ and newer."""

from datetime import datetime

try:
    UTC = datetime.UTC  # type: ignore[attr-defined]
except AttributeError:
    from datetime import timezone
    UTC = timezone.utc

