from __future__ import annotations

from typing import Any, Callable, Dict

Boundary = bytes
Callback = Callable[[bytes, int, int], None]


class MultipartParseError(RuntimeError):
    """Raised when the simplified multipart parser encounters invalid data."""


def parse_options_header(value: Any) -> tuple[Any, Dict[bytes, bytes]]:
    """Parse a Content-Type style header into its main value and parameters.

    This simplified implementation mirrors the subset of behaviour required by
    Starlette's multipart parser when the real ``python-multipart`` package is
    unavailable. Parameters are returned as a dictionary with byte keys and
    byte values, matching the interface FastAPI expects.
    """

    if value is None:
        return None, {}

    if isinstance(value, bytes):
        header = value.decode("latin-1")
    else:
        header = str(value)

    parts = [part.strip() for part in header.split(";") if part.strip()]
    if not parts:
        return header, {}

    main_value = parts[0]
    params: Dict[bytes, bytes] = {}
    for item in parts[1:]:
        if "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        key_bytes = key.strip().lower().encode("latin-1")
        value_str = raw_value.strip().strip('"')
        params[key_bytes] = value_str.encode("latin-1")
    return main_value, params


class MultipartParser:
    """Minimal multipart/form-data parser compatible with Starlette callbacks.

    The implementation buffers the incoming body and performs parsing during
    :meth:`finalize`. While not as memory efficient as the upstream library, it
    is sufficient for handling smaller uploads in environments where installing
    ``python-multipart`` is not possible.
    """

    def __init__(self, boundary: Boundary | str, callbacks: Dict[str, Callable[..., None]]):
        if isinstance(boundary, str):
            boundary_bytes = boundary.encode("latin-1")
        else:
            boundary_bytes = boundary
        if not boundary_bytes:
            raise MultipartParseError("Multipart boundary may not be empty")
        self._boundary = boundary_bytes if boundary_bytes.startswith(b"--") else b"--" + boundary_bytes
        self._callbacks = callbacks
        self._buffer = bytearray()

    def write(self, data: bytes) -> None:
        if not isinstance(data, (bytes, bytearray)):
            raise MultipartParseError("Multipart data must be bytes")
        self._buffer.extend(data)

    def finalize(self) -> None:
        data = bytes(self._buffer)
        if not data:
            return

        parts = data.split(self._boundary)
        if len(parts) < 2:
            return

        # Skip the preamble (index 0) and iterate over actual parts.
        for segment in parts[1:]:
            if segment.startswith(b"--"):
                break
            segment = segment.lstrip(b"\r\n")
            if not segment:
                continue
            self._parse_segment(segment.rstrip(b"\r\n"))

        end_callback = self._callbacks.get("on_end")
        if end_callback:
            end_callback()

    def _parse_segment(self, segment: bytes) -> None:
        header_blob, sep, body = segment.partition(b"\r\n\r\n")
        if not sep:
            raise MultipartParseError("Multipart segment missing header/body separator")
        headers = header_blob.split(b"\r\n")

        part_begin = self._callbacks.get("on_part_begin")
        if part_begin:
            part_begin()

        for header_line in headers:
            if not header_line:
                continue
            name, colon, value = header_line.partition(b":")
            if not colon:
                continue
            name = name.strip()
            value = value.strip()
            header_field_cb: Callback | None = self._callbacks.get("on_header_field")
            header_value_cb: Callback | None = self._callbacks.get("on_header_value")
            header_end_cb = self._callbacks.get("on_header_end")
            if header_field_cb:
                header_field_cb(name, 0, len(name))
            if header_value_cb:
                header_value_cb(value, 0, len(value))
            if header_end_cb:
                header_end_cb()

        headers_finished = self._callbacks.get("on_headers_finished")
        if headers_finished:
            headers_finished()

        part_data_cb: Callback | None = self._callbacks.get("on_part_data")
        if part_data_cb and body:
            part_data_cb(body, 0, len(body))

        part_end_cb = self._callbacks.get("on_part_end")
        if part_end_cb:
            part_end_cb()
