"""msgpack serialization helpers for numpy arrays.

Provides custom default/object_hook so that numpy ndarrays survive a
msgpack round-trip.  Inspired by the starVLA serialization layer.

Usage::

    from xarm_toolkit.deploy.msgpack_numpy import packb, unpackb

    data = {"image": np.zeros((240, 320, 3), dtype=np.uint8)}
    raw = packb(data)
    restored = unpackb(raw)
"""

from __future__ import annotations

import functools

import numpy as np

try:
    import msgpack
except ImportError as exc:
    raise ImportError(
        "msgpack is required for deploy module. Install with: pip install msgpack"
    ) from exc

# ---------------------------------------------------------------------------
# Encode / Decode hooks
# ---------------------------------------------------------------------------

_NDARRAY_TAG = "__ndarray__"


def pack_array(obj):
    """msgpack *default* hook — convert ndarray to a serializable dict."""
    if isinstance(obj, np.ndarray):
        return {
            _NDARRAY_TAG: True,
            "data": obj.tobytes(),
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }
    raise TypeError(f"Object of type {type(obj)} is not msgpack serializable")


def unpack_array(obj):
    """msgpack *object_hook* — restore ndarray from the dict representation."""
    if isinstance(obj, dict) and obj.get(_NDARRAY_TAG):
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(
            obj["shape"]
        )
    return obj


# ---------------------------------------------------------------------------
# Convenience wrappers (drop-in replacements with hooks pre-configured)
# ---------------------------------------------------------------------------

Packer = functools.partial(msgpack.Packer, default=pack_array, use_bin_type=True)

packb = functools.partial(msgpack.packb, default=pack_array, use_bin_type=True)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array, raw=False)

unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array, raw=False)
