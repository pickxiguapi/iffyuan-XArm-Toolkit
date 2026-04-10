"""Tests for the deploy module (msgpack_numpy, VLAClient, VLAServer).

Run with:  pytest tests/test_deploy.py -v
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Pre-mock heavy / hardware-only dependencies
# ---------------------------------------------------------------------------
for mod_name in (
    "websockets",
    "websockets.sync",
    "websockets.sync.client",
    "websockets.asyncio",
    "websockets.asyncio.server",
):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()


# ---------------------------------------------------------------------------
# Async WebSocket mock helper
# ---------------------------------------------------------------------------

class MockAsyncWebSocket:
    """A mock async WebSocket that yields pre-loaded messages."""

    def __init__(self, messages: list[bytes]):
        self._messages = messages
        self.remote_address = ("127.0.0.1", 12345)
        self.sent: list[bytes] = []

    async def send(self, data: bytes):
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


# =========================================================================
# Tests: msgpack_numpy
# =========================================================================

class TestMsgpackNumpy:
    """Round-trip serialization of numpy arrays via msgpack."""

    def test_roundtrip_float64(self):
        from xarm_toolkit.deploy.msgpack_numpy import packb, unpackb

        arr = np.random.randn(6).astype(np.float64)
        data = {"values": arr}
        restored = unpackb(packb(data))
        np.testing.assert_array_equal(restored["values"], arr)

    def test_roundtrip_uint8_image(self):
        from xarm_toolkit.deploy.msgpack_numpy import packb, unpackb

        img = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        data = {"image": img}
        restored = unpackb(packb(data))
        np.testing.assert_array_equal(restored["image"], img)

    def test_roundtrip_float32(self):
        from xarm_toolkit.deploy.msgpack_numpy import packb, unpackb

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        restored = unpackb(packb({"a": arr}))
        np.testing.assert_array_equal(restored["a"], arr)
        assert restored["a"].dtype == np.float32

    def test_nested_dict(self):
        from xarm_toolkit.deploy.msgpack_numpy import packb, unpackb

        data = {
            "instruction": "pick up the cup",
            "images": [
                np.zeros((240, 320, 3), dtype=np.uint8),
                np.ones((240, 320, 3), dtype=np.uint8),
            ],
            "state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        }
        restored = unpackb(packb(data))
        assert restored["instruction"] == "pick up the cup"
        assert len(restored["images"]) == 2
        np.testing.assert_array_equal(restored["images"][0], data["images"][0])
        np.testing.assert_array_equal(restored["state"], data["state"])

    def test_pack_array_preserves_shape(self):
        from xarm_toolkit.deploy.msgpack_numpy import packb, unpackb

        arr = np.zeros((16, 7), dtype=np.float32)
        restored = unpackb(packb({"actions": arr}))
        assert restored["actions"].shape == (16, 7)

    def test_non_array_passthrough(self):
        from xarm_toolkit.deploy.msgpack_numpy import packb, unpackb

        data = {"name": "test", "value": 42, "flag": True}
        restored = unpackb(packb(data))
        assert restored["name"] == "test"
        assert restored["value"] == 42
        assert restored["flag"] is True

    def test_pack_non_serializable_raises(self):
        from xarm_toolkit.deploy.msgpack_numpy import pack_array

        with pytest.raises(TypeError, match="not msgpack serializable"):
            pack_array(object())


# =========================================================================
# Tests: VLAClient message assembly
# =========================================================================

class TestVLAClientMessages:
    """Test that VLAClient assembles messages in the correct format."""

    def test_predict_message_format(self):
        """Verify the message dict that predict() would send."""
        from xarm_toolkit.deploy.msgpack_numpy import packb, unpackb

        # Simulate what client.predict() builds internally
        images = [
            np.zeros((240, 320, 3), dtype=np.uint8),
            np.ones((240, 320, 3), dtype=np.uint8),
        ]
        instruction = "pick up the red cup"
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        msg = {
            "type": "infer",
            "images": images,
            "instruction": instruction,
            "state": state,
        }

        # Round-trip through msgpack
        restored = unpackb(packb(msg))

        assert restored["type"] == "infer"
        assert restored["instruction"] == instruction
        assert len(restored["images"]) == 2
        np.testing.assert_array_equal(restored["images"][0], images[0])
        np.testing.assert_array_equal(restored["state"], state)

    def test_ping_message_format(self):
        from xarm_toolkit.deploy.msgpack_numpy import packb, unpackb

        msg = {"type": "ping"}
        restored = unpackb(packb(msg))
        assert restored["type"] == "ping"


# =========================================================================
# Tests: VLAServer message routing
# =========================================================================

class TestVLAServerRouting:
    """Test server message routing logic using a mock policy and WebSocket."""

    @pytest.fixture()
    def mock_policy(self):
        policy = MagicMock()
        policy.predict_action.return_value = {
            "actions": np.zeros((16, 7), dtype=np.float32),
        }
        return policy

    @pytest.fixture()
    def server(self, mock_policy):
        # Import with websockets already mocked
        from xarm_toolkit.deploy.server import VLAServer
        return VLAServer(
            policy=mock_policy,
            host="0.0.0.0",
            port=10093,
            metadata={"model": "pi0.5"},
        )

    def test_server_construction(self, server, mock_policy):
        assert server.policy is mock_policy
        assert server.port == 10093
        assert server.metadata["model"] == "pi0.5"

    def test_ok_response_format(self):
        from xarm_toolkit.deploy.msgpack_numpy import unpackb
        from xarm_toolkit.deploy.server import _ok_response

        raw = _ok_response({"type": "pong"})
        resp = unpackb(raw)
        assert resp["status"] == "ok"
        assert resp["data"]["type"] == "pong"

    def test_error_response_format(self):
        from xarm_toolkit.deploy.msgpack_numpy import unpackb
        from xarm_toolkit.deploy.server import _error_response

        raw = _error_response("test_error", "something went wrong")
        resp = unpackb(raw)
        assert resp["status"] == "error"
        assert resp["error"]["message"] == "test_error"
        assert resp["error"]["detail"] == "something went wrong"

    def test_handle_ping(self, server):
        """Simulate a ping message through the handler."""
        from xarm_toolkit.deploy.msgpack_numpy import packb, unpackb

        ping_msg = packb({"type": "ping"})
        ws = MockAsyncWebSocket([ping_msg])

        asyncio.get_event_loop().run_until_complete(server._handle(ws))

        # First send = metadata handshake, second = pong response
        assert len(ws.sent) == 2
        pong_resp = unpackb(ws.sent[1])
        assert pong_resp["status"] == "ok"
        assert pong_resp["data"]["type"] == "pong"

    def test_handle_infer(self, server, mock_policy):
        """Simulate an infer message through the handler."""
        from xarm_toolkit.deploy.msgpack_numpy import packb, unpackb

        infer_msg = packb({
            "type": "infer",
            "images": [np.zeros((240, 320, 3), dtype=np.uint8)],
            "instruction": "test",
            "state": np.zeros(6),
        })
        ws = MockAsyncWebSocket([infer_msg])

        asyncio.get_event_loop().run_until_complete(server._handle(ws))

        # Policy should have been called
        mock_policy.predict_action.assert_called_once()

        # Check response
        assert len(ws.sent) == 2  # metadata + actions
        infer_resp = unpackb(ws.sent[1])
        assert infer_resp["status"] == "ok"
        assert infer_resp["data"]["type"] == "actions"
        assert "actions" in infer_resp["data"]

    def test_handle_unknown_type(self, server):
        """Unknown message types should return an error."""
        from xarm_toolkit.deploy.msgpack_numpy import packb, unpackb

        bad_msg = packb({"type": "unknown_cmd"})
        ws = MockAsyncWebSocket([bad_msg])

        asyncio.get_event_loop().run_until_complete(server._handle(ws))

        assert len(ws.sent) == 2  # metadata + error
        err_resp = unpackb(ws.sent[1])
        assert err_resp["status"] == "error"
        assert "unknown_type" in err_resp["error"]["message"]

    def test_handle_inference_error(self, server, mock_policy):
        """Inference errors should return an error response, not crash."""
        from xarm_toolkit.deploy.msgpack_numpy import packb, unpackb

        mock_policy.predict_action.side_effect = ValueError("model exploded")

        infer_msg = packb({
            "type": "infer",
            "images": [],
            "instruction": "fail",
            "state": np.zeros(6),
        })
        ws = MockAsyncWebSocket([infer_msg])

        asyncio.get_event_loop().run_until_complete(server._handle(ws))

        assert len(ws.sent) == 2  # metadata + error
        err_resp = unpackb(ws.sent[1])
        assert err_resp["status"] == "error"
        assert "inference_error" in err_resp["error"]["message"]
        assert "model exploded" in err_resp["error"]["detail"]
