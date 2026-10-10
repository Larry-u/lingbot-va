# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Serve ``LingbotRemoteModel`` to the DexDojo simulator over XPolicyLab's
websocket protocol.

Same launcher skeleton as the W0 ``policy_bridge.py``: the protocol shell
(WS framing, session bookkeeping) is imported at runtime from the DexDojo
checkout (``client_server.ws.model_server.PolicyServer``) so this bridge
speaks exactly the protocol the pinned simulator expects. Run inside an env
with websockets>=13 + msgpack (the shared policy venv works); model weights
stay in the LingBot-VA server process — this one is CPU-only.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import subprocess
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from lingbot_policy_model import LingbotRemoteModel  # noqa: E402


def _dexdojo_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _load_policy_server(root: Path):
    xpl = root / "XPolicyLab"
    marker = xpl / "client_server" / "ws" / "model_server.py"
    if not marker.is_file():
        raise SystemExit(
            f"[bridge] not a DexDojo checkout (missing {marker}); "
            "set --dexdojo-root / DEXDOJO_ROOT"
        )
    if str(xpl) not in sys.path:
        sys.path.insert(0, str(xpl))
    import inspect
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    params = list(inspect.signature(PolicyServer).parameters)
    if params[:2] != ["model", "config"]:
        raise SystemExit(
            f"[bridge] PolicyServer signature changed (params: {params}); "
            "re-pin the DexDojo checkout"
        )
    config_params = set(inspect.signature(PolicyServerConfig).parameters)
    if not {"host", "port"} <= config_params:
        raise SystemExit(
            f"[bridge] PolicyServerConfig no longer takes host/port "
            f"(params: {sorted(config_params)}); re-pin the DexDojo checkout"
        )
    if not hasattr(PolicyServer, "serve_forever"):
        raise SystemExit("[bridge] PolicyServer.serve_forever is gone; re-pin the checkout")
    return PolicyServer, PolicyServerConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True,
                        help="port the simulator connects to")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--server-url", default="ws://127.0.0.1:29536",
                        help="LingBot-VA websocket server (wan_va_server)")
    parser.add_argument("--dexdojo-root", default=os.environ.get("DEXDOJO_ROOT"),
                        help="Xspark-wuji checkout (default: $DEXDOJO_ROOT)")
    parser.add_argument("--server-wait", type=float, default=3600.0,
                        help="seconds to wait for the LingBot-VA server "
                             "(checkpoint load + warmup)")
    parser.add_argument("--predict-timeout", type=float, default=300.0)
    args = parser.parse_args()

    if not args.dexdojo_root:
        raise SystemExit("[bridge] --dexdojo-root or DEXDOJO_ROOT is required")
    root = Path(args.dexdojo_root).resolve()
    PolicyServer, PolicyServerConfig = _load_policy_server(root)
    print(f"[bridge] dexdojo checkout: {root} @ {_dexdojo_commit(root)}", flush=True)

    model = LingbotRemoteModel(server_url=args.server_url,
                               timeout=args.predict_timeout)
    info = model.wait_server(timeout_s=args.server_wait)
    print(f"[bridge] server_info: {info}", flush=True)
    print(f"[bridge] listening on {args.host}:{args.port}", flush=True)
    # no websocket keepalive: Isaac's synchronous scene setup can stall the
    # sim for minutes; default 20s ping would have the bridge close the
    # connection mid-episode
    cfg_kwargs = {"host": args.host, "port": args.port}
    config_params = set(inspect.signature(PolicyServerConfig).parameters)
    for extra in ("ws_ping_interval_s", "ws_ping_timeout_s"):
        if extra in config_params:
            cfg_kwargs[extra] = None
    server = PolicyServer(model, PolicyServerConfig(**cfg_kwargs))
    asyncio.run(server.serve_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
