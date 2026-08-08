"""Run the G3 platform-independent API on loopback only."""

import argparse

import uvicorn

from probstat_tutor.api import create_api_app

LOCAL_API_HOST = "127.0.0.1"
LOCAL_API_PORT = 8765


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the app and route table without opening a socket",
    )
    parser.add_argument("--port", type=int, default=LOCAL_API_PORT)
    args = parser.parse_args()
    app = create_api_app()
    if args.check:
        routes = sorted(route.path for route in app.routes)
        expected = ["/health", "/v1/diagnose", "/v1/hint", "/v1/recommend"]
        if routes != sorted(expected):
            print(f"API 路由检查失败：{routes}")
            return 1
        print("本地 API 契约检查通过；未打开端口。")
        return 0
    if not 1024 <= args.port <= 65535:
        parser.error("端口必须在 1024–65535 之间")
    uvicorn.run(
        app,
        host=LOCAL_API_HOST,
        port=args.port,
        access_log=False,
        server_header=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
