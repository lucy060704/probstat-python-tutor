"""Export or verify the deterministic G3.4 OpenAPI JSON document."""

import argparse
import json
from pathlib import Path

from probstat_tutor.api.openapi import build_openapi_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "api" / "openapi.json"


def _render() -> str:
    return f"{json.dumps(build_openapi_contract(), ensure_ascii=False, indent=2, sort_keys=True)}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = _render()
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            print("OpenAPI 契约与 Pydantic schema 不一致，请重新导出。")
            return 1
        print("OpenAPI 契约与 Pydantic schema 一致。")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"已写入：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
