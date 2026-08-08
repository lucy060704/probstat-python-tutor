"""Run five consecutive G1 journeys and print machine-readable evidence."""

import asyncio
import tempfile
from pathlib import Path

from probstat_tutor.demo import run_g1_offline_demo


def main() -> int:
    """Run the local-only demo and return a shell-friendly status code."""

    with tempfile.TemporaryDirectory(prefix="probstat-g1-demo-") as temporary_directory:
        summary = asyncio.run(run_g1_offline_demo(Path(temporary_directory)))
    print(summary.model_dump_json(indent=2))
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
