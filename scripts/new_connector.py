#!/usr/bin/env python3
"""Scaffold a new connector package from the template.

Usage:
  python scripts/new_connector.py <connector_name>

This will create `connectors/<connector_name>/connector.py` copied from
`connectors/template.py` and a test skeleton under `tests/unit/`.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


TEMPLATE = Path("connectors/template.py")


def create_connector(name: str) -> None:
    target_dir = Path("connectors") / name
    target_dir.mkdir(parents=True, exist_ok=True)

    target_file = target_dir / "connector.py"
    if target_file.exists():
        print(f"Connector {name} already exists at {target_file}")
        return

    if not TEMPLATE.exists():
        print("Template not found: connectors/template.py")
        return

    # Copy template and replace placeholder source_code
    content = TEMPLATE.read_text()
    content = content.replace("YOUR_SOURCE_CODE", name.upper())
    target_file.write_text(content)
    print(f"Created {target_file}")

    # Create test skeleton
    tests_dir = Path("tests/unit")
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_file = tests_dir / f"test_connectors_{name}.py"
    if not test_file.exists():
        test_file.write_text(
            """from connectors.{name}.connector import {cls}


def test_smoke():
    # basic import smoke test
    assert {cls} is not None
""".replace("{name}", name).replace("{cls}", f"{name.capitalize()}Connector")
        )
        print(f"Created test skeleton {test_file}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: new_connector.py <name>")
        return 2
    name = argv[1]
    create_connector(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
