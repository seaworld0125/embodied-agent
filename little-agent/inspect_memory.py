#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory import MemoryStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="./data/agent.db")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    store = MemoryStore(Path(args.db))
    store.initialize()

    events = store.recent_events(limit=args.limit)

    for event in events:
        print(json.dumps(event, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
