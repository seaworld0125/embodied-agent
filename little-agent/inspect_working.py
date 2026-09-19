#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from context import build_realtime_context
from memory import MemoryStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_id")
    parser.add_argument("--db", default="./data/agent.db")
    args = parser.parse_args()

    store = MemoryStore(args.db)
    store.initialize()
    snapshot = build_realtime_context(store, args.episode_id)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
