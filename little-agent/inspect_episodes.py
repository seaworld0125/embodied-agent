#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from memory import MemoryStore


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        default="./data/agent.db",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--events",
        action="store_true",
        help="also show events inside each episode",
    )

    args = parser.parse_args()

    store = MemoryStore(args.db)
    store.initialize()

    episodes = store.recent_episodes(
        limit=args.limit
    )

    for episode in episodes:
        print("=" * 72)

        print(
            json.dumps(
                episode,
                ensure_ascii=False,
                indent=2,
            )
        )

        if args.events:
            events = store.episode_events(
                episode["id"]
            )

            for event in events:
                print(
                    "  "
                    + json.dumps(
                        event,
                        ensure_ascii=False,
                    )
                )


if __name__ == "__main__":
    main()
