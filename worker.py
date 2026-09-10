"""RQ worker entry point.

Run alongside the web process. The worker owns its own embedding model and
writes into the Qdrant service, so both processes see the same index.
"""

import sys

from doc_agent import config, embeddings
from doc_agent.jobs import QUEUE_NAME, connection


def main() -> int:
    if not config.REDIS_URL:
        print("REDIS_URL is not set. The worker has no queue to consume.", file=sys.stderr)
        return 1

    if not config.QDRANT_URL:
        print(
            "QDRANT_URL is not set. A worker writing to an in-process store would "
            "produce an index the web process cannot see.",
            file=sys.stderr,
        )
        return 1

    # Load the model before consuming work so the first job is not slower than
    # the rest.
    embeddings.model()

    from rq import Worker

    print(f"worker ready, consuming {QUEUE_NAME!r}")
    Worker([QUEUE_NAME], connection=connection()).work()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
