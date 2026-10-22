#!/bin/bash
cd "$(dirname "$0")/.."
set -eo pipefail
[ "$1" = "--skip-features" ] || uv run python scripts/features.py 2>&1 | grep -v "it/s" | tail -5
uv run python scripts/labels.py 2>&1 | tail -2
uv run python scripts/ranker_v2.py walkforward
uv run python scripts/ranker_v2.py train
uv run python research/static_seeds.py 1,2,3,4,5
echo PROD_DONE
