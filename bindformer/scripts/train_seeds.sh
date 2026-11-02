#!/bin/zsh
# Train one configuration over several seeds, sequentially.
#
#   scripts/train_seeds.sh <config> <checkpoint_dir_prefix> <seed> [seed...]
#
# Each seed goes to <prefix><seed>, which is what scripts/compare.py expects as one
# arm of a comparison. Run it under scripts/run_detached.py for anything long.

set -u
if [ $# -lt 3 ]; then
  echo "usage: $0 <config> <checkpoint_dir_prefix> <seed> [seed...]" >&2
  exit 2
fi

config=$1
prefix=$2
shift 2

cd "${0:A:h}/.." || exit 1
PY=${PY:-../.venv/bin/python}

for s in "$@"; do
  echo "=== $prefix seed $s start $(date)"
  if ! $PY -u scripts/train.py --config "$config" --seed "$s" --checkpoint_dir "$prefix$s"; then
    echo "=== $prefix seed $s FAILED $(date)"
    exit 1
  fi
  echo "=== $prefix seed $s done $(date)"
done
echo "=== ALL SEEDS DONE $prefix $(date)"
