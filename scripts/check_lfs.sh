#!/bin/bash
# Detect Git LFS files that are still unresolved POINTERS in the working tree,
# i.e. a collaborator cloned without `git lfs install`. Such files are a few
# hundred bytes of text beginning with the LFS spec URL, so the notebooks would
# fail to load their cached results with a confusing error.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

unresolved=""
while IFS= read -r f; do
  [ -f "$f" ] || continue
  if head -c 64 "$f" 2>/dev/null | grep -q 'git-lfs.github.com/spec'; then
    unresolved+="   $f"$'\n'
  fi
done < <(git lfs ls-files --name-only 2>/dev/null)

if [ -n "$unresolved" ]; then
  echo "!! Unresolved Git LFS pointers — run:  git lfs install && git lfs pull" >&2
  printf '%s' "$unresolved" >&2
  exit 1
fi
echo "OK: all Git LFS files are resolved."
