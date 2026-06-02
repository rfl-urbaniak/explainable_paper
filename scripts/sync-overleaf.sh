#!/bin/bash
# Bidirectional sync between `main` and the lean `overleaf` branch that mirrors
# the Overleaf project (https://www.overleaf.com/project/<id>).
#
#   scripts/sync-overleaf.sh push   # main paper sources -> overleaf branch -> push to origin
#   scripts/sync-overleaf.sh pull   # pull Overleaf edits from origin -> stage onto main
#   scripts/sync-overleaf.sh files  # print the paper-file whitelist and exit
#
# The `overleaf` branch is an ORPHAN branch (no shared history with main, so the
# Overleaf clone stays tiny) holding ONLY the files needed to compile the paper,
# at IDENTICAL paths to main. That keeps main.tex byte-identical across branches
# and makes prose merges conflict-free.
#
# Workflow discipline: always `pull` (and commit the staged prose on main)
# BEFORE you `push`, otherwise a push overwrites unmerged Overleaf edits with
# main's versions.
set -euo pipefail

BRANCH=overleaf
REMOTE=origin
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# --- paper-file whitelist, derived so it stays correct as the paper evolves ---
paper_files() {
  # fixed sources
  printf '%s\n' main.tex references.bib neurips_2024.sty
  # main.bbl is shipped as a fallback so refs render even before bibtex runs
  [ -f main.bbl ] && printf '%s\n' main.bbl
  # all section / appendix sources (plain glob: independent of the git index,
  # which matters because `push` builds the tree under a throwaway GIT_INDEX_FILE)
  printf '%s\n' sections/*.tex
  # every figure referenced by \includegraphics, at its real path
  grep -rhoE 'includegraphics(\[[^]]*\])?\{[^}]*\}' main.tex sections/*.tex \
    | sed -E 's/.*\{([^}]*)\}/\1/' | sort -u
}

case "${1:-}" in
  files)
    paper_files | sort -u
    ;;

  push)
    echo ">> Building '$BRANCH' tree from main's paper files..."
    git fetch --quiet "$REMOTE" "$BRANCH" 2>/dev/null || true
    parent="$(git rev-parse -q --verify "$REMOTE/$BRANCH" \
              || git rev-parse -q --verify "refs/heads/$BRANCH" || true)"

    # Build the commit with a throwaway index so the working tree is untouched.
    tmpidx="$(mktemp)"; export GIT_INDEX_FILE="$tmpidx"
    git read-tree --empty
    while IFS= read -r f; do
      [ -f "$f" ] || { echo "   !! missing: $f" >&2; continue; }
      blob="$(git hash-object -w "$f")"
      git update-index --add --cacheinfo "100644,$blob,$f"
    done < <(paper_files | sort -u)
    tree="$(git write-tree)"
    unset GIT_INDEX_FILE; rm -f "$tmpidx"

    msg="Overleaf sync: paper sources from main @ $(git rev-parse --short HEAD)"
    if [ -n "$parent" ]; then
      commit="$(git commit-tree "$tree" -p "$parent" -m "$msg")"
    else
      commit="$(git commit-tree "$tree" -m "$msg")"  # first time: orphan
    fi
    git update-ref "refs/heads/$BRANCH" "$commit"
    echo ">> Updated local '$BRANCH' -> $commit"
    echo ">> Pushing to $REMOTE/$BRANCH ..."
    git push "$REMOTE" "$BRANCH"
    ;;

  pull)
    echo ">> Fetching Overleaf edits from $REMOTE/$BRANCH ..."
    git fetch "$REMOTE" "$BRANCH"
    git update-ref "refs/heads/$BRANCH" "$REMOTE/$BRANCH"
    cur="$(git rev-parse --abbrev-ref HEAD)"
    echo ">> Staging prose (main.tex + sections/) from '$BRANCH' onto '$cur' ..."
    git checkout "$BRANCH" -- main.tex sections
    echo ">> Done. Review with: git diff --cached ; then commit on '$cur'."
    ;;

  *)
    sed -n '2,17p' "$0"; exit 1 ;;
esac
