#!/bin/bash
# Bidirectional sync between `main` and the Overleaf project.
#
#   scripts/sync-overleaf.sh push   # main's paper sources -> Overleaf (and mirror to GitHub)
#   scripts/sync-overleaf.sh pull   # Overleaf edits        -> stage main.tex + sections/ onto main
#   scripts/sync-overleaf.sh files  # print the paper-file whitelist and exit
#
# Topology:
#   - Overleaf is reached through its Git bridge, remote `overleaf-bridge`
#     (branch `master`). Authenticate with an Overleaf Git token (username `git`).
#   - GitHub keeps a mirror on the lean `overleaf` branch of remote `origin`,
#     so collaborators see the paper sources there too.
#   - The synced tree holds ONLY the files needed to compile the paper, at paths
#     IDENTICAL to main, so main.tex is byte-identical across branches and prose
#     merges back into main's sections/ are conflict-free.
#
# Overleaf forbids force-push, so every `push` is layered as a child of
# Overleaf's current head. That means: ALWAYS `pull` (and commit the staged
# prose on main) BEFORE you `push`, or a push overwrites unmerged Overleaf edits.
set -euo pipefail

OL=overleaf-bridge       # Overleaf git-bridge remote
OL_BRANCH=master         # Overleaf's branch
GH=origin                # GitHub remote
GH_BRANCH=overleaf       # lean mirror branch on GitHub
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# --- paper-file whitelist, derived so it stays correct as the paper evolves ---
paper_files() {
  printf '%s\n' main.tex references.bib neurips_2024.sty
  # project README, so the setup/workflow docs are visible inside Overleaf too
  [ -f README.md ] && printf '%s\n' README.md
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
    # Guard A: main is the source of truth — refuse to push uncommitted paper
    # edits, which would live on Overleaf but not in git history.
    mapfile -t _wl < <(paper_files | sort -u)
    if ! git diff --quiet HEAD -- "${_wl[@]}" 2>/dev/null; then
      echo "!! Uncommitted changes to paper files; commit them on main first:" >&2
      git --no-pager diff --name-only HEAD -- "${_wl[@]}" | sed 's/^/     /' >&2
      exit 1
    fi

    echo ">> Fetching Overleaf head (to layer on top of it) ..."
    git fetch --quiet "$OL" "$OL_BRANCH"
    parent="$(git rev-parse "$OL/$OL_BRANCH")"

    # Guard B: refuse to push if Overleaf advanced since our last sync (someone
    # edited on Overleaf); those edits must be pulled before we overwrite them.
    last="$(git rev-parse -q --verify "refs/heads/$GH_BRANCH" || true)"
    if [ -n "$last" ] && \
       [ "$(git rev-parse "$parent^{tree}")" != "$(git rev-parse "$last^{tree}")" ]; then
      echo "!! Overleaf has changes not yet in the repo (edited on Overleaf)." >&2
      echo "!! Run 'make pull-from-overleaf', review + commit on main, then push again." >&2
      exit 1
    fi

    echo ">> Building paper tree from main's sources ..."
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

    if [ "$(git rev-parse "$parent^{tree}")" = "$tree" ]; then
      echo ">> Overleaf already matches main's paper sources. Nothing to push."
      exit 0
    fi

    msg="Sync paper sources from main @ $(git rev-parse --short HEAD)"
    commit="$(git commit-tree "$tree" -p "$parent" -m "$msg")"
    git update-ref "refs/heads/$GH_BRANCH" "$commit"

    echo ">> Pushing to Overleaf ($OL/$OL_BRANCH) ..."
    git push "$OL" "$commit:$OL_BRANCH"
    echo ">> Mirroring to GitHub ($GH/$GH_BRANCH) ..."
    git push --force "$GH" "$GH_BRANCH"
    echo ">> Done."
    ;;

  pull)
    cur="$(git rev-parse --abbrev-ref HEAD)"
    # Guard: don't clobber uncommitted local edits to the files we overwrite.
    if ! git diff --quiet HEAD -- main.tex sections 2>/dev/null; then
      echo "!! Uncommitted changes to main.tex/sections would be overwritten." >&2
      echo "!! Commit or stash them on '$cur' first." >&2
      exit 1
    fi
    echo ">> Fetching Overleaf edits ($OL/$OL_BRANCH) ..."
    git fetch "$OL" "$OL_BRANCH"
    git update-ref "refs/heads/$GH_BRANCH" "$OL/$OL_BRANCH"
    git push --force "$GH" "$GH_BRANCH"   # keep the GitHub mirror in step
    echo ">> Staging prose (main.tex + sections/) from Overleaf onto '$cur' ..."
    git checkout "$GH_BRANCH" -- main.tex sections
    echo ">> Done. Review with: git diff --cached ; then commit on '$cur'."
    ;;

  *)
    sed -n '2,8p' "$0"; exit 1 ;;
esac
