#!/bin/bash
# Build a self-contained arXiv submission tree in arxiv/, then test-compile it.
#
#   scripts/make-arxiv.sh          # build arxiv/ + arxiv.tar.gz, then verify
#   scripts/make-arxiv.sh files    # print the file list and exit
#
# arXiv runs pdflatex but not bibtex, so main.bbl ships and references.bib does
# not. Paths inside arxiv/ mirror the repo, so main.tex compiles unedited.
#
# arXiv publishes the source tarball, so the copy drops whole-line comments:
# superseded definitions and preamble bookkeeping stay in the repo and out of
# the public tree. A trailing % (LaTeX line continuation) is left alone.
#
# Verification compiles the tree in a scratch directory with no access to the
# repo, which is the only way to catch a figure the tree forgot to copy.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
OUT=arxiv
TAR=arxiv.tar.gz

submission_files() {
  printf '%s\n' main.tex neurips_2024.sty main.bbl
  printf '%s\n' sections/*.tex
  grep -rhoE 'includegraphics(\[[^]]*\])?\{[^}]*\}' main.tex sections/*.tex \
    | sed -E 's/.*\{([^}]*)\}/\1/' | sort -u
}

if [ "${1:-}" = files ]; then
  submission_files | sort -u
  exit 0
fi

# main.bbl is a build product; a stale one silently ships wrong references.
if [ main.bbl -ot references.bib ]; then
  echo "!! main.bbl is older than references.bib. Run 'make main' first." >&2
  exit 1
fi

# Inside verbatim a leading % is literal text, so refuse to strip rather than
# silently delete a line of the paper.
if grep -qE '\\begin\{(verbatim|Verbatim|lstlisting|minted)\}' main.tex sections/*.tex; then
  echo "!! Verbatim environment found; the comment stripper would corrupt it." >&2
  exit 1
fi

echo ">> Collecting sources into $OUT/ ..."
rm -rf "$OUT" "$TAR"
missing=0
stripped=0
while IFS= read -r f; do
  if [ ! -f "$f" ]; then echo "   !! missing: $f" >&2; missing=1; continue; fi
  mkdir -p "$OUT/$(dirname "$f")"
  case "$f" in
    *.tex)
      n="$(grep -c '^[[:space:]]*%' "$f" || true)"
      stripped=$((stripped + n))
      grep -v '^[[:space:]]*%' "$f" > "$OUT/$f" || true
      ;;
    *) cp "$f" "$OUT/$f" ;;
  esac
done < <(submission_files | sort -u)
[ "$missing" -eq 0 ] || { echo "!! Aborting: sources missing." >&2; exit 1; }
echo "   dropped $stripped comment lines from the .tex copies"

echo ">> Test-compiling in a scratch copy (no access to the repo) ..."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp -r "$OUT/." "$tmp/"
(
  cd "$tmp"
  for i in 1 2 3; do
    pdflatex -interaction=nonstopmode -halt-on-error main.tex >"pass$i.log" 2>&1 \
      || { echo "!! pdflatex failed on pass $i:" >&2; tail -30 "pass$i.log" >&2; exit 1; }
  done
)
pages="$(grep -oP 'Output written on main\.pdf \(\K[0-9]+' "$tmp/pass3.log")"
undef="$(grep -c 'Warning:.*undefined' "$tmp/pass3.log" || true)"
echo "   compiled $pages pages, $undef undefined reference/citation warnings"
[ "$undef" -eq 0 ] || { echo "!! Undefined references in the isolated build." >&2; exit 1; }

tar czf "$TAR" -C "$OUT" .
echo ">> Wrote $OUT/ and $TAR ($(du -h "$TAR" | cut -f1), $(submission_files | sort -u | wc -l) files)."
echo ">> Upload $TAR to arXiv. Bundled main.bbl; no bibtex run needed."
