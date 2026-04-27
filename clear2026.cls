\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{clear2026}[2024/07/20 v1.0 CLeaR 2025 Proceedings]
\newcommand{\todo}{\color{red}}

\@ifundefined{ifanonsubmission}{
  \newif\ifanonsubmission
  \anonsubmissionfalse
}{}

\@ifundefined{iffinalsubmission}{
  \newif\iffinalsubmission
  \finalsubmissionfalse
}{}

\DeclareOption{anon}{\anonsubmissiontrue}
\DeclareOption{final}{\finalsubmissiontrue}
\ProcessOptions\relax

\LoadClass[pmlr]{jmlr}

\jmlrvolume{275}  % will need to be updated for the final submissions
\jmlryear{2026}
\jmlrproceedings{PMLR}{Proceedings of Machine Learning Research}

\ifanonsubmission
 \newcommand{\clearauthor}[1]{}
 \author{Author names withheld}
 \editor{Under Review for CLeaR 2026}
 \jmlrworkshop{5th Conference on Causal Learning and Reasoning}
 \renewcommand{\acks}[1]{}
\else
 \newcommand{\clearauthor}[1]{\author{#1}}
 \iffinalsubmission
  \editors{Bijan Mazaheri and Niels Richard Hansen}
  \jmlrworkshop{5th Conference on Causal Learning and Reasoning}
 \fi
\fi



