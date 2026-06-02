%%
%% This is file `edrfcm.cls',

\NeedsTeXFormat{LaTeX2e}[1994/06/01]
\ProvidesClass{edrfcm}[2026/03/28 EDRFCM document class updated for Madrid 2026]
\typeout{TYPESETTING in EDRFCM symposium format ...}
\typeout{}

\LoadClass[oneside,twocolumn,letterpaper]{article} %

\RequirePackage{edrfcm}     % edrfcm.sty provided in the filepackage
\RequirePackage{graphics}   % Standard package
\RequirePackage{graphicx}   % Standard package
\RequirePackage[margin=23.1mm]{geometry}
%%%%%%%%%%%%%%%%%%%%
% General layout
%
 
\setlength{\topmargin}{-5mm}
\setlength{\headsep}{0mm}
\setlength{\footskip}{32pt}
\setlength{\textheight}{240.0mm}
\setlength{\textwidth}{169.7mm}
\setlength{\columnsep}{9.53mm}
\renewcommand{\baselinestretch}{1.05}

\let\@date\@empty
\doublehyphendemerits=100000
\setlength\hfuzz{3pt}    % reduce the quantity of \overfull warnings

%%%%%%%%%%%%%%%%%%
% Modify floatings
%
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.9}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.8}
\renewcommand{\dbltopfraction}{\floatpagefraction}
\renewcommand{\dblfloatpagefraction}{\floatpagefraction}

\setcounter{topnumber}{10}
\setcounter{dbltopnumber}{\value{topnumber}}
\setcounter{bottomnumber}{\value{topnumber}}
\setcounter{totalnumber}{\value{topnumber}}
\addtocounter{totalnumber}{\value{bottomnumber}}

\setlength{\floatsep}        { 5pt plus 2pt minus 2pt}
\setlength{\textfloatsep}    { 5pt plus 2pt minus 3pt}
\setlength{\intextsep}       { 5pt plus 2pt minus 2pt}
\setlength{\dblfloatsep}     {\floatsep}
\setlength{\dbltextfloatsep} {\textfloatsep}

\renewcommand{\and}{\\*[-8pt]}

%%%%%%%%%%%%%%%%%%%%%
% Title
%

\def\@corrauthor{}
\newcommand{\corrauthor}[1]{\gdef\@corrauthor{#1}}

\def\@keywords{}
\newcommand{\keywords}[1]{\gdef\@keywords{#1}}

\def\@maketitle{%
  \newpage
\begin{flushright}
{\bf
ERCOFTAC\\
European Drag Reduction and Flow Control Meeting -- EDRFCM 2026 \\
September 8$^{\rm{th}}$--11$^{\rm{th}}$, 2026, Madrid, Spain \\
Universidad Carlos III de Madrid (Puerta de Toledo) \\ \vspace{5mm}}
\end{flushright}

  \begin{center}%
  \let \footnote \thanks
  \parbox{150mm}{\centering
  \setlength{\baselineskip}{17pt}
   {\Large{\sffamily\bfseries\MakeUppercase
   {\@title}}}}
   \par%
    \vskip 10pt%
    {\large{\textsf\@author}}
    \par%
    \ifx\@corrauthor\@empty\else
      \vskip 0.6em%
      {\normalsize\textsf{\@corrauthor}}\par%
    \fi
    \vskip 1em%
  \end{center}%
  \par%
  \ifx\@keywords\@empty\else
    \vskip 0.2em%
    \noindent\textsf{\textbf{Keywords:}} \textsf{\@keywords}\par%
  \fi
  \vskip 1em%
}

%%%%%%%%%%%%%%%%%%%%%%%
% Sectioning
%

\setcounter{secnumdepth}{-2}% instead of having to use the *'d
                            % section commands

\renewcommand{\section}[1]{
\par
\vspace{16pt}
\noindent\textsf{\textbf{\MakeUppercase{#1}}}
\par
\vspace{2pt}
}
\renewcommand{\subsection}[1]{
\par
\vspace{16pt}
\noindent\textsf{\textbf{#1}}
\par
}
\renewcommand{\subsubsection}[1]{%
\par
\vspace{16pt}
\textsf{\textbf{#1.}}}

\renewcommand\paragraph{\@startsection{paragraph}{4}%
  {\parindent}%
  {0.3ex}%
  {-1em}%
  {\normalfont\normalsize\sffamily}}
\renewcommand\subparagraph{\@startsection{subparagraph}{5}%
  {\parindent}%
  {0.3ex}%
  {-1em}%
  {\normalfont\normalsize\itshape}}
\listfiles
\endinput
%%
%% End of file
