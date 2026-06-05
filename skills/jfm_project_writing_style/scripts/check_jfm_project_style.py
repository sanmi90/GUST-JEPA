#!/usr/bin/env python3
"""Lightweight JFM project style gate for vortex-jepa manuscripts.

Usage:
    python scripts/check_jfm_project_style.py paper
    python scripts/check_jfm_project_style.py paper/main.tex --pdf paper/main.pdf

The checker is intentionally conservative: it reports warnings for review and exits
with code 1 only for high-risk submission artifacts and known contradiction phrases.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Finding:
    severity: str
    path: pathlib.Path
    line: int
    message: str
    excerpt: str


HIGH_RISK_PATTERNS = [
    (r"Abstract must", "JFM template artifact visible"),
    (r"Focus on Fluids", "JFM template artifact visible"),
    (r"Rapids articles", "JFM template artifact visible"),
    (r"all scales (?:are|of .* are) resolved", "unqualified DNS/all-scales claim while Table 1 may be pending"),
    (r"single shared autoregressive predictor", "ambiguous shared-weight predictor wording"),
    (r"shared predictor", "ambiguous shared-weight predictor wording"),
    (r"same autoregressive predictor", "ambiguous shared-weight predictor wording"),
    (r"same probes", "ambiguous probe-weight wording"),
    (r"controller would plan against", "overstrong controller wording"),
    (r"AdaLN-Zero on \(c,", "phase appears as default predictor conditioning"),
    (r"within-encounter phase enter only the predictor", "phase appears as default predictor conditioning"),
    (r"impact frame, with no latent input", "conditioning floor may be described at impact frame instead of H=16"),
    (r"attached to the frozen encoder", "auxiliary heads may be confused with post-hoc decoder"),
]

STYLE_PATTERNS = [
    (r"\bisometry\b|\bisometric\b", "avoid mathematical isometry language unless proving it"),
    (r"OT[- ]geodesic|\bgeodesic\b", "avoid geodesic wording unless a geodesic path is computed"),
    (r"\binterventional\b|\bcausal\b|\bcounterfactual\b", "verify causal/controller boundary is explicitly negative"),
    (r"\\paragraph\{|\\subparagraph\{", "avoid run-in paragraph headings in main text"),
    (r"^\s*\\textbf\{[^}]+\}\.?", "avoid bold run-in paragraph starts in main text"),
    (r"^\s*(Distance|Topology|Transport geometry|Latent drift|Conditioning-only floor)\.\s", "convert checklist labels to flowing prose"),
    (r"—", "em-dash character found; project convention is no em-dashes"),
    (r"\bdecisive\b", "check whether claim is stronger than evidence"),
]


def iter_text_files(target: pathlib.Path):
    if target.is_file():
        if target.suffix.lower() in {".tex", ".txt", ".md"}:
            yield target
        return
    for ext in ("*.tex", "*.bib", "*.md"):
        for p in target.rglob(ext):
            if "_v2_md_archive" in p.parts:
                continue
            yield p


def scan_text(path: pathlib.Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        for pat, msg in HIGH_RISK_PATTERNS:
            if re.search(pat, line, flags=re.IGNORECASE):
                findings.append(Finding("ERROR", path, i, msg, line.strip()))
        for pat, msg in STYLE_PATTERNS:
            if re.search(pat, line, flags=re.IGNORECASE):
                findings.append(Finding("WARN", path, i, msg, line.strip()))
    return findings


def extract_pdf_text(pdf: pathlib.Path) -> str:
    if not pdf.exists():
        return ""
    if shutil.which("pdftotext") is None:
        return ""
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except Exception:
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def abstract_word_count_from_tex(text: str) -> int | None:
    m = re.search(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", text, flags=re.S)
    if not m:
        return None
    body = m.group(1)
    body = re.sub(r"%.*", " ", body)
    body = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^]]*\])?(?:\{[^{}]*\})?", " ", body)
    body = re.sub(r"[{}$^_\\]", " ", body)
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", body)
    return len(words)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=pathlib.Path, help="paper directory or TeX file")
    ap.add_argument("--pdf", type=pathlib.Path, default=None, help="compiled PDF to scan with pdftotext")
    args = ap.parse_args(argv)

    findings: list[Finding] = []
    tex_texts: list[str] = []
    for p in iter_text_files(args.target):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            print(f"WARN: could not read {p}: {exc}", file=sys.stderr)
            continue
        tex_texts.append(text)
        findings.extend(scan_text(p, text))

    if args.pdf:
        pdf_text = extract_pdf_text(args.pdf)
        if pdf_text:
            findings.extend(scan_text(args.pdf, pdf_text))

    for text in tex_texts:
        wc = abstract_word_count_from_tex(text)
        if wc is not None:
            if wc > 250:
                findings.append(Finding("ERROR", args.target, 0, f"abstract word count exceeds 250 ({wc})", ""))
            break

    if not findings:
        print("JFM project style gate: no findings.")
        return 0

    for f in findings:
        loc = f"{f.path}:{f.line}" if f.line else str(f.path)
        print(f"{f.severity}: {loc}: {f.message}")
        if f.excerpt:
            print(f"       {f.excerpt[:220]}")

    return 1 if any(f.severity == "ERROR" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
