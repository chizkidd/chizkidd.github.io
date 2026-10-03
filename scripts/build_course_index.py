#!/usr/bin/env python3
"""Render the landing page (public/index.html) for a notebook course site.

Run from the root of a course repo after the notebooks were converted to HTML
under ./public (same relative layout as the .ipynb files).

Optional course.yml:
  title:    page heading (default: first "# " heading of README.md, else repo name)
  credit:   {text: ..., url: ...}   shown in the footer as "Course by <text>"
  sections: ordered list of {title, folder?, match?}
            folder: exact top-level directory of the notebook
            match:  fnmatch pattern on the lowercase file stem; prefix "!" negates
            first matching section wins; unmatched notebooks go to a final "Other" section
Without sections, notebooks are grouped by folder.
"""
import argparse
import datetime
import fnmatch
import html
import os
import re
import sys

import yaml

CSS_URL = "https://chizkidd.github.io/assets/course-index.css"
HEADER_URL = "https://chizkidd.github.io/header-loader.js"


def natural_key(text):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", text)]


def prettify(name, strip_prefix=False):
    if strip_prefix:
        name = re.sub(r"^\d+[_\-\s]*", "", name) or name
    words = re.sub(r"[_\-]+", " ", name).split()
    return " ".join(w[0].upper() + w[1:] if w[:1].islower() else w for w in words)


def first_heading(readme):
    try:
        with open(readme, encoding="utf-8") as f:
            for line in f:
                if line.startswith("# "):
                    return line[2:].strip()
    except OSError:
        pass
    return None


def matches(pattern, stem):
    negate = pattern.startswith("!")
    ok = fnmatch.fnmatch(stem.lower(), pattern.lstrip("!").lower())
    return not ok if negate else ok


def collect(public):
    notebooks = []
    for root, _, files in os.walk(public):
        for fn in files:
            if not fn.endswith(".html"):
                continue
            rel = os.path.relpath(os.path.join(root, fn), public).replace(os.sep, "/")
            if rel == "index.html":
                continue
            folder = os.path.dirname(rel)
            notebooks.append({"file": rel, "stem": fn[:-5], "folder": folder})
    notebooks.sort(key=lambda n: natural_key(n["file"]))
    return notebooks


def build_sections(notebooks, cfg):
    specs = cfg.get("sections")
    if not specs:
        order, groups = [], {}
        for nb in notebooks:
            key = nb["folder"]
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(nb)
        order.sort(key=natural_key)
        return [(prettify(k.replace("/", " / "), True) if k else None, groups[k]) for k in order]

    sections = [(s["title"], []) for s in specs]
    other = []
    for nb in notebooks:
        for (title, items), spec in zip(sections, specs):
            top = nb["folder"].split("/")[0]
            if "folder" in spec and spec["folder"] != top:
                continue
            if "match" in spec and not matches(spec["match"], nb["stem"]):
                continue
            items.append(nb)
            break
        else:
            other.append(nb)
    if other:
        sections.append(("Other", other))
    return [(t, i) for t, i in sections if i]


def render(title, credit, repo, sections, updated):
    body = []
    for heading, items in sections:
        if heading:
            body.append(f'      <h2 class="section-heading">{html.escape(heading)}</h2>')
        body.append('      <ul class="home-nav-list">')
        for nb in items:
            href = html.escape(nb["file"], quote=True)
            body.append(f'        <li><a href="{href}">{html.escape(prettify(nb["stem"]))}</a></li>')
        body.append("      </ul>")

    footer = []
    if credit and credit.get("text"):
        text = html.escape(credit["text"])
        link = f'<a href="{html.escape(credit["url"], quote=True)}" target="_blank">{text}</a>' if credit.get("url") else text
        footer.append(f"Course by {link}")
    if repo:
        footer.append(f'<a href="https://github.com/{html.escape(repo, quote=True)}" target="_blank">View Repository</a>')
    footer_line = " • ".join(footer)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)} | Chizoba Obasi</title>
  <script>
    try {{
      var t = localStorage.getItem('theme');
      var d = window.matchMedia('(prefers-color-scheme: dark)').matches;
      document.documentElement.setAttribute('data-theme', t || (d ? 'dark' : 'light'));
    }} catch (e) {{}}
  </script>
  <link rel="stylesheet" href="{CSS_URL}">
</head>
<body>
  <script src="{HEADER_URL}"></script>
  <div class="page-content">
    <div class="wrap">
      <h1 class="header-matched-heading">{html.escape(title)}</h1>
{chr(10).join(body)}
    </div>
  </div>
  <footer class="site-footer">
    <div class="wrap">
      <p>{footer_line}</p>
      <p class="last-updated">Last updated: {updated}</p>
    </div>
  </footer>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default="public")
    ap.add_argument("--config", default="course.yml")
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    ap.add_argument("--readme", default="README.md")
    args = ap.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    notebooks = collect(args.public)
    if not notebooks:
        sys.exit(f"no .html notebooks found under {args.public}")

    title = cfg.get("title") or first_heading(args.readme) or (args.repo.split("/")[-1] if args.repo else "Notebooks")
    updated = datetime.date.today().strftime("%B %d, %Y")
    page = render(title, cfg.get("credit"), args.repo, build_sections(notebooks, cfg), updated)
    with open(os.path.join(args.public, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {args.public}/index.html ({len(notebooks)} notebooks)")


if __name__ == "__main__":
    main()
