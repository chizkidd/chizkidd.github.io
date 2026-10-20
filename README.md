# chizkidd.github.io

Source for [Chizoba Obasi's blog](https://chizkidd.github.io/): posts on ML, DL, RL and LLMs, plus course notebooks and projects. It is a [Jekyll](https://jekyllrb.com/) site served by GitHub Pages from `main`.

## Layout

| Path | What it is |
|---|---|
| `_posts/` | Blog posts (Markdown). Front matter: `title`, `date`, `excerpt`, optional `mathjax: true`, `comments: true`, `image` (for link previews). |
| `index.html`, `about.md`, `courses.html`, `projects.html`, `posts.html` | Pages. A page with `includelink: true` shows up in the header menu. |
| `_layouts/`, `_includes/` | Page templates and partials (header, footer, comments, MathJax, head tags). |
| `css/theme.css` | Light and dark colour variables, shared by the blog, the header and the course landing pages. |
| `css/header.css` | Header styles, shared with the course sites. |
| `css/main.css` | Everything else. |
| `nav.js`, `dark-mode.js`, `scroll-to-top.js` | Header menu, theme toggle, back-to-top button. |
| `header-fragment.html`, `header-loader.js` | The blog header, fetched by the course sites so they always match. |
| `404.html`, `sitemap.xml`, `robots.txt` | Not-found page and search-engine files. |

## Course sites

The course repos (fastai, Karpathy, HuggingFace RL, deep-learning-models, transformers, Nielsen) are built by one shared workflow here, `.github/workflows/notebooks-site.yml`, with `scripts/build_course_index.py` and `assets/course-index.css` for the landing pages. Each course repo only has a small caller workflow and a `course.yml`.

- Changes to the header, CSS or loader go live for every course site with the normal Pages build.
- Changes to `notebooks-site.yml` or `scripts/**` trigger `redeploy-courses.yml`, which re-runs the deploy in all six repos. It needs the repo secret `COURSES_DEPLOY_TOKEN` (fine-grained token, Actions read/write on the six repos). `token-expiry-check.yml` opens an issue before that token expires.

## Run locally

```sh
gem install jekyll
jekyll serve
```

Then open http://localhost:4000/.
