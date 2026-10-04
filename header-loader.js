/* Injects the blog's own header (the _includes/header.html output, via /header-fragment.html)
   into course sub-sites, so every page shares one header. Optional data attributes on the
   script tag: data-repo, data-index add the course footer bar (notebook pages). */
(function () {
    const SELF = document.currentScript;
    const REPO = SELF && SELF.dataset.repo;
    const INDEX = SELF && SELF.dataset.index;
    const BASE = 'https://chizkidd.github.io';

    // 0. Theme: reuse the choice saved by the blog (same origin), else follow the OS
    try {
        const stored = localStorage.getItem('theme');
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        document.documentElement.setAttribute('data-theme', stored || (prefersDark ? 'dark' : 'light'));
    } catch (e) {}

    // 1. Google Analytics
    const gaScript = document.createElement('script');
    gaScript.async = true;
    gaScript.src = 'https://www.googletagmanager.com/gtag/js?id=G-WFFQCFF6S8';
    document.head.appendChild(gaScript);
    window.dataLayer = window.dataLayer || [];
    function gtag() { dataLayer.push(arguments); }
    gtag('js', new Date());
    gtag('config', 'G-WFFQCFF6S8');

    // 2. Shared styles: same files the blog uses, plus the course footer bar
    ['theme.css', 'header.css'].forEach(function (f) {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = BASE + '/css/' + f;
        document.head.appendChild(link);
    });
    const style = document.createElement('style');
    style.textContent = `
        #site-header-slot { min-height: 61px; }
        #course-footer {
            position: fixed; left: 0; right: 0; bottom: 0; height: 36px; z-index: 9998;
            background: var(--bg); border-top: 1px solid var(--border);
            font-family: Helvetica, Arial, sans-serif; font-size: 14px; font-weight: 300;
        }
        #course-footer .wrap {
            max-width: 800px; margin: 0 auto; padding: 0 30px; height: 100%;
            display: flex; justify-content: space-between; align-items: center;
        }
        #course-footer a { color: var(--muted) !important; text-decoration: none !important; }
        #course-footer a:hover { color: var(--text) !important; text-decoration: underline !important; }
        body.has-course-footer { padding-bottom: 36px; }
        @media screen and (max-width: 600px) { #course-footer .wrap { padding: 0 12px; } }
        @media print { #course-footer { display: none; } body.has-course-footer { padding-bottom: 0; } }
    `;
    document.head.appendChild(style);

    // 3. Header: fetched from the blog so it can never drift from the real one
    const FALLBACK = '<header class="site-header"><div class="wrap"><a class="site-title" href="' + BASE +
        '/">Chizoba Obasi blog</a></div></header>';

    function wireThemeToggle(slot) {
        const btn = slot.querySelector('#theme-toggle');
        if (!btn) return;
        btn.addEventListener('click', function () {
            const root = document.documentElement;
            const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            root.setAttribute('data-theme', next);
            try { localStorage.setItem('theme', next); } catch (e) {}
        });
    }

    function absolutize(slot) {
        slot.querySelectorAll('a[href^="/"], img[src^="/"]').forEach(function (el) {
            const attr = el.tagName === 'A' ? 'href' : 'src';
            el.setAttribute(attr, BASE + el.getAttribute(attr));
        });
    }

    function injectHeader() {
        const slot = document.createElement('div');
        slot.id = 'site-header-slot';
        // Break out of any page padding (nbconvert pages) so the header spans the full width
        const cs = getComputedStyle(document.body);
        const out = function (side) { return -(parseFloat(cs['margin' + side]) + parseFloat(cs['padding' + side])) + 'px'; };
        slot.style.margin = out('Top') + ' ' + out('Right') + ' 0 ' + out('Left');
        slot.style.minWidth = '0';
        document.body.prepend(slot);

        fetch(BASE + '/header-fragment.html')
            .then(function (r) { if (!r.ok) throw new Error(r.status); return r.text(); })
            .catch(function () { return FALLBACK; })
            .then(function (html) {
                slot.innerHTML = html;
                slot.style.minHeight = 'auto';
                absolutize(slot);
                wireThemeToggle(slot);
            });

        if (REPO) {
            const bar = document.createElement('footer');
            bar.id = 'course-footer';
            const wrap = document.createElement('div');
            wrap.className = 'wrap';
            const back = document.createElement('a');
            back.href = INDEX || BASE + '/';
            back.textContent = '← Course index';
            const repo = document.createElement('a');
            repo.href = 'https://github.com/' + REPO;
            repo.target = '_blank';
            repo.rel = 'noopener';
            repo.textContent = 'View Repository';
            wrap.append(back, repo);
            bar.appendChild(wrap);
            document.body.appendChild(bar);
            document.body.classList.add('has-course-footer');
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectHeader);
    } else {
        injectHeader();
    }
})();
