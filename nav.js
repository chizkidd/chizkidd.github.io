/* Tap/click menu for the site header on narrow screens (wide screens show the links inline).
   Runs on the blog automatically; header-loader.js calls window.initHeaderMenu(root) after
   it injects the header into course pages. */
(function () {
    function init(root) {
        const nav = root.querySelector('.site-nav');
        const btn = nav && nav.querySelector('.menu-icon');
        if (!btn || btn.dataset.ready) return;
        btn.dataset.ready = '1';

        function setOpen(open) {
            nav.classList.toggle('open', open);
            btn.setAttribute('aria-expanded', open ? 'true' : 'false');
        }

        btn.addEventListener('click', function (e) {
            e.stopPropagation();
            setOpen(!nav.classList.contains('open'));
        });
        document.addEventListener('click', function (e) {
            if (!nav.contains(e.target)) setOpen(false);
        });
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && nav.classList.contains('open')) { setOpen(false); btn.focus(); }
        });
        window.matchMedia('(min-width: 641px)').addEventListener('change', function (e) {
            if (e.matches) setOpen(false);
        });
    }

    window.initHeaderMenu = init;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () { init(document); });
    } else {
        init(document);
    }
})();
