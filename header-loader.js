(function() {
    // Course pages pass data-repo / data-index on this script tag to get the footer bar
    const SELF = document.currentScript;
    const REPO = SELF && SELF.dataset.repo;
    const INDEX = SELF && SELF.dataset.index;

    // 0. Theme: reuse the choice saved by the main site (same origin), else follow the OS
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
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'G-WFFQCFF6S8');

    // 2. Inject CSS for Hover States and Dropdown Layout
    const style = document.createElement('style');
    style.innerHTML = `
        :root {
            --hl-bg: #f6f7f9; --hl-surface: #ffffff; --hl-text: #17202c; --hl-muted: #5a6576;
            --hl-border: #dae0e7; --hl-hover: #eef1f5; --hl-bar: #17202c;
        }
        html[data-theme="dark"] {
            --hl-bg: #0f141b; --hl-surface: #161c25; --hl-text: #e3e8ef; --hl-muted: #93a0b2;
            --hl-border: #2a3341; --hl-hover: #1f2733; --hl-bar: #8fa8ff;
        }
        /* Main Link Styling */
        #nav-trigger a { 
            display: block; 
            padding: 10px 15px; 
            border-bottom: 1px solid var(--hl-border); 
            color: var(--hl-muted) !important; 
            text-decoration: none !important;
            font-family: Helvetica, Arial, sans-serif;
            font-size: 16px;
            font-weight: 300;
        }
        #nav-trigger a:last-child { border-bottom: none; }
        
        /* Hover Effect for Links */
        #nav-trigger a:hover { 
            background-color: var(--hl-hover); 
            color: var(--hl-text) !important; 
            text-decoration: underline !important; 
        }

        /* HOVER LOGIC: Shows menu when mouse is over the nav area */
        .site-nav:hover #nav-trigger {
            display: flex !important;
        }
        
        .site-title:hover { text-decoration: none !important; }

        /* Course footer bar (notebook pages) */
        #course-footer {
            position: fixed; left: 0; right: 0; bottom: 0; height: 36px; z-index: 9998;
            background: var(--hl-bg); border-top: 1px solid var(--hl-border);
            font-family: Helvetica, Arial, sans-serif; font-size: 14px; font-weight: 300;
        }
        #course-footer .wrap {
            max-width: 800px; margin: 0 auto; padding: 0 30px; height: 100%;
            display: flex; justify-content: space-between; align-items: center;
        }
        #course-footer a { color: var(--hl-muted) !important; text-decoration: none !important; }
        #course-footer a:hover { color: var(--hl-text) !important; text-decoration: underline !important; }
        body.has-course-footer { padding-bottom: 36px; }
        @media screen and (max-width: 600px) { #course-footer .wrap { padding: 0 12px; } }
        @media print { #course-footer { display: none; } body.has-course-footer { padding-bottom: 0; } }
    `;
    document.head.appendChild(style);

    // 3. Header HTML
    const headerHTML = `
    <header class="site-header" style="border-top: 4px solid var(--hl-bar); border-bottom: 1px solid var(--hl-border); min-height: 56px; background-color: var(--hl-bg); font-family: Helvetica, Arial, sans-serif;">
        <div class="wrap" style="max-width: 800px; margin: 0 auto; padding: 0 30px; display: flex; justify-content: space-between; align-items: center; height: 56px; position: relative;">
            <div style="display: flex; align-items: center;">
                <a href="https://chizkidd.github.io/" style="margin-right: 10px; display: flex; align-items: center;">
                    <img src="https://chizkidd.github.io/assets/rssicon.svg" alt="RSS" style="width: 24px; height: 24px;">
                </a>
                <a class="site-title" href="https://chizkidd.github.io/" style="color: var(--hl-text); font-size: 26px; letter-spacing: -1px; text-decoration: none; font-weight: 400;">Chizoba Obasi blog</a>
            </div>
            
            <nav class="site-nav" style="position: relative; padding: 10px 0;">
                <div id="menu-toggle" style="color: var(--hl-text); cursor: pointer; padding: 10px; display: block;">
                    <svg viewBox="0 0 18 15" width="18px" height="15px"><path fill="currentColor" d="M18,1.484c0,0.82-0.665,1.484-1.484,1.484H1.484C0.665,2.969,0,2.304,0,1.484l0,0C0,0.665,0.665,0,1.484,0 h15.031C17.335,0,18,0.665,18,1.484L18,1.484z M18,7.516C18,8.335,17.335,9,16.516,9H1.484C0.665,9,0,8.335,0,7.516l0,0 c0-0.82,0.665-1.484,1.484-1.484h15.031C17.335,6.031,18,6.696,18,7.516L18,7.516z M18,13.516C18,14.335,17.335,15,16.516,15H1.484 C0.665,15,0,14.335,0,13.516l0,0c0-0.82,0.665-1.483,1.484-1.483h15.031C17.335,12.031,18,12.695,18,13.516L18,13.516z"/></svg>
                </div>
                <div id="nav-trigger" style="display: none; position: absolute; right: 0; top: 45px; background: var(--hl-surface); border: 1px solid var(--hl-border); border-radius: 5px; min-width: 160px; z-index: 9999; box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden; flex-direction: column;">
                    <a href="https://chizkidd.github.io/about/">About</a>
                    <a href="https://chizkidd.github.io/courses/">Courses</a>
                    <a href="https://chizkidd.github.io/projects/">Projects</a>
                </div>
            </nav>
        </div>
    </header>`;

    function injectHeader() {
        const container = document.createElement('div');
        container.innerHTML = headerHTML;
        document.body.prepend(container);
        
        const toggle = document.getElementById('menu-toggle');
        const trigger = document.getElementById('nav-trigger');
        
        if (REPO) {
            const bar = document.createElement('footer');
            bar.id = 'course-footer';
            const wrap = document.createElement('div');
            wrap.className = 'wrap';
            const back = document.createElement('a');
            back.href = INDEX || '/';
            back.textContent = '\u2190 Course index';
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

        // Manual Toggle (for mobile/touch where hover doesn't exist)
        toggle.onclick = (e) => {
            e.stopPropagation();
            const isHidden = trigger.style.display === 'none' || trigger.style.display === '';
            trigger.style.display = isHidden ? 'flex' : 'none';
        };

        // Close menu when clicking elsewhere
        document.onclick = () => {
            trigger.style.display = 'none';
        };
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectHeader);
    } else {
        injectHeader();
    }
})();