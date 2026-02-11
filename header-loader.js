// https://chizkidd.github.io/shared-header.js
(function() {
    // 1. Google Analytics Injection
    const gaScript = document.createElement('script');
    gaScript.async = true;
    gaScript.src = 'https://www.googletagmanager.com/gtag/js?id=G-WFFQCFF6S8';
    document.head.appendChild(gaScript);

    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'G-WFFQCFF6S8');

    // 2. Header HTML (Matching your main.css exactly)
    const headerHTML = `
    <header class="site-header" style="border-top: 5px solid #333; border-bottom: 1px solid #e8e8e8; min-height: 56px; background-color: white; font-family: Helvetica, Arial, sans-serif;">
        <div class="wrap" style="max-width: 800px; margin: 0 auto; padding: 0 30px; display: flex; justify-content: space-between; align-items: center; height: 56px;">
            <a class="site-title" href="https://chizkidd.github.io/" style="color: #333; font-size: 26px; letter-spacing: -1px; text-decoration: none; font-weight: 400; line-height: 56px;">Chizoba Obasi blog</a>
            <nav class="site-nav" style="line-height: 56px;">
                <a href="https://chizkidd.github.io/fastai/" style="color: #2a7ae2; text-decoration: none; margin-left: 20px; font-weight: 300; font-size: 16px;">Fast.ai</a>
                <a href="https://chizkidd.github.io/Karpathy-Neural-Networks-Zero-to-Hero/" style="color: #2a7ae2; text-decoration: none; margin-left: 20px; font-weight: 300; font-size: 16px;">Karpathy Course</a>
            </nav>
        </div>
    </header>`;

    function injectHeader() {
        const headerContainer = document.createElement('div');
        headerContainer.innerHTML = headerHTML;
        document.body.prepend(headerContainer);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectHeader);
    } else {
        injectHeader();
    }
})();