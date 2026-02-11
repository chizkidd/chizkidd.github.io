(function() {
    // 1. Google Analytics
    const gaScript = document.createElement('script');
    gaScript.async = true;
    gaScript.src = 'https://www.googletagmanager.com/gtag/js?id=G-WFFQCFF6S8';
    document.head.appendChild(gaScript);
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'G-WFFQCFF6S8');

    // 2. CSS for Theme Consistency and Hover States
    const style = document.createElement('style');
    style.innerHTML = `
        /* Matches Minima theme hover and dropdown logic */
        .site-nav .menu-icon { display: none; }
        .site-nav .trigger { display: block; }

        @media screen and (max-width: 600px) {
            .site-nav {
                position: absolute;
                top: 9px;
                right: 15px;
                background-color: #fff;
                border: 1px solid #e8e8e8;
                border-radius: 5px;
                text-align: right;
            }
            .site-nav .menu-icon {
                display: block;
                float: right;
                width: 36px;
                height: 36px;
                line-height: 33px;
                text-align: center;
                cursor: pointer;
            }
            .site-nav .trigger {
                display: none;
                padding-bottom: 5px;
                clear: both;
            }
            /* The 'Hover' fix for mobile: shows menu when hovering the nav container */
            .site-nav:hover .trigger {
                display: block;
            }
            .site-nav .page-link {
                display: block;
                padding: 5px 10px;
                margin-left: 20px;
                color: #828282 !important;
                text-decoration: none;
                line-height: 1.5;
            }
            .site-nav .page-link:hover {
                color: #333 !important;
                background-color: #f9f9f9;
            }
        }
    `;
    document.head.appendChild(style);

    // 3. Replicating the Blog's exact HTML structure
    const headerHTML = `
    <header class="site-header" style="border-top: 5px solid #333; border-bottom: 1px solid #e8e8e8; min-height: 56px; background-color: white; font-family: Helvetica, Arial, sans-serif;">
        <div class="wrapper" style="max-width: 800px; margin: 0 auto; padding: 0 30px; display: flex; justify-content: space-between; align-items: center; height: 56px; position: relative;">
            <div style="display: flex; align-items: center;">
                <a href="https://chizkidd.github.io/" style="margin-right: 10px; display: flex; align-items: center;">
                    <img src="https://chizkidd.github.io/assets/rssicon.svg" alt="RSS" style="width: 24px; height: 24px;">
                </a>
                <a class="site-title" href="https://chizkidd.github.io/" style="color: #333; font-size: 26px; letter-spacing: -1px; text-decoration: none; font-weight: 400;">Chizoba Obasi blog</a>
            </div>
            
            <nav class="site-nav">
                <span class="menu-icon">
                    <svg viewBox="0 0 18 15" width="18px" height="15px">
                        <path fill="#424242" d="M18,1.484c0,0.82-0.665,1.484-1.484,1.484H1.484C0.665,2.969,0,2.304,0,1.484l0,0C0,0.665,0.665,0,1.484,0 h15.031C17.335,0,18,0.665,18,1.484L18,1.484z M18,7.516C18,8.335,17.335,9,16.516,9H1.484C0.665,9,0,8.335,0,7.516l0,0 c0-0.82,0.665-1.484,1.484-1.484h15.031C17.335,6.031,18,6.696,18,7.516L18,7.516z M18,13.516C18,14.335,17.335,15,16.516,15H1.484 C0.665,15,0,14.335,0,13.516l0,0c0-0.82,0.665-1.483,1.484-1.483h15.031C17.335,12.031,18,12.695,18,13.516L18,13.516z"/>
                    </svg>
                </span>

                <div class="trigger">
                    <a class="page-link" href="https://chizkidd.github.io/about/">About</a>
                    <a class="page-link" href="https://chizkidd.github.io/fastai/">Fast.ai</a>
                    <a class="page-link" href="https://chizkidd.github.io/Karpathy-Neural-Networks-Zero-to-Hero/">Karpathy Course</a>
                    <a class="page-link" href="https://chizkidd.github.io/games/">Games</a>
                </div>
            </nav>
        </div>
    </header>`;

    function injectHeader() {
        const container = document.createElement('div');
        container.innerHTML = headerHTML;
        document.body.prepend(container);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectHeader);
    } else {
        injectHeader();
    }
})();