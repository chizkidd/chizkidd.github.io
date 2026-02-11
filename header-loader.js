// https://chizkidd.github.io/header-loader.js
(function() {
    const headerHTML = `
    <header style="border-top: 5px solid #333; border-bottom: 1px solid #e8e8e8; background: white; padding: 15px 0; font-family: Helvetica, Arial, sans-serif;">
        <div style="max-width: 800px; margin: 0 auto; padding: 0 30px; display: flex; justify-content: space-between; align-items: center;">
            <a href="https://chizkidd.github.io" style="color: #333; font-size: 26px; text-decoration: none; font-weight: bold; letter-spacing: -1px;">chizkidd</a>
            <nav>
                <a href="https://chizkidd.github.io/fastai/" style="color: #2a7ae2; text-decoration: none; margin-left: 20px;">Fast.ai</a>
                <a href="https://chizkidd.github.io/Karpathy-Neural-Networks-Zero-to-Hero/" style="color: #2a7ae2; text-decoration: none; margin-left: 20px;">Karpathy Course</a>
            </nav>
        </div>
    </header>`;

    function injectHeader() {
        const div = document.createElement('div');
        div.innerHTML = headerHTML;
        document.body.prepend(div);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectHeader);
    } else {
        injectHeader();
    }
})();