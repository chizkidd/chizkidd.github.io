(function () {
  var btn = document.getElementById('scroll-to-top');
  if (!btn) return;

  function update() {
    btn.classList.toggle('visible', window.scrollY > 300);
  }

  window.addEventListener('scroll', update, { passive: true });
  update();

  btn.addEventListener('click', function () {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
})();
