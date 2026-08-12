(function () {
  var btn = document.getElementById('scroll-to-top');
  if (!btn) return;

  window.addEventListener('scroll', function () {
    btn.style.display = window.scrollY > 300 ? 'block' : 'none';
  });

  btn.addEventListener('click', function () {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
})();