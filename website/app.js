const menuButton = document.querySelector('[data-menu-button]');
const mobileNav = document.querySelector('[data-mobile-nav]');

menuButton?.addEventListener('click', () => {
  const willOpen = mobileNav.hasAttribute('hidden');
  mobileNav.toggleAttribute('hidden', !willOpen);
  menuButton.setAttribute('aria-expanded', String(willOpen));
  menuButton.setAttribute('aria-label', willOpen ? 'Close navigation' : 'Open navigation');
});

mobileNav?.addEventListener('click', (event) => {
  if (!event.target.closest('a')) return;
  mobileNav.setAttribute('hidden', '');
  menuButton?.setAttribute('aria-expanded', 'false');
  menuButton?.setAttribute('aria-label', 'Open navigation');
});

const filterButtons = [...document.querySelectorAll('[data-filter]')];
const modelCards = [...document.querySelectorAll('[data-category]')];

filterButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const selected = button.dataset.filter;
    filterButtons.forEach((candidate) => candidate.classList.toggle('is-active', candidate === button));
    modelCards.forEach((card) => {
      card.hidden = selected !== 'all' && card.dataset.category !== selected;
    });
  });
});
