const menuBtn = document.querySelector('.menu-btn');
const nav = document.querySelector('#nav');

menuBtn.addEventListener('click', () => {
  nav.style.display = nav.style.display === 'flex' ? 'none' : 'flex';
});

document.querySelectorAll('#nav a').forEach(link => {
  link.addEventListener('click', () => {
    if (window.innerWidth <= 700) {
      nav.style.display = 'none';
    }
  });
});

document.querySelector('#year').textContent = new Date().getFullYear();


const mouseGlow = document.createElement('div');

mouseGlow.classList.add('mouse-glow');

document.body.appendChild(mouseGlow);

document.addEventListener('mousemove', (event) => {
  mouseGlow.style.left = `${event.clientX}px`;
  mouseGlow.style.top = `${event.clientY}px`;
});

const particleCount = 35;

for (let i = 0; i < particleCount; i++) {

  const particle = document.createElement('div');

  particle.classList.add('particle');

  particle.style.left = `${Math.random() * 100}vw`;
  particle.style.top = `${Math.random() * 100}vh`;

  particle.style.animationDuration =
    `${8 + Math.random() * 12}s`;

  particle.style.animationDelay =
    `${Math.random() * 10}s`;

  const size = 2 + Math.random() * 3;

  particle.style.width = `${size}px`;
  particle.style.height = `${size}px`;

  document.body.appendChild(particle);
}
