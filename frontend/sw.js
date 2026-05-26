const CACHE = 'mexc-trader-v1';
const PRECACHE = ['/', '/ml_details.html'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(PRECACHE)));
  self.skipWaiting();
});
self.addEventListener('activate', e => e.waitUntil(clients.claim()));
self.addEventListener('fetch', e => {
  if (e.request.mode === 'navigate') {
    e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
  }
});
