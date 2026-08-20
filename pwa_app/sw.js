const CACHE_NAME = 'kalpana-pwa-v5-live';

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            console.log('[PWA SW] Purging stale cache:', key);
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  // Pass dynamic and API requests straight to network
  if (
    event.request.url.includes('wikipedia.org') ||
    event.request.url.includes('huggingface.co') ||
    event.request.url.includes('cdn.jsdelivr.net') ||
    event.request.url.includes('/api/')
  ) {
    event.respondWith(fetch(event.request));
    return;
  }

  // Network-First with Cache fallback for app assets
  event.respondWith(
    fetch(event.request)
      .then((networkResponse) => {
        if (networkResponse && networkResponse.status === 200 && event.request.method === 'GET') {
          const responseClone = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseClone);
          });
        }
        return networkResponse;
      })
      .catch(() => caches.match(event.request))
  );
});

