# RealImage privacy boundary

RealImage examines images displayed on ordinary `http` and `https` pages so it
can show an optional local AI-likelihood label. The classifier, preprocessing,
and evidence checks run inside the Chrome extension. RealImage has no
telemetry, analytics, inference API, account system, advertising SDK, or image
upload code.

## Data handled

- The content script reads an image's current URL, rendered dimensions, and
  pixels when a Blob fallback is required.
- The extension fetches public cross-origin image URLs with credentials omitted.
  It never returns fetched bytes or response headers to the webpage.
- Encoded bytes and decoded pixels live only long enough to preprocess and score
  an image. RealImage does not write raw images to extension storage.
- A bounded worker-memory cache stores only a SHA-256-derived lookup key and the
  small score/evidence result. It disappears when Chrome stops the Manifest V3
  worker. Chrome or the visited site may independently retain normal HTTP-cache
  entries; RealImage does not control that browser cache.
- `chrome.storage.local` contains user settings only.

## Network and offline behavior

The model, JavaScript, WebAssembly runtime, and UI assets are packaged with the
extension. Image requests are limited to the page content being analyzed. There
is no model CDN or inference-related network request after installation, and the
detector continues to initialize with the network disabled.

Some protected images cannot be reacquired without website credentials. Those
images receive an explicit unavailable/error state; RealImage does not invent a
score. Chrome-internal pages, the Chrome Web Store, other extensions, closed
shadow roots, and inaccessible authenticated cross-origin images are outside the
supported acquisition scope.

## User control

Detection can be paused globally. Blur and hide are optional presentation
choices and are reversible per image. Removing the extension deletes its local
settings. Scores are probabilistic signals, not proof of authorship and should
not be used alone for consequential decisions.
