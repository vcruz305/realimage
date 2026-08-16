# Manual E2E: file://, localhost/127.0.0.1, and offline checks

Run this once in your own installed Chrome (Claude-in-Chrome is not
connected in the session that prepared this). Steps:

1. **Build & load.** `npm run build`. If not already loaded, open
   `chrome://extensions`, enable Developer Mode, "Load unpacked" -> select
   `dist/`.
2. **Allow file URLs.** On the RealImage card in `chrome://extensions`,
   click **Details** -> toggle **Allow access to file URLs** on.
3. **file:// gallery.** Open
   `file:///<repo-path>/tests/fixture/local-gallery/index.html` directly in
   the address bar. **Expected: 6 badges**, one per image, roughly 3
   flagged AI-leaning and 3 real-leaning (exact scores will vary -- PASS
   means every image gets SOME scored badge, none show an error/unavailable
   state).
4. **localhost + 127.0.0.1.** Run `npm run fixture` (serves
   `tests/fixture/` at `http://127.0.0.1:4173` by default). Open it once as
   `http://localhost:4173/` and once as `http://127.0.0.1:4173/`. **Expected
   each time:** both fixture images get a scored badge, no
   unavailable/error state.
5. **Backend + threads.** Click the toolbar icon to open the popup, then
   click **Settings** at the bottom (or right-click the toolbar icon ->
   **Options**). On the Options page, click **Run local readiness check**.
   The result appears next to that button (`#model-status` in
   `src/options/options.html`, rendered by `src/options/options.js`) as
   `Ready · <backend> · <N> threads · <isolated|single-thread safety mode>`
   -- note the backend (`WebGPU` or `WASM`) and thread count shown there.

6. **Fresh profile, offline (the evaluator's exact scenario).** In a new
   Chrome profile (`chrome://settings/manageProfile` -> Add), `npm run
   build:fresh` once with internet on, load unpacked `dist/`, enable *Allow
   access to file URLs* (step 2). Then **disconnect from the internet
   entirely** (turn off Wi-Fi/network) and, still offline, open the
   `file://` gallery from step 3 again. **Expected: still 6 badges**, no
   network-error/unavailable state, and the Options page readiness check
   (step 5) still reports a backend and thread count. This proves no
   post-build network access is required.

## Record your results

| Case | Pass/Fail | Notes (console/extension errors, actual scores) |
|---|---|---|
| file:// gallery -> 6 badges | | |
| http://localhost:4173/ -> badges, no errors | | |
| http://127.0.0.1:4173/ -> badges, no errors | | |
| Options page backend/thread readout | Backend: ____  Threads: ____ | |
| Fresh profile, offline, file:// -> 6 badges | | |

Note any red text in `chrome://extensions` (service worker errors) or the
page/DevTools console during any of the above.
