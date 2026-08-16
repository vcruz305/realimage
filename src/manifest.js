import { RELEASE } from './shared/constants.js';

export default {
  manifest_version: 3,
  name: RELEASE.name,
  short_name: RELEASE.shortName,
  version: RELEASE.version,
  description: 'Detects AI-generated images locally in your browser and shows a confidence score. No cloud, no uploads.',
  minimum_chrome_version: '148',
  permissions: ['storage', 'offscreen', 'activeTab'],
  host_permissions: ['http://*/*', 'https://*/*', 'file:///*'],
  background: {
    service_worker: 'src/background/index.js',
    type: 'module'
  },
  icons: {
    16: 'icons/icon-16.png',
    32: 'icons/icon-32.png',
    48: 'icons/icon-48.png',
    128: 'icons/icon-128.png'
  },
  action: {
    default_title: 'RealImage',
    default_icon: {
      16: 'icons/icon-16.png',
      32: 'icons/icon-32.png',
      48: 'icons/icon-48.png'
    },
    default_popup: 'src/popup/popup.html'
  },
  options_page: 'src/options/options.html',
  content_scripts: [
    {
      matches: ['http://*/*', 'https://*/*', 'file:///*'],
      js: ['src/content/index.js'],
      run_at: 'document_idle',
      all_frames: true,
      match_about_blank: true,
      match_origin_as_fallback: true
    }
  ],
  cross_origin_opener_policy: {
    value: 'same-origin'
  },
  cross_origin_embedder_policy: {
    value: 'require-corp'
  },
  content_security_policy: {
    extension_pages: "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'"
  }
};
