// Populated at container start by frontend/entrypoint.sh (deploy); absent in
// `bun run dev`, where the fallbacks below (talk straight to localhost, no
// path prefix) are correct as-is.
declare global {
  interface Window {
    __CONFIG__?: { API_BASE?: string; BASE_PATH?: string };
  }
}

// index.html ships literal "__API_BASE__"/"__BASE_PATH__" placeholders that
// entrypoint.sh substitutes at container start; in `bun run dev` they're
// never substituted, so treat an unreplaced placeholder as "unset" too.
const isSet = (v?: string) => !!v && !v.startsWith('__');

export const API_BASE = isSet(window.__CONFIG__?.API_BASE) ? window.__CONFIG__!.API_BASE! : 'http://localhost:8000';
export const BASE_PATH = isSet(window.__CONFIG__?.BASE_PATH) ? window.__CONFIG__!.BASE_PATH! : '';
