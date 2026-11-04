// Unlock shell: PBKDF2-SHA256 -> AES-256-GCM, decrypt in memory, hash routing (#/slug).
// The derived key (not the password) is kept in sessionStorage for this tab only.
(() => {
  "use strict";
  const KEY = "topic-notes-key";
  const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
  const enc = new TextEncoder();
  let site = null;

  async function loadBlob() {
    const r = await fetch("data/site.json", { cache: "no-store" });
    if (!r.ok) throw new Error("data unavailable");
    return r.json();
  }

  async function derive(pw, blob) {
    const base = await crypto.subtle.importKey("raw", enc.encode(pw), "PBKDF2", false, ["deriveKey"]);
    return crypto.subtle.deriveKey(
      { name: "PBKDF2", hash: "SHA-256", salt: b64(blob.salt), iterations: blob.iter },
      base, { name: "AES-GCM", length: 256 }, true, ["decrypt"]);
  }

  async function open(key, blob) {
    const pt = await crypto.subtle.decrypt(
      { name: "AES-GCM", iv: b64(blob.iv), additionalData: enc.encode(blob.aad) }, key, b64(blob.ct));
    return JSON.parse(new TextDecoder().decode(pt));
  }

  function route() {
    if (!site) return;
    const slug = (location.hash.match(/^#\/([\w-]+)/) || [])[1] || "index";
    const p = site.pages[slug] || site.pages.index;
    document.getElementById("app").innerHTML = p.html;
    document.title = p.title + " · Research notes";
    const cur = document.querySelector('nav.side a[aria-current="page"]');
    const nav = document.querySelector("nav.side");
    if (cur && nav && nav.scrollWidth > nav.clientWidth) nav.scrollLeft = cur.offsetLeft - 16;
    window.scrollTo(0, 0);
  }

  async function unlockWith(key, blob) {
    site = await open(key, blob);
    window.addEventListener("hashchange", route);
    route();
  }

  async function start() {
    let blob;
    try { blob = await loadBlob(); } catch (e) { return; }
    try {
      const raw = sessionStorage.getItem(KEY);
      if (raw) {
        const key = await crypto.subtle.importKey("raw", b64(raw), "AES-GCM", true, ["decrypt"]);
        await unlockWith(key, blob);
        return;
      }
    } catch (e) { try { sessionStorage.removeItem(KEY); } catch (_) {} }

    const form = document.getElementById("unlock");
    const input = document.getElementById("pw");
    const err = document.getElementById("err");
    const btn = form.querySelector("button");
    input.focus();
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      err.textContent = ""; btn.disabled = true; btn.textContent = "…";
      try {
        const key = await derive(input.value, blob);
        await unlockWith(key, blob);
        try {
          const raw = new Uint8Array(await crypto.subtle.exportKey("raw", key));
          sessionStorage.setItem(KEY, btoa(String.fromCharCode(...raw)));
        } catch (_) {}
      } catch (e) {
        err.textContent = "비밀번호가 맞지 않습니다.";
        btn.disabled = false; btn.textContent = "Unlock"; input.select();
      }
    });
  }

  start();
})();
