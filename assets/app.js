// Unlock shell: PBKDF2-SHA256 -> AES-256-GCM, decrypt in memory, hash routing (#/slug).
// The derived key (not the password) is kept in sessionStorage for this tab only.
(() => {
  "use strict";
  const KEY = "topic-notes-key";
  const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
  const enc = new TextEncoder();
  const app = document.getElementById("app");
  const desktop = window.matchMedia("(min-width: 1024px)");
  const noteLabels = {
    index: "연구 개요",
    framework: "시계열 임베딩과 모형",
    q0: "경력별 평균 제곱 쌍거리",
    h2: "Comoving frame과 중심 변위",
    h12: "분야별 초과 퍼짐",
    integration: "논문 내·논문 간 공분산",
    measurement: "임베딩 타당도와 robustness",
    data: "관측 모집단과 분석 표본",
    hypotheses: "가설·추정량·판정 기준",
  };
  const navigationGroups = ["모형과 가설", "관측과 분해", "데이터와 검증"];
  let site = null;
  let currentSlug = null;
  let storyOffset = null;
  let storyFrameRequest = 0;
  let pendingStoryAnchor = false;
  let mathLayoutFrame = 0;

  function updateMathLayout() {
    mathLayoutFrame = 0;
    const main = app.querySelector("main#main");
    if (!main) return;
    const inlineMath = [...main.querySelectorAll(".math.inline")];
    const widths = new Map();
    const wide = inlineMath.map((formula) => {
      let container = formula.parentElement;
      while (container !== main && getComputedStyle(container).display === "inline") {
        container = container.parentElement;
      }
      if (!widths.has(container)) {
        const style = getComputedStyle(container);
        widths.set(container, container.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight));
      }
      const available = widths.get(container);
      const segments = formula.querySelectorAll(".katex-html > :is(.katex-base, .base)");
      return available > 0 && [...segments].some((segment) => segment.getBoundingClientRect().width > available + 1);
    });
    inlineMath.forEach((formula, index) => formula.classList.toggle("is-wide", wide[index]));
    const blocks = [...main.querySelectorAll(".math.block")];
    for (const block of blocks) block.style.removeProperty("--math-scale");
    const scales = blocks.map((block) => block.scrollWidth > block.clientWidth + 1
      ? (block.clientWidth - 2) / block.scrollWidth : 1);
    blocks.forEach((block, index) => {
      if (scales[index] >= 0.9 && scales[index] < 1) block.style.setProperty("--math-scale", scales[index]);
    });
    for (const formula of [...blocks, ...main.querySelectorAll(".math.inline.is-wide")]) {
      if (formula.scrollWidth > formula.clientWidth + 1) formula.tabIndex = 0;
      else formula.removeAttribute("tabindex");
    }
    for (const formula of main.querySelectorAll(".math.inline:not(.is-wide)[tabindex]")) {
      formula.removeAttribute("tabindex");
    }
    for (const chart of main.querySelectorAll(".career-chart-scroll")) {
      if (chart.scrollWidth > chart.clientWidth + 1) chart.tabIndex = 0;
      else chart.removeAttribute("tabindex");
    }
  }

  function scheduleMathLayout() {
    if (!mathLayoutFrame) mathLayoutFrame = requestAnimationFrame(updateMathLayout);
  }

  function reportStoryScroll() {
    if (storyOffset === null || storyFrameRequest) return;
    storyFrameRequest = requestAnimationFrame(() => {
      storyFrameRequest = 0;
      const frame = document.getElementById("trajectory-example");
      if (!frame || storyOffset === null) return;
      frame.contentWindow.postMessage({
        type: "topic-trajectory-scroll",
        top: frame.getBoundingClientRect().top + storyOffset,
        viewport: window.innerHeight,
      }, "*"); // Sandboxed srcdoc has an opaque origin.
    });
  }

  window.addEventListener("scroll", reportStoryScroll, {passive:true});
  window.addEventListener("resize", reportStoryScroll);
  window.addEventListener("resize", scheduleMathLayout);
  document.fonts.addEventListener("loadingdone", scheduleMathLayout);
  app.addEventListener("toggle", (event) => {
    if (event.target.matches("#main details") && event.target.querySelector(".math")) scheduleMathLayout();
  }, true);

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
    if (blob.enc === "gzip") {
      return new Response(new Blob([pt]).stream().pipeThrough(new DecompressionStream("gzip"))).json();
    }
    return JSON.parse(new TextDecoder().decode(pt));
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function arrangeNavigation() {
    const header = app.querySelector(".top-in");
    const actions = element("div", "page-actions");
    const metadata = header.querySelector(".meta");
    if (metadata) {
      actions.append(metadata);
      app.querySelector("footer .top-in")?.append(element("span", "meta mobile-page-meta", metadata.textContent));
    }
    const lock = element("button", "page-lock", "잠그기");
    lock.type = "button";
    lock.addEventListener("click", () => {
      try { sessionStorage.removeItem(KEY); } catch (_) {}
      location.reload();
    });
    actions.append(lock);
    header.append(actions);

    const nav = app.querySelector("nav.side");
    nav.setAttribute("aria-label", "노트 목록");
    nav.querySelectorAll("h2").forEach((heading, index) => {
      if (navigationGroups[index]) heading.textContent = navigationGroups[index];
    });
    const pager = app.querySelector(".pager");
    pager?.setAttribute("aria-label", "이전 및 다음 노트");
    for (const link of app.querySelectorAll(".side a, .pager a")) {
      const label = noteLabels[link.hash.slice(2)];
      if (!label) continue;
      const direction = link.querySelector("small");
      link.replaceChildren(...(direction ? [direction, label] : [label]));
    }
    const disclosure = element("details", "mobile-notes");
    disclosure.open = desktop.matches;
    nav.before(disclosure);
    disclosure.append(element("summary", "", "노트 목록"), nav);

    const skip = element("a", "skip", "본문 바로가기");
    skip.href = "#main";
    app.prepend(skip);
  }

  function literatureCell(source, label) {
    const group = element("div");
    const value = element("dd");
    group.append(element("dt", "", label), value);
    if (source.classList.contains("sch-gen-ours")) {
      value.append(...source.childNodes);
      return group;
    }
    const list = element("ul");
    let item = element("li");
    for (const node of [...source.childNodes]) {
      if (node.nodeName === "BR") {
        if (item.textContent.trim()) list.append(item);
        item = element("li");
      } else {
        if (!item.hasChildNodes() && node.nodeType === Node.TEXT_NODE) {
          node.textContent = node.textContent.replace(/^\s*·\s*/, "");
        }
        item.append(node);
      }
    }
    if (item.textContent.trim()) list.append(item);
    value.append(list);
    return group;
  }

  function arrangeSchematics(main) {
    const labels = ["선행 연구", "핵심 문헌", "최근 후속 연구", "본 연구의 기여"];
    for (const head of main.querySelectorAll(".sch-genealogy-row:has(> .sch-genealogy-head)")) {
      const schematic = head.parentElement;
      const list = element("ol", "literature-list");
      const rows = schematic.querySelectorAll(":scope > .sch-genealogy-row:has(> .sch-gen-stream)");
      for (const row of rows) {
        const [stream, ...cells] = row.children;
        const item = element("li", "literature-stream");
        const heading = element("h3");
        heading.append(...stream.childNodes);
        if (heading.firstChild?.nodeType === Node.TEXT_NODE) {
          heading.firstChild.textContent = heading.firstChild.textContent.replace(/^\s*\d+[.)]\s*/, "");
        }
        heading.querySelectorAll("br").forEach((lineBreak) => lineBreak.replaceWith(" "));
        const groups = element("dl");
        cells.forEach((cell, index) => groups.append(literatureCell(cell, labels[index])));
        item.append(heading, groups);
        list.append(item);
        row.remove();
      }
      head.replaceWith(list);
    }

    for (const schematic of main.querySelectorAll(".sch-wrap")) {
      const kicker = schematic.querySelector(":scope > .sch-head .sch-kicker")?.textContent || "";
      if (kicker.includes("CURRENT-ARCHIVE EMPIRICAL EXECUTION BLUEPRINT")) {
        schematic.classList.add("execution-blueprint");
        const inputs = schematic.querySelector(":scope > .sch-grid-4");
        const branches = schematic.querySelector(":scope > .sch-grid-3");
        inputs?.before(element("h3", "architecture-stage", "검증에 쓰는 자료"));
        branches?.before(element("h3", "architecture-stage", "가설별 검증 경로"));
      }
      if (kicker.includes("DUAL CAREER TRANSITIONS")) {
        schematic.classList.add("career-transition");
        const chart = schematic.querySelector("svg.sch-svg");
        if (chart) {
          const plot = element("div", "career-chart-scroll");
          plot.setAttribute("role", "region");
          plot.setAttribute("aria-label", "경력 단계별 궤적 그래프");
          const legend = chart.nextElementSibling;
          if (legend?.classList.contains("sch-row-between")) legend.style.removeProperty("font-size");
          chart.replaceWith(plot);
          plot.append(chart);
        }
      }
      if (kicker.includes("COMOVING OPTIMAL TRANSPORT")) {
        schematic.classList.add("reading-architecture", "spectrum-architecture");
        addArchitectureLabels(schematic, ["변위 추정", "스펙트럼 판별"]);
      }
      if (kicker.includes("INSTITUTIONAL HIERARCHY (H3)")) {
        schematic.classList.add("reading-architecture", "pathways-architecture");
        addArchitectureLabels(schematic, ["기관 위계와 선발", "공저와 신규 주제"]);
      }
    }
    window.arrangeComparisonArchitectures(main);
  }

  function addArchitectureLabels(schematic, labels) {
    const grid = schematic.querySelector(":scope > .sch-grid-2");
    if (!grid || grid.children.length !== labels.length) return;
    [...grid.children].forEach((lane, index) => {
      lane.prepend(element("div", "architecture-lane-label", labels[index]));
    });
  }

  function addOutline(main, slug) {
    const flow = main.querySelector(".methodology") || main;
    const headings = [...flow.querySelectorAll(":scope > h2, :scope > h3, :scope > section > h2, :scope > section > h3")];
    const sections = headings.filter((heading) => heading.tagName === "H2");
    if (sections.length < 3) return;

    function outlineLink(heading, id) {
      const target = heading.id ? heading : heading.tagName === "H2" ? heading.closest("section[id]") || heading : heading;
      target.id ||= id;
      const label = (heading.querySelector(".katex-html") ? heading.innerText : heading.textContent).trim();
      const link = element("a", "", label);
      link.href = `#/${slug}/${target.id}`;
      return link;
    }

    const list = element("ul", "outline-tree");
    sections.forEach((heading, index) => {
      const start = headings.indexOf(heading);
      const end = index + 1 < sections.length ? headings.indexOf(sections[index + 1]) : headings.length;
      const children = headings.slice(start + 1, end);
      const item = element("li");
      const link = outlineLink(heading, `section-${index + 1}`);
      item.append(link);
      if (children.length) {
        const nested = element("ul");
        children.forEach((child, childIndex) => {
          const subitem = element("li");
          subitem.append(outlineLink(child, `section-${index + 1}-${childIndex + 1}`));
          nested.append(subitem);
        });
        item.append(nested);
      }
      list.append(item);
    });
    const outline = element("details", "note-outline");
    const outlineNav = element("nav");
    outlineNav.setAttribute("aria-label", "이 페이지의 목차");
    outlineNav.append(list);
    outline.append(element("summary", "", "이 페이지의 목차"), outlineNav);
    if (slug === "index" || slug === "framework") outline.open = desktop.matches;
    (sections[0].closest("section") || sections[0]).before(outline);
    if (slug === "index" || slug === "framework") main.classList.add("hierarchical-note");
  }

  function route(focus = false) {
    if (!site) return;
    const [, requested, anchor] = location.hash.match(/^#\/([\w-]+)(?:\/([\w-]+))?$/) || [];
    const slug = requested && Object.hasOwn(site.pages, requested) ? requested : "index";
    if (slug !== currentSlug) {
      const page = site.pages[slug];
      storyOffset = null;
      app.innerHTML = page.html;
      document.title = page.title + " · 연구 노트";
      const main = app.querySelector("main");
      main.tabIndex = -1;
      arrangeNavigation();
      for (const lineBreak of main.querySelectorAll("p.lede > br, p.intro > br")) lineBreak.replaceWith(" ");
      arrangeSchematics(main);
      addOutline(main, slug);
      updateMathLayout();
      currentSlug = slug;
    }

    const main = app.querySelector("main");
    const anchorId = anchor === "trajectory" ? "trajectory-example" : anchor;
    const target = anchorId && document.getElementById(anchorId);
    pendingStoryAnchor = Boolean(target && anchor === "trajectory" && storyOffset === null);
    if (target && main.contains(target)) {
      if (focus && target.tabIndex < 0) target.tabIndex = -1;
      if (focus) target.focus({preventScroll: true});
      target.scrollIntoView({behavior:"instant", block:"start"});
      reportStoryScroll();
    } else {
      if (focus) main.focus({preventScroll: true});
      window.scrollTo(0, 0);
    }
  }

  desktop.addEventListener("change", () => {
    const disclosure = app.querySelector(".mobile-notes");
    if (disclosure) disclosure.open = desktop.matches;
  });

  app.addEventListener("click", (event) => {
    if (event.target.closest("a.skip")) {
      event.preventDefault();
      const main = app.querySelector("main");
      main.focus();
      main.scrollIntoView({block: "start"});
      return;
    }
    const link = event.target.closest(".side a");
    if (!link || event.defaultPrevented || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    if (!desktop.matches) link.closest(".mobile-notes").open = false;
    if (link.hash === location.hash) route(true);
  });

  window.addEventListener("message", (event) => {
    if (event.origin !== "null") return;
    if (event.data?.type === "topic-faculty-flow-height") {
      const flowFrame = document.querySelector(".faculty-flow-embed");
      const h = event.data.height;
      if (flowFrame?.contentWindow === event.source && Number.isInteger(h) && h >= 280 && h <= 2400) {
        flowFrame.style.height = h + "px";
      }
      return;
    }
    if (event.data?.type !== "topic-trajectory-height") return;
    const frame = document.querySelector(".trajectory-embed");
    const height = event.data.height;
    if (frame?.contentWindow === event.source && Number.isInteger(height) && height >= 320 && height <= 4000) {
      frame.style.height = height + "px";
      const offset = event.data.storyOffset;
      if (Number.isInteger(offset) && offset >= 0 && offset < height) {
        storyOffset = offset;
        if (pendingStoryAnchor) {
          pendingStoryAnchor = false;
          requestAnimationFrame(() => {
            if (frame.isConnected) frame.scrollIntoView({behavior:"instant", block:"start"});
            reportStoryScroll();
          });
        } else reportStoryScroll();
      }
    }
  });

  function showSite(unlocked, focus = false) {
    site = unlocked;
    route(focus);
    window.addEventListener("hashchange", () => route(true));
  }

  async function start() {
    const form = document.getElementById("unlock");
    const input = document.getElementById("pw");
    const err = document.getElementById("err");
    const btn = form.querySelector("button");
    const pending = loadBlob().catch(() => null);
    let manuallyUnlocking = false;

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      manuallyUnlocking = true;
      err.textContent = "";
      input.removeAttribute("aria-invalid");
      btn.disabled = true;
      btn.textContent = "노트 여는 중…";
      try {
        const blob = await pending || await loadBlob();
        const key = await derive(input.value, blob);
        showSite(await open(key, blob), true);
        try {
          const raw = new Uint8Array(await crypto.subtle.exportKey("raw", key));
          sessionStorage.setItem(KEY, btoa(String.fromCharCode(...raw)));
        } catch (_) {}
      } catch (error) {
        if (error.name === "OperationError") {
          err.textContent = "비밀번호를 다시 확인하세요.";
          input.setAttribute("aria-invalid", "true");
          input.select();
        } else {
          err.textContent = "연구 노트를 불러오지 못했습니다. 잠시 후 다시 시도하세요.";
        }
        btn.disabled = false;
        btn.textContent = "노트 열기";
      }
    });

    const blob = await pending;
    if (manuallyUnlocking || !blob) return;
    try {
      const raw = sessionStorage.getItem(KEY);
      if (raw) {
        const key = await crypto.subtle.importKey("raw", b64(raw), "AES-GCM", true, ["decrypt"]);
        const cachedSite = await open(key, blob);
        if (!manuallyUnlocking) showSite(cachedSite);
      }
    } catch (_) {
      try { sessionStorage.removeItem(KEY); } catch (_) {}
    }
  }

  start();
})();
