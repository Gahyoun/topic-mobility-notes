(() => {
  "use strict";

  function element(tag, className, content) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content) node.textContent = content;
    return node;
  }

  function moveAs(source, tag, className) {
    const replacement = element(tag, className);
    replacement.append(...source.childNodes);
    source.replaceWith(replacement);
    return replacement;
  }

  function clearDiagramStyles(schematic) {
    for (const node of schematic.querySelectorAll("[style]")) {
      if (!node.closest(".katex")) node.removeAttribute("style");
    }
  }

  function designDecomposition(schematic) {
    const header = schematic.querySelector(":scope > .sch-head");
    const identity = header?.querySelector(":scope > .sch-badge");
    const lanes = schematic.querySelector(":scope > .sch-grid-2");
    if (!identity || lanes?.children.length !== 2) return;

    clearDiagramStyles(schematic);
    schematic.classList.add("comparison-architecture", "variance-architecture");
    const equation = moveAs(identity, "div", "architecture-equation");
    const formula = equation.querySelector(".katex");
    if (formula) formula.classList.add("math", "inline");
    equation.append(element("p", "architecture-context", "두 포트폴리오 모두 주제 A와 B의 비중이 50%씩입니다."));
    header.after(equation);

    const legend = element("p", "architecture-legend");
    legend.append(element("span", "architecture-topic-a", "● 주제 A"), element("span", "architecture-topic-b", "◆ 주제 B"));
    equation.after(legend);
    lanes.className = "architecture-lanes";

    for (const panel of [...lanes.children]) {
      const lane = moveAs(panel, "section", "architecture-lane");
      const titleRow = lane.querySelector(".sch-row-between");
      const ratio = titleRow?.querySelector(".sch-badge");
      if (ratio) ratio.className = "architecture-ratio";
      const papers = lane.querySelector(".sch-paper-grid");
      papers?.classList.replace("sch-paper-grid", "architecture-papers");
      for (const paper of lane.querySelectorAll(".sch-paper-slot")) {
        paper.className = "architecture-paper";
        const [title, tokens] = paper.children;
        moveAs(title, "p", "architecture-paper-title");
        const refs = moveAs(tokens, "ul", "architecture-references");
        refs.setAttribute("role", "list");
        refs.setAttribute("aria-label", "참고문헌 주제");
        for (const token of [...refs.children]) {
          const variant = token.classList.contains("a") ? "a" : "b";
          const [symbol, ...text] = token.textContent.split(" ");
          const marker = element("span", "", symbol);
          marker.setAttribute("aria-hidden", "true");
          const item = element("li", `architecture-topic-${variant}`);
          item.append(marker, element("span", "sr", text.join(" ")));
          token.replaceWith(item);
        }
      }

      const interpretation = lane.querySelector(".sch-pill-box");
      if (interpretation) {
        const findings = element("ul", "architecture-findings");
        let line = element("li");
        for (const node of [...interpretation.childNodes]) {
          if (node.nodeName === "BR") {
            findings.append(line);
            line = element("li");
          } else {
            if (!line.firstChild && node.nodeType === Node.TEXT_NODE) node.data = node.data.replace(/^\s*•\s*/, "");
            line.append(node);
          }
        }
        findings.append(line);
        interpretation.replaceWith(findings);
      }
    }

    const correction = schematic.querySelector(":scope > .sch-formula-bar");
    if (correction) {
      correction.className = "architecture-correction";
      for (const formula of correction.querySelectorAll(".katex")) formula.classList.add("math", "inline");
    }
  }

  function labeledNode(source, className) {
    const node = moveAs(source, "div", className);
    const lineBreak = node.querySelector(":scope > br");
    if (lineBreak) {
      const detail = element("span", "architecture-node-detail");
      while (lineBreak.nextSibling) detail.append(lineBreak.nextSibling);
      lineBreak.replaceWith(detail);
    }
    return node;
  }

  function metricTable(biology, physics) {
    const table = element("table", "architecture-metrics");
    table.createCaption().textContent = "두 채널의 퍼짐 변화";
    const headings = table.createTHead().insertRow();
    for (const name of ["측정 채널", "생명과학", "물리학"]) {
      const header = element("th", "", name);
      header.scope = "col";
      headings.append(header);
    }
    const body = table.createTBody();
    const names = ["분류 채널", "참고문헌 채널"];
    for (const [index, name] of names.entries()) {
      const row = body.insertRow();
      const label = element("th", "", name);
      label.scope = "row";
      const formula = biology[index].querySelector(".katex");
      if (formula) label.append(" ", formula);
      row.append(label);
      for (const finding of [biology[index], physics[index]]) {
        const cell = row.insertCell();
        const firstText = finding.firstChild;
        if (firstText?.nodeType === Node.TEXT_NODE) firstText.data = firstText.data.replace(name, "").trimStart();
        const formula = finding.querySelector(".katex");
        if (formula?.textContent.includes("widehat{Q}")) formula.remove();
        const firstValue = finding.querySelector("b");
        if (firstValue) {
          firstValue.className = firstValue.textContent.trim().startsWith("−") ? "architecture-negative" : "architecture-positive";
        }
        if (firstText?.nodeType === Node.TEXT_NODE && !firstText.data.trim()) firstText.remove();
        const punctuation = finding.firstChild;
        if (punctuation?.nodeType === Node.TEXT_NODE) punctuation.data = punctuation.data.replace(/^\s*:\s*/, "");
        cell.append(...finding.childNodes);
        finding.remove();
      }
    }
    return table;
  }

  function designDivergence(schematic) {
    const lanes = schematic.querySelector(":scope > .sch-grid-2");
    if (lanes?.children.length !== 2) return;
    const [bioSource, physicsSource] = lanes.children;
    const [core, ...bioFindings] = bioSource.querySelectorAll(":scope > .sch-pill-box");
    const bioTracks = bioSource.querySelector(":scope > .sch-grid-3");
    const physicsFindings = [...physicsSource.querySelectorAll(":scope > .sch-pill-box")];
    const [references, papers] = physicsSource.querySelectorAll(":scope > .sch-grid-3");
    if (!core || !bioTracks || bioFindings.length !== 2 || physicsFindings.length !== 2 ||
        !references || references.children.length !== papers?.children.length) return;

    clearDiagramStyles(schematic);
    schematic.classList.add("comparison-architecture", "divergence-architecture");
    lanes.className = "architecture-lanes";
    const bio = moveAs(bioSource, "section", "architecture-lane");
    const physics = moveAs(physicsSource, "section", "architecture-lane");
    labeledNode(core, "architecture-core");
    const bioFlow = bio.querySelector(".sch-arrow-down");
    if (bioFlow) {
      bioFlow.className = "architecture-flow-note";
      bioFlow.textContent = bioFlow.textContent.replace(/^[\s↙↓↘(]+|\)$/g, "");
    }
    bioTracks.className = "architecture-bio-tracks";
    for (const track of [...bioTracks.children]) labeledNode(track, "architecture-track");

    const pairList = element("div", "architecture-pairs");
    const referenceTracks = [...references.children];
    const paperTracks = [...papers.children];
    for (const [index, track] of referenceTracks.entries()) {
      const pair = element("div", "architecture-pair");
      pair.append(labeledNode(track, "architecture-track"), labeledNode(paperTracks[index], "architecture-track"));
      pairList.append(pair);
    }
    references.replaceWith(pairList);
    physics.querySelector(".sch-arrow-down")?.remove();
    papers.remove();
    lanes.after(metricTable(bioFindings, physicsFindings));
  }

  function arrangeExamples(main) {
    for (const example of main.querySelectorAll(".example-box")) {
      const description = example.querySelector(":scope > p:not(.example-title)");
      if (!description || !/②[\s\S]*③/.test(description.textContent)) continue;
      const intro = document.createDocumentFragment();
      const steps = element("ol", "example-steps");
      let target = intro;
      for (const node of [...description.childNodes]) {
        if (node.nodeType !== Node.TEXT_NODE) {
          if (node.nodeName !== "BR") target.append(node);
          continue;
        }
        const segments = node.data.split(/[①②③④⑤]\s*/);
        target.append(segments[0]);
        for (const text of segments.slice(1)) {
          target = element("li", "", text);
          steps.append(target);
        }
      }
      if (intro.textContent.trim()) description.replaceChildren(intro);
      else description.remove();
      example.append(steps);
    }
  }

  window.arrangeComparisonArchitectures = (main) => {
    for (const schematic of main.querySelectorAll(".sch-wrap:not(.comparison-architecture)")) {
      const name = schematic.querySelector(":scope > .sch-head .sch-kicker")?.textContent || "";
      if (name.startsWith("ORTHOGONAL TWO-BODY VARIANCE")) designDecomposition(schematic);
      if (name.startsWith("EPISTEMIC CULTURES & DUAL-CHANNEL DIVERGENCE")) designDivergence(schematic);
    }
    arrangeExamples(main);
  };
})();
