/* The A/B player of a SeamSplit listening kit, shared by the GUI and by the page that is
   written next to a kit (index.html). It plays the clips of a seam in step, so that
   switching between them changes only how the seam is joined, and records which one
   sounds best. By default the buttons are shuffled and unnamed (blind). See
   seamsplit/audition.py for what the clips are.

   SeamAudition.mount(root, kit, options)
     kit      the contents of audition.json
     options  url(file) -> where to load a clip from
              t(key, vars) -> text (SeamAudition.TEXT holds the English and Vietnamese)
              save(seam, choice, blind) -> Promise, optionally resolving to {summary}
              initial  {seam: {choice, blind}} answers given before
              download(doc) -> called with the answer sheet when the user asks for it
              docHref  a link to the answer sheet, instead of building one in the page
   Returns {answers(), refresh()}. */
(function (global) {
  "use strict";

  const TEXT = {
    en: {
      listenTitle: "3. Listen to the seams",
      listenIntro: "Each seam is joined in different ways. Press play, switch between the buttons while it plays (keys 1, 2, 3), and answer which one you hear the seam least in. Until you answer, the buttons are shuffled and unnamed, so what you expect cannot sway you.",
      seamHeading: "Seam {seam} of {of}, at {at}",
      seamHeadingNoTime: "Seam {seam} of {of}",
      metrics: "Agreement over the overlap {agree} dB · offset {lag} ms · level {gain} dB",
      play: "Play", pause: "Pause",
      loop: "Loop around the seam (±2 s)",
      blind: "Blind: shuffle and hide the names",
      option: "Version {letter}",
      best: "Which one sounds best?",
      tie: "No audible difference",
      answered: "Your answer: {name}",
      answeredTie: "Your answer: no audible difference",
      reveal: "{letter} = {name}",
      same: "Here the seam search made the same audio as the crossfade, so there are only two to compare.",
      kind_straight: "Straight join",
      kind_crossfade: "Crossfade ({method})",
      kind_seam_search: "Seam search",
      loadError: "Could not load {file}.",
      saveError: "Could not save the answer: {msg}",
      summaryTitle: "Your answers",
      noAnswers: "No answers yet.",
      counts: "{n} answer(s): straight join {straight}, crossfade {crossfade}, seam search {seam_search}, no audible difference {tie}.",
      nonBlind: "{n} of them were not blind and are left out of the tests below.",
      svc: "Seam search against the crossfade: {seam_search} to {crossfade} (p = {p}): {verdict}.",
      svp: "Straight join against the other two: {straight} to {processed} (p = {p}): {verdict}.",
      v_too_few: "too few decided seams to tell (at least 6 are needed)",
      v_no_clear_preference: "no clear preference",
      v_seam_search: "seam search preferred",
      v_crossfade: "the crossfade over the whole overlap preferred",
      v_straight: "the straight join preferred",
      v_processed: "the processed joins preferred to the straight one",
      caveat: "A few seams of one song say little. Answers from several songs and services say more: seamsplit audition-summary reads any number of answer files together.",
      downloadAnswers: "Download the answers (.json)",
    },
    vi: {
      listenTitle: "3. Nghe các mối nối",
      listenIntro: "Mỗi mối nối được nối theo nhiều cách. Bấm phát, chuyển qua lại giữa các nút khi đang phát (phím 1, 2, 3), rồi chọn cách mà bạn nghe mối nối ít lộ nhất. Trước khi trả lời, các nút bị xáo trộn và không ghi tên, nên kỳ vọng của bạn không làm lệch kết quả.",
      seamHeading: "Mối nối {seam} trên {of}, tại {at}",
      seamHeadingNoTime: "Mối nối {seam} trên {of}",
      metrics: "Độ khớp trên phần chồng lấp {agree} dB · lệch {lag} ms · mức âm {gain} dB",
      play: "Phát", pause: "Tạm dừng",
      loop: "Lặp quanh mối nối (±2 giây)",
      blind: "Nghe mù: xáo trộn và ẩn tên",
      option: "Bản {letter}",
      best: "Bản nào nghe tốt nhất?",
      tie: "Không nghe ra khác biệt",
      answered: "Bạn đã chọn: {name}",
      answeredTie: "Bạn đã chọn: không nghe ra khác biệt",
      reveal: "{letter} = {name}",
      same: "Ở mối nối này tìm-mối-nối cho ra đúng âm thanh của crossfade, nên chỉ còn hai bản để so.",
      kind_straight: "Nối thẳng",
      kind_crossfade: "Crossfade ({method})",
      kind_seam_search: "Tìm mối nối",
      loadError: "Không tải được {file}.",
      saveError: "Không lưu được câu trả lời: {msg}",
      summaryTitle: "Các câu trả lời của bạn",
      noAnswers: "Chưa có câu trả lời nào.",
      counts: "{n} câu trả lời: nối thẳng {straight}, crossfade {crossfade}, tìm mối nối {seam_search}, không khác biệt {tie}.",
      nonBlind: "{n} câu trong đó không nghe mù nên không được tính trong các phép kiểm bên dưới.",
      svc: "Tìm mối nối so với crossfade: {seam_search} - {crossfade} (p = {p}): {verdict}.",
      svp: "Nối thẳng so với hai cách còn lại: {straight} - {processed} (p = {p}): {verdict}.",
      v_too_few: "chưa đủ số mối nối có phân thắng bại (cần ít nhất 6)",
      v_no_clear_preference: "chưa có thiên hướng rõ ràng",
      v_seam_search: "tìm mối nối được ưa hơn",
      v_crossfade: "crossfade trên cả phần chồng lấp được ưa hơn",
      v_straight: "nối thẳng được ưa hơn",
      v_processed: "các cách nối đã xử lý được ưa hơn nối thẳng",
      caveat: "Vài mối nối của một bài nói được rất ít. Câu trả lời từ nhiều bài và nhiều dịch vụ nói được nhiều hơn: lệnh seamsplit audition-summary đọc được nhiều file trả lời cùng lúc.",
      downloadAnswers: "Tải các câu trả lời (.json)",
    },
  };

  const CSS = `
.sa-seam { border: 1px solid var(--line, #d9dde6); border-radius: 10px; padding: 12px 14px; margin: 12px 0; background: var(--card, #fff); }
.sa-seam:focus-visible { outline: 2px solid var(--accent, #2f6fed); }
.sa-seam h3 { margin: 0 0 2px; font-size: 15px; }
.sa-metrics { color: var(--muted, #5d6578); font-size: 13px; margin: 0 0 8px; }
.sa-transport { display: flex; gap: 10px; align-items: center; }
.sa-track { position: relative; flex: 1; height: 28px; }
.sa-track input[type=range] { position: absolute; inset: 0; width: 100%; margin: 0; background: none; }
.sa-mark, .sa-band { position: absolute; top: 4px; bottom: 4px; pointer-events: none; }
.sa-mark { width: 2px; margin-left: -1px; background: var(--warn, #a15c00); }
.sa-band { background: var(--accent, #2f6fed); opacity: .22; border-radius: 3px; }
.sa-time { font-variant-numeric: tabular-nums; font-size: 13px; color: var(--muted, #5d6578); min-width: 5.5em; text-align: right; }
.sa-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 10px; }
.sa-row .sa-label { color: var(--muted, #5d6578); font-size: 13px; }
.sa-seam button { font: inherit; padding: 6px 14px; border-radius: 8px; cursor: pointer; border: 1px solid var(--line, #d9dde6); background: var(--card, #fff); color: var(--text, inherit); }
.sa-seam button.sa-on { background: var(--accent, #2f6fed); color: var(--accent-text, #fff); border-color: var(--accent, #2f6fed); font-weight: 600; }
.sa-seam button.sa-picked { box-shadow: inset 0 0 0 2px var(--ok, #1a7f4b); }
.sa-note { font-size: 13px; color: var(--muted, #5d6578); margin: 6px 0 0; }
.sa-note.sa-err { color: var(--err, #b3261e); }
.sa-summary { margin-top: 16px; }
.sa-summary p { margin: 4px 0; }
.sa-opts { display: flex; gap: 18px; flex-wrap: wrap; margin: 8px 0; font-size: 14px; }
.sa-opts label { display: flex; gap: 6px; align-items: center; }
`;

  function el(tag, props, ...kids) {
    const node = document.createElement(tag);
    for (const k in props || {}) {
      if (k === "class") node.className = props[k];
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), props[k]);
      else if (k in node && k !== "list") node[k] = props[k];
      else node.setAttribute(k, props[k]);
    }
    for (const kid of kids) if (kid != null) node.append(kid);
    return node;
  }

  function shuffled(list) {
    const a = list.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  const num = (v, d, sign) => (v === null || v === undefined ? "–" : (sign && v > 0 ? "+" : "") + Number(v).toFixed(d));
  function clock(s) { s = Math.max(0, s); const m = Math.floor(s / 60); return m + ":" + (s - m * 60).toFixed(1).padStart(4, "0"); }

  function mount(root, kit, options) {
    if (!document.getElementById("seamaudition-css")) {
      const style = el("style", { id: "seamaudition-css" });
      style.textContent = CSS;
      document.head.append(style);
    }
    const t = options.t;
    const answers = Object.assign({}, options.initial || {});
    let blind = true;
    let loop = true;
    let playingSeam = null; // the controller of the seam that is playing
    let summaryData = null;
    const controllers = [];

    root.textContent = "";
    const opts = el("div", { class: "sa-opts" },
      el("label", {}, el("input", { type: "checkbox", checked: true, onchange: (e) => { blind = e.target.checked; controllers.forEach((c) => c.render()); } }), t("blind")),
      el("label", {}, el("input", { type: "checkbox", checked: true, onchange: (e) => { loop = e.target.checked; } }), t("loop")));
    const summary = el("div", { class: "sa-summary" });
    root.append(el("p", { class: "muted" }, t("listenIntro")), opts);

    function kindName(variant) { return t("kind_" + variant.kind, { method: variant.id }); }

    function buildDoc() {
      const choices = Object.keys(answers).map(Number).sort((a, b) => a - b).map((n) => {
        const a = answers[n], seam = kit.seams.find((s) => s.seam === n);
        const variant = kit.variants.find((v) => v.id === a.choice);
        const search = kit.variants.find((v) => v.kind === "seam_search");
        return {
          seam: n, choice: a.choice, kind: a.choice === "tie" ? "tie" : variant.kind, blind: !!a.blind,
          at: a.at || new Date().toISOString(), metrics: seam.metrics,
          seam_search_identical: !!(search && seam.clips[search.id] && seam.clips[search.id].identical_to),
        };
      });
      return { format: 1, source_file: kit.source_file, created: kit.created, settings: kit.settings, window_s: kit.window_s, choices };
    }

    function renderSummary() {
      summary.textContent = "";
      const doc = buildDoc();
      summary.append(el("h3", {}, t("summaryTitle")));
      if (!doc.choices.length) {
        summary.append(el("p", { class: "muted" }, t("noAnswers")));
      } else {
        const counts = { straight: 0, crossfade: 0, seam_search: 0, tie: 0 };
        doc.choices.forEach((c) => { counts[c.kind] += 1; });
        summary.append(el("p", {}, t("counts", Object.assign({ n: doc.choices.length }, counts))));
        const notBlind = doc.choices.filter((c) => !c.blind).length;
        if (notBlind) summary.append(el("p", { class: "muted" }, t("nonBlind", { n: notBlind })));
        if (summaryData) {
          const sv = summaryData.seam_search_vs_crossfade, sp = summaryData.straight_vs_processed;
          summary.append(el("p", {}, t("svc", { seam_search: sv.seam_search, crossfade: sv.crossfade, p: sv.p_value.toFixed(3), verdict: t("v_" + sv.verdict) })));
          summary.append(el("p", {}, t("svp", { straight: sp.straight, processed: sp.processed, p: sp.p_value.toFixed(3), verdict: t("v_" + sp.verdict) })));
        }
        summary.append(el("p", { class: "muted" }, t("caveat")));
        if (options.docHref) {
          summary.append(el("a", { class: "btn", href: options.docHref() }, t("downloadAnswers")));
        } else if (options.download) {
          summary.append(el("button", { type: "button", onclick: () => options.download(buildDoc()) }, t("downloadAnswers")));
        }
      }
    }

    function seamController(seam) {
      const card = el("div", { class: "sa-seam", tabindex: 0 });
      const heading = el("h3", {});
      const metrics = el("p", { class: "sa-metrics" });
      const m = seam.metrics || {};
      heading.textContent = seam.at ? t("seamHeading", { seam: seam.seam, of: seam.of, at: seam.at }) : t("seamHeadingNoTime", { seam: seam.seam, of: seam.of });
      metrics.textContent = t("metrics", { agree: num(m.overlap_match_db, 1), lag: num(m.lag_ms, 1, true), gain: num(m.gain_db, 1, true) });

      const variants = kit.variants.filter((v) => !(seam.clips[v.id] && seam.clips[v.id].identical_to));
      const identical = variants.length < kit.variants.length;
      const ids = variants.map((v) => v.id);
      const blindOrder = shuffled(ids);
      let activeId = blindOrder[0];
      let loadFailed = false;
      const length = Math.max(...ids.map((id) => seam.clips[id].length_ms)) / 1000;
      const centre = seam.centre_ms / 1000;

      const audios = {};
      ids.forEach((id) => {
        const a = new Audio();
        a.preload = "auto";
        a.muted = id !== activeId;
        a.addEventListener("error", () => {
          if (loadFailed) return;
          loadFailed = true;
          note.className = "sa-note sa-err";
          note.textContent = t("loadError", { file: seam.clips[id].file });
        });
        a.src = options.url(seam.clips[id].file);
        audios[id] = a;
      });

      const playBtn = el("button", { type: "button", onclick: () => (playing ? pause() : play()) }, t("play"));
      const slider = el("input", { type: "range", min: 0, max: length, step: 0.01, value: 0, "aria-label": "position" });
      const mark = el("div", { class: "sa-mark" });
      mark.style.left = (100 * centre / length) + "%";
      const band = el("div", { class: "sa-band" });
      const time = el("span", { class: "sa-time" }, "0:00.0");
      const track = el("div", { class: "sa-track" }, band, mark, slider);
      const abRow = el("div", { class: "sa-row" });
      const pickRow = el("div", { class: "sa-row" });
      const note = el("p", { class: "sa-note" });
      let playing = false;
      let raf = 0;

      const master = () => audios[activeId];
      function seek(s) { ids.forEach((id) => { audios[id].currentTime = s; }); }
      function loopRegion() { return [Math.max(0, centre - 2), Math.min(length, centre + 2)]; }

      function tick() {
        const a = master();
        if (loop) {
          const [from, to] = loopRegion();
          if (a.currentTime >= to) seek(from);
        }
        ids.forEach((id) => { if (id !== activeId && Math.abs(audios[id].currentTime - a.currentTime) > 0.04) audios[id].currentTime = a.currentTime; });
        slider.value = a.currentTime;
        time.textContent = clock(a.currentTime) + " / " + clock(length);
        if (a.ended) { pause(); seek(0); return; }
        if (playing) raf = requestAnimationFrame(tick);
      }

      function play() {
        if (playingSeam && playingSeam !== ctl) playingSeam.pause();
        playingSeam = ctl;
        const a = master();
        if (loop) { const [from, to] = loopRegion(); if (a.currentTime < from || a.currentTime >= to) seek(from); }
        ids.forEach((id) => { audios[id].currentTime = a.currentTime; });
        Promise.all(ids.map((id) => audios[id].play())).catch(() => {});
        playing = true;
        playBtn.textContent = t("pause");
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(tick);
      }
      function pause() {
        playing = false;
        ids.forEach((id) => audios[id].pause());
        playBtn.textContent = t("play");
        cancelAnimationFrame(raf);
      }
      function select(id) {
        activeId = id;
        ids.forEach((other) => { audios[other].muted = other !== id; });
        render();
        if (!playing) { const a = master(); slider.value = a.currentTime; }
      }

      slider.addEventListener("input", () => { seek(Number(slider.value)); time.textContent = clock(Number(slider.value)) + " / " + clock(length); });

      function letterOf(id) { return "ABC"[blindOrder.indexOf(id)]; }

      function render() {
        const answer = answers[seam.seam];
        const revealed = !blind || !!answer;
        abRow.textContent = "";
        const slots = blind ? blindOrder : ids;
        slots.forEach((id) => {
          const variant = kit.variants.find((v) => v.id === id);
          const label = blind ? t("option", { letter: letterOf(id) }) : kindName(variant);
          abRow.append(el("button", { type: "button", class: id === activeId ? "sa-on" : "", onclick: () => select(id) }, label));
        });
        pickRow.textContent = "";
        pickRow.append(el("span", { class: "sa-label" }, t("best")));
        slots.forEach((id) => {
          const variant = kit.variants.find((v) => v.id === id);
          const label = blind ? letterOf(id) : kindName(variant);
          pickRow.append(el("button", { type: "button", class: answer && answer.choice === id ? "sa-picked" : "", onclick: () => answer_(id) }, label));
        });
        pickRow.append(el("button", { type: "button", class: answer && answer.choice === "tie" ? "sa-picked" : "", onclick: () => answer_("tie") }, t("tie")));
        if (!loadFailed) {
          note.className = "sa-note";
          const parts = [];
          if (identical) parts.push(t("same"));
          if (answer) {
            const chosen = kit.variants.find((v) => v.id === answer.choice);
            parts.push(answer.choice === "tie" ? t("answeredTie") : t("answered", { name: kindName(chosen) }));
          }
          if (blind && answer) {
            parts.push(blindOrder.map((id) => t("reveal", { letter: letterOf(id), name: kindName(kit.variants.find((v) => v.id === id)) + (seam.clips[id].match_db != null ? " (" + num(seam.clips[id].match_db, 1) + " dB)" : "") })).join(", "));
          }
          note.textContent = parts.join(" · ");
        }
        const clip = seam.clips[activeId];
        const showBand = revealed && clip.fade_ms > 0 && clip.seam_ms != null;
        band.style.display = showBand ? "" : "none";
        if (showBand) {
          band.style.left = (100 * (clip.seam_ms - clip.fade_ms / 2) / 1000 / length) + "%";
          band.style.width = (100 * clip.fade_ms / 1000 / length) + "%";
        }
      }

      async function answer_(choice) {
        const record = { choice, blind, at: new Date().toISOString() };
        answers[seam.seam] = record;
        render();
        try {
          const res = options.save ? await options.save(seam.seam, choice, blind) : null;
          if (res && res.summary) summaryData = res.summary;
        } catch (err) {
          note.className = "sa-note sa-err";
          note.textContent = t("saveError", { msg: err.message || err });
        }
        renderSummary();
      }

      card.addEventListener("keydown", (e) => {
        if (e.target !== card) return;
        if (e.key === " ") { e.preventDefault(); playing ? pause() : play(); }
        const k = Number(e.key);
        const slots = blind ? blindOrder : ids;
        if (k >= 1 && k <= slots.length) select(slots[k - 1]);
      });

      card.append(heading, metrics, el("div", { class: "sa-transport" }, playBtn, track, time), abRow, pickRow, note);
      const ctl = { card, render, pause };
      render();
      return ctl;
    }

    kit.seams.forEach((seam) => { const c = seamController(seam); controllers.push(c); root.append(c.card); });
    root.append(summary);
    renderSummary();
    return {
      answers: () => buildDoc(),
      refresh: (data) => { summaryData = data || summaryData; renderSummary(); },
    };
  }

  global.SeamAudition = { TEXT, mount };
})(window);
