"use strict";

// 输入校验与结果渲染全部以服务端 API 为准；前端只做轻量预检与展示。
// 任何输入改动都会立即标记当前结论失效。

const form = document.getElementById("calib-form");
const taA = document.getElementById("times_a");
const taB = document.getElementById("times_b");
const loEl = document.getElementById("offset_lo");
const hiEl = document.getElementById("offset_hi");
const tolEl = document.getElementById("tolerance");
const minEl = document.getElementById("min_pairs");
const submitBtn = document.getElementById("submit-btn");
const staleNote = document.getElementById("stale-note");
const errorNote = document.getElementById("error-note");
const resultEl = document.getElementById("result");
const bannerEl = document.getElementById("result-banner");
const bodyEl = document.getElementById("result-body");

let lastSubmittedFingerprint = null;

function parseTimes(text) {
  const parts = text.split(/[\s,;]+/).filter(Boolean);
  const values = [];
  for (const p of parts) {
    if (!/^[+-]?\d+$/.test(p)) return { error: `无法解析为整数: ${p}` };
    values.push(parseInt(p, 10));
  }
  if (values.length < 6 || values.length > 24) {
    return { error: `需要 6 至 24 个时间，当前 ${values.length} 个` };
  }
  for (let i = 1; i < values.length; i++) {
    if (values[i] <= values[i - 1]) {
      return { error: `第 ${i + 1} 个时间不严格递增` };
    }
  }
  return { values };
}

function fingerprint() {
  return [taA.value, taB.value, loEl.value, hiEl.value, tolEl.value, minEl.value].join("|");
}

function markStale() {
  if (lastSubmittedFingerprint !== null && fingerprint() !== lastSubmittedFingerprint) {
    staleNote.hidden = false;
    resultEl.classList.add("stale");
    resultEl.style.opacity = "0.55";
  }
}

[taA, taB, loEl, hiEl, tolEl, minEl].forEach((el) => {
  el.addEventListener("input", markStale);
});

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function renderResult(data) {
  const meets = data.meets_threshold;
  bannerEl.innerHTML = "";
  bodyEl.innerHTML = "";

  const banner = document.createElement("div");
  banner.className = "banner " + (meets ? "ok" : "fail");
  if (meets) {
    banner.textContent =
      `校准可行：在偏移 ${data.offset} ns 处形成 ${data.pair_count} 对符合事件，` +
      `残差绝对值和 ${data.sum_abs_residual}，最大残差 ${data.max_abs_residual}。`;
  } else {
    // 关键：不伪造校准值，如实展示实际最大配对数与原因
    banner.innerHTML =
      `无法形成足够符合事件：实际最大配对数为 <strong>${data.pair_count}</strong>，` +
      `低于门槛 ${data.min_pairs}。` +
      `<span class="reason">${escapeHtml(data.reason)}</span>`;
  }
  bannerEl.appendChild(banner);

  // 无论是否达标都展示最优尝试的完整数值，便于质控人员判断
  const summary = document.createElement("div");
  summary.className = "summary";
  summary.innerHTML = `
    <dl>
      <dt>联合优选偏移</dt><dd>${meets ? data.offset : "—（未达标，不给出校准值）"}</dd>
      <dt>实际配对数</dt><dd>${data.pair_count}</dd>
      <dt>残差绝对值总和</dt><dd>${data.pair_count > 0 ? data.sum_abs_residual : "—"}</dd>
      <dt>最大残差</dt><dd>${data.max_abs_residual === null ? "—" : data.max_abs_residual}</dd>
    </dl>`;
  bodyEl.appendChild(summary);

  if (data.pairs.length > 0) {
    const table = document.createElement("table");
    table.innerHTML = `
      <thead><tr>
        <th>A序号</th><th>B序号</th><th>校正后 A 时间</th><th>B 时间</th>
        <th>带符号残差</th><th>|残差|</th>
      </tr></thead>`;
    const tbody = document.createElement("tbody");
    for (const p of data.pairs) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${p.index_a}</td><td>${p.index_b}</td>
        <td>${p.time_a_corrected}</td><td>${p.time_b}</td>
        <td>${p.residual >= 0 ? "+" : ""}${p.residual}</td>
        <td>${p.abs_residual}</td>`;
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    bodyEl.appendChild(table);
  }

  const up = document.createElement("div");
  up.className = "unpaired";
  up.innerHTML =
    `未配对 A（序号）：<code>${data.unpaired_a.length ? data.unpaired_a.join(", ") : "无"}</code><br>` +
    `未配对 B（序号）：<code>${data.unpaired_b.length ? data.unpaired_b.join(", ") : "无"}</code>`;
  bodyEl.appendChild(up);
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  errorNote.hidden = true;

  const aParsed = parseTimes(taA.value);
  if (aParsed.error) {
    errorNote.textContent = `探头 A：${aParsed.error}`;
    errorNote.hidden = false;
    return;
  }
  const bParsed = parseTimes(taB.value);
  if (bParsed.error) {
    errorNote.textContent = `探头 B：${bParsed.error}`;
    errorNote.hidden = false;
    return;
  }
  const intFields = [
    [loEl, "偏移下界"], [hiEl, "偏移上界"],
    [tolEl, "符合容差"], [minEl, "最低配对数"],
  ];
  for (const [el, name] of intFields) {
    if (!/^[+-]?\d+$/.test(el.value.trim())) {
      errorNote.textContent = `${name} 必须是整数`;
      errorNote.hidden = false;
      return;
    }
  }

  const payload = {
    times_a: aParsed.values,
    times_b: bParsed.values,
    offset_lo: parseInt(loEl.value, 10),
    offset_hi: parseInt(hiEl.value, 10),
    tolerance: parseInt(tolEl.value, 10),
    min_pairs: parseInt(minEl.value, 10),
  };

  submitBtn.disabled = true;
  submitBtn.textContent = "计算中…";
  try {
    const resp = await fetch("/api/calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || `服务端返回 ${resp.status}`);
    }
    lastSubmittedFingerprint = fingerprint();
    staleNote.hidden = true;
    resultEl.style.opacity = "1";
    resultEl.classList.remove("stale");
    resultEl.hidden = false;
    renderResult(data);
  } catch (err) {
    errorNote.textContent = err.message;
    errorNote.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "提交校准";
  }
});
