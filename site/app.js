const textarea = document.querySelector("#sampleText");
const scoreValue = document.querySelector("#scoreValue");
const verdict = document.querySelector("#verdict");
const wordCount = document.querySelector("#wordCount");
const greenHits = document.querySelector("#greenHits");
const beforeScore = document.querySelector("#beforeScore");
const afterScore = document.querySelector("#afterScore");
const deltaCopy = document.querySelector("#deltaCopy");
const threshold = 2.33;

const replayModel = document.querySelector("#replayModel");
const replayView = document.querySelector("#replayView");
const replayCell = document.querySelector("#replayCell");
const replayCellLabel = document.querySelector("#replayCellLabel");
const replayStatus = document.querySelector("#replayStatus");
const replayPanel = document.querySelector("#replayPanel");
let replayData = null;

function percent(value) {
  return value === null ? "Not run" : `${(value * 100).toFixed(1)}%`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function readableCondition(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function metric(label, value, note = "") {
  return `<div class="replay-metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong>${note ? `<small>${escapeHtml(note)}</small>` : ""}</div>`;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function scorePosition(value, axis) {
  return ((value - axis.min) / (axis.max - axis.min)) * 100;
}

function scoreAxis(axis) {
  const zero = scorePosition(0, axis);
  return `<div class="plot-axis" aria-hidden="true"><span>${axis.min.toFixed(0)}</span><span style="--x:${zero}%">0</span><span>${axis.max.toFixed(0)} z</span></div>
    <p class="axis-disclosure">Fixed ${axis.min.toFixed(0)} to ${axis.max.toFixed(0)} z-score axis across both models and every clean/edit score view. No values are clipped.</p>`;
}

function scorePlot(clean, axis) {
  const dots = (values, kind) => values.map((value, index) =>
    `<i class="score-dot ${kind}" aria-hidden="true" style="--x:${scorePosition(value, axis)}%;--row:${index % 5}"></i>`
  ).join("");
  return `<div class="score-plot" role="img" aria-label="Cached detector score distribution on a fixed ${axis.min} to ${axis.max} z-score axis">
    ${dots(clean.controlScores, "control")}${dots(clean.watermarkedScores, "marked")}
    <b class="score-threshold" style="--x:${scorePosition(clean.threshold, axis)}%"><span>threshold ${clean.threshold.toFixed(3)}</span></b>
  </div>${scoreAxis(axis)}<div class="plot-key"><span><i class="control"></i>Control scores (diamonds)</span><span><i class="marked"></i>Watermarked scores (circles)</span></div>`;
}

function scoreSummaryRows(clean) {
  const groups = [
    ["Control", clean.controlScores],
    ["Watermarked", clean.watermarkedScores],
  ];
  return groups.map(([label, values]) => `<tr><th scope="row">${label}</th><td>${values.length}</td><td>${Math.min(...values).toFixed(3)}</td><td>${median(values).toFixed(3)}</td><td>${Math.max(...values).toFixed(3)}</td></tr>`).join("");
}

function tableShell(caption, headings, rows) {
  const safeCaption = escapeHtml(caption);
  return `<div class="table-wrap" role="region" aria-label="${safeCaption}" tabindex="0"><table><caption>${safeCaption}</caption><thead><tr>${headings.map((heading) => `<th scope="col">${escapeHtml(heading)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

function cleanTable(model) {
  const clean = model.clean;
  return tableShell(
    `Numeric detector-score summary for ${model.modelId}`,
    ["Series", "n", "Minimum z", "Median z", "Maximum z"],
    scoreSummaryRows(clean),
  );
}

function lengthTable(model, selectedIndex) {
  const rows = model.lengthCells.map((cell, index) => `<tr${index === selectedIndex ? ' class="selected-row"' : ""}><th scope="row">${cell.tokens}</th><td>${percent(cell.tpr)}</td><td>${percent(cell.fpr)}</td><td>${cell.threshold.toFixed(3)}</td><td>${cell.nWatermarked}</td><td>${cell.nControl}</td><td>Measured historical</td></tr>`).join("");
  return tableShell(
    `All historical length cells for ${model.modelId}`,
    ["Tokens", "TPR", "FPR", "Threshold z", "Watermarked n", "Control n", "State"],
    rows,
  );
}

function editTable(model, selectedIndex) {
  const rows = model.editCells.map((cell, index) => {
    const measured = cell.status === "measured_historical";
    const textState = cell.transformedTextState === "not_applicable" ? "Not applicable" : measured ? "Unavailable" : "Not run";
    return `<tr${index === selectedIndex ? ' class="selected-row"' : ""}><th scope="row">${escapeHtml(readableCondition(cell.condition))}</th><td>${measured ? "Measured historical" : "Not run"}</td><td>${cell.n ?? "—"}</td><td>${percent(cell.tprAtHistoricalThreshold)}</td><td>${cell.meanZ === null ? "—" : cell.meanZ.toFixed(3)}</td><td>${cell.medianZ === null ? "—" : cell.medianZ.toFixed(3)}</td><td>${textState}</td></tr>`;
  }).join("");
  return tableShell(
    `All historical edit cells for ${model.modelId}`,
    ["Condition", "Evidence state", "n", "TPR", "Mean z", "Median z", "Transformed text"],
    rows,
  );
}

function scoreStrip(cell, axis) {
  const ticks = cell.scores.map((value) => `<i aria-hidden="true" style="--x:${scorePosition(value, axis)}%"></i>`).join("");
  return `<div class="score-strip" role="img" aria-label="${cell.n} cached scores on a fixed ${axis.min} to ${axis.max} z-score axis">${ticks}</div>${scoreAxis(axis)}`;
}

function populateReplayCells(model, view) {
  replayCell.replaceChildren();
  const cells = view === "length" ? model.lengthCells : model.editCells;
  cells.forEach((cell, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = view === "length" ? `${cell.tokens} tokens` : readableCondition(cell.condition);
    replayCell.append(option);
  });
}

function renderReplay() {
  if (!replayData) return;
  const model = replayData.models[replayModel.value];
  const view = replayView.value;
  const scoreScale = replayData.display.scoreAxis;
  replayCellLabel.hidden = view === "clean";
  if (view === "clean") {
    const clean = model.clean;
    replayPanel.innerHTML = `<div class="replay-metrics">
      ${metric("Cached TPR", percent(clean.tpr), `n=${clean.nWatermarked}`)}
      ${metric("Cached realized FPR", percent(clean.realizedFpr), `n=${clean.nControl}`)}
      ${metric("Historical threshold", clean.threshold.toFixed(3), "strict score > threshold")}
      ${metric("Perplexity ratio", clean.pplRatio.toFixed(3), "watermarked / control")}
    </div>${scorePlot(clean, scoreScale)}${cleanTable(model)}<p class="replay-note">${escapeHtml(clean.calibrationLimitation)}</p>`;
    return;
  }
  const cells = view === "length" ? model.lengthCells : model.editCells;
  const selectedIndex = Number(replayCell.value || 0);
  const cell = cells[selectedIndex];
  if (!cell || cell.status === "not_run") {
    replayPanel.innerHTML = `<div class="not-run"><strong>Not run</strong><p>This model-condition experiment was not run in the preserved study. Numeric values are null; this is different from an unavailable transformed-text artifact.</p></div>${view === "edit" ? editTable(model, selectedIndex) : ""}`;
    return;
  }
  if (view === "length") {
    const tprPosition = window.ReplayContract.ratePercent(cell.tpr);
    const fprPosition = window.ReplayContract.ratePercent(cell.fpr);
    replayPanel.innerHTML = `<div class="replay-metrics">
      ${metric("Cached TPR", percent(cell.tpr), `n=${cell.nWatermarked}`)}
      ${metric("Cached FPR", percent(cell.fpr), `n=${cell.nControl}`)}
      ${metric("Per-length threshold", cell.threshold.toFixed(3), `${cell.tokens} tokens`)}
    </div><div class="rate-plot" role="img" aria-label="TPR ${percent(cell.tpr)} and FPR ${percent(cell.fpr)} on a fixed zero to one-hundred-percent scale"><i class="rate-tick tpr" aria-hidden="true" style="--value:${tprPosition}%"></i><span class="rate-label tpr" style="--label-value:${tprPosition}%">TPR ${percent(cell.tpr)}</span><i class="rate-tick fpr" aria-hidden="true" style="--value:${fprPosition}%"></i><span class="rate-label fpr" style="--label-value:${fprPosition}%">FPR ${percent(cell.fpr)}</span></div><div class="rate-axis"><span>0%</span><span>50%</span><span>100%</span></div>${lengthTable(model, selectedIndex)}<p class="replay-note">Every rate tick stays at its exact value on a fixed 0–100% scale; only the text label is inset at the edges. This historical cell used its own threshold on the same cache; it is descriptive, not held-out confirmatory calibration.</p>`;
    return;
  }
  const transformedTextValue = cell.transformedTextState === "not_applicable" ? "Not applicable" : "Unavailable";
  const transformedTextNote = cell.transformedTextState === "not_applicable"
    ? "unmodified historical completion"
    : cell.transformationReconstructable ? "deterministic edit can be reconstructed" : "scores only; cannot reconstruct";
  const availabilityNote = cell.transformedTextState === "not_applicable"
    ? "Baseline is the unmodified historical completion; transformed text does not apply."
    : "Measured scores are available; transformed attack text is unavailable.";
  replayPanel.innerHTML = `<div class="replay-metrics">
    ${metric("TPR at historical threshold", percent(cell.tprAtHistoricalThreshold), `n=${cell.n}`)}
    ${metric("Mean z-score", cell.meanZ.toFixed(3))}
    ${metric("Median z-score", cell.medianZ.toFixed(3))}
    ${metric("Transformed text", transformedTextValue, transformedTextNote)}
  </div>${scoreStrip(cell, scoreScale)}${editTable(model, selectedIndex)}
  <p class="replay-note">${availabilityNote} The strip uses the fixed shared score axis and the historical clean threshold. It does not run a detector.</p>`;
}

async function loadReplay() {
  try {
    replayData = await window.ReplayContract.loadReplayContract(fetch, "/historical-replay.json");
    replayStatus.textContent = `Exporter-verified replay · ${replayData.buildVerification.verifiedArtifactCount} source artifacts · browser contract valid · SynthID not run`;
    renderReplay();
  } catch (error) {
    replayData = null;
    replayStatus.textContent = "Replay unavailable · saved evidence contract did not load or validate";
    replayPanel.innerHTML = `<div class="not-run unavailable"><strong>Unavailable</strong><p>The saved evidence contract could not be validated. No fallback values or headline metrics are shown.</p><small>${escapeHtml(error.message)}</small></div>`;
  }
}

replayModel.addEventListener("change", () => {
  if (!replayData) return;
  if (replayView.value !== "clean") populateReplayCells(replayData.models[replayModel.value], replayView.value);
  renderReplay();
});
replayView.addEventListener("change", () => {
  if (!replayData) return;
  if (replayView.value !== "clean") populateReplayCells(replayData.models[replayModel.value], replayView.value);
  renderReplay();
});
replayCell.addEventListener("change", renderReplay);
loadReplay();

const vocabulary = [
  "evidence", "systems", "signals", "models", "policy", "robust", "public",
  "careful", "measured", "reliable", "language", "content", "technical",
  "context", "research", "design", "method", "results", "people", "tools",
  "evaluate", "explain", "record", "compare", "detect", "preserve", "change"
];

function hash(value) {
  let result = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    result ^= value.charCodeAt(i);
    result = Math.imul(result, 16777619);
  }
  return result >>> 0;
}

function tokenize(text) {
  return text.toLowerCase().match(/[a-z0-9']+/g) || [];
}

function isGreen(previous, current) {
  return hash(`42:${previous}:${current}`) % 2 === 0;
}

function score(text) {
  const words = tokenize(text);
  if (words.length < 2) return { z: 0, total: 0, green: 0 };
  let green = 0;
  for (let i = 1; i < words.length; i += 1) {
    if (isGreen(words[i - 1], words[i])) green += 1;
  }
  const total = words.length - 1;
  return { z: (green - total * 0.5) / Math.sqrt(total * 0.25), total, green };
}

function render() {
  const result = score(textarea.value);
  scoreValue.textContent = result.z.toFixed(2);
  wordCount.textContent = String(result.total);
  greenHits.textContent = String(result.green);
  verdict.textContent = result.total < 8 ? "Not enough text" : result.z > threshold ? "Signal detected" : "Below threshold";
  verdict.classList.toggle("detected", result.total >= 8 && result.z > threshold);
  return result;
}

function generatedSample() {
  const words = ["watermark"];
  for (let index = 0; index < 95; index += 1) {
    const previous = words[words.length - 1];
    const candidates = vocabulary.filter((word) => isGreen(previous, word));
    words.push(candidates[index % candidates.length]);
  }
  return words.join(" ");
}

function lightEdit(words) {
  return words.map((word, index) => index % 10 === 4 ? vocabulary[(hash(word) + index) % vocabulary.length] : word);
}

function paraphraseProxy(words) {
  const replacements = { evidence: "proof", systems: "services", signals: "marks", models: "engines", policy: "rules", robust: "durable", measured: "tested", reliable: "steady", language: "text", content: "material", research: "study", results: "findings", people: "readers", tools: "methods" };
  const changed = words.map((word) => replacements[word] || word);
  const chunks = [];
  for (let index = 0; index < changed.length; index += 5) chunks.push(changed.slice(index, index + 5));
  return chunks.map((chunk, index) => index % 2 ? chunk.reverse() : chunk).flat();
}

function translationProxy(words) {
  const map = { evidence: "indication", systems: "platforms", signals: "indicators", policy: "guidance", careful: "deliberate", measured: "quantified", language: "wording", context: "setting", method: "procedure", compare: "contrast", detect: "identify", preserve: "retain", change: "alter" };
  return words.map((word, index) => map[word] || (index % 7 === 0 ? vocabulary[(index * 3) % vocabulary.length] : word));
}

function applyAttack(kind) {
  const initial = score(textarea.value);
  const words = tokenize(textarea.value);
  let attacked = words;
  if (kind === "light") attacked = lightEdit(words);
  if (kind === "paraphrase") attacked = paraphraseProxy(words);
  if (kind === "translate") attacked = translationProxy(words);
  if (kind === "truncate") attacked = words.slice(0, Math.max(1, Math.floor(words.length / 2)));
  textarea.value = attacked.join(" ");
  const final = render();
  beforeScore.textContent = initial.z.toFixed(2);
  afterScore.textContent = final.z.toFixed(2);
  const delta = final.z - initial.z;
  deltaCopy.textContent = `${kind.replace(/\b\w/g, (letter) => letter.toUpperCase())}: ${delta >= 0 ? "+" : ""}${delta.toFixed(2)} z-score change.`;
}

document.querySelector("#generateButton").addEventListener("click", () => {
  textarea.value = generatedSample();
  beforeScore.textContent = "—";
  afterScore.textContent = "—";
  deltaCopy.textContent = "Choose a transformation to compare scores.";
  render();
});

document.querySelectorAll("[data-attack]").forEach((button) => {
  button.addEventListener("click", () => applyAttack(button.dataset.attack));
});

textarea.addEventListener("input", render);
textarea.value = generatedSample();
render();
