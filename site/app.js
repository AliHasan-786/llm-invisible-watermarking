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

function readableCondition(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function metric(label, value, note = "") {
  return `<div class="replay-metric"><span>${label}</span><strong>${value}</strong>${note ? `<small>${note}</small>` : ""}</div>`;
}

function scorePlot(clean) {
  const all = [...clean.controlScores, ...clean.watermarkedScores, clean.threshold];
  const min = Math.min(...all);
  const max = Math.max(...all);
  const position = (value) => ((value - min) / (max - min)) * 100;
  const dots = (values, kind) => values.map((value, index) =>
    `<i class="score-dot ${kind}" aria-hidden="true" style="--x:${position(value)}%;--row:${index % 5}" title="${value.toFixed(3)}"></i>`
  ).join("");
  return `<div class="score-plot" aria-label="Cached detector score distribution">
    ${dots(clean.controlScores, "control")}${dots(clean.watermarkedScores, "marked")}
    <b class="score-threshold" style="--x:${position(clean.threshold)}%"><span>threshold ${clean.threshold.toFixed(3)}</span></b>
  </div><div class="plot-key"><span><i class="control"></i>Control scores</span><span><i class="marked"></i>Watermarked scores</span></div>`;
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
  replayCellLabel.hidden = view === "clean";
  if (view === "clean") {
    const clean = model.clean;
    replayPanel.innerHTML = `<div class="replay-metrics">
      ${metric("Cached TPR", percent(clean.tpr), `n=${clean.nWatermarked}`)}
      ${metric("Cached realized FPR", percent(clean.realizedFpr), `n=${clean.nControl}`)}
      ${metric("Historical threshold", clean.threshold.toFixed(3), "strict score > threshold")}
      ${metric("Perplexity ratio", clean.pplRatio.toFixed(3), "watermarked / control")}
    </div>${scorePlot(clean)}<p class="replay-note">${clean.calibrationLimitation}</p>`;
    return;
  }
  const cells = view === "length" ? model.lengthCells : model.editCells;
  const cell = cells[Number(replayCell.value || 0)];
  if (!cell || cell.status === "not_run") {
    replayPanel.innerHTML = `<div class="not-run"><strong>Not run</strong><p>No preserved evidence exists for this cell. Values remain null.</p></div>`;
    return;
  }
  if (view === "length") {
    replayPanel.innerHTML = `<div class="replay-metrics">
      ${metric("Cached TPR", percent(cell.tpr), `n=${cell.nWatermarked}`)}
      ${metric("Cached FPR", percent(cell.fpr), `n=${cell.nControl}`)}
      ${metric("Per-length threshold", cell.threshold.toFixed(3), `${cell.tokens} tokens`)}
    </div><p class="replay-note">This cell used its own threshold on the same historical cache. It is descriptive, not held-out confirmatory calibration.</p>`;
    return;
  }
  replayPanel.innerHTML = `<div class="replay-metrics">
    ${metric("TPR at historical threshold", percent(cell.tprAtHistoricalThreshold), `n=${cell.n}`)}
    ${metric("Mean z-score", cell.meanZ.toFixed(3))}
    ${metric("Median z-score", cell.medianZ.toFixed(3))}
    ${metric("Transformed text cache", cell.transformedTextCached ? "Present" : "Missing", cell.transformationReconstructable ? "deterministic attack can be rerun" : "scores only")}
  </div><div class="score-strip">${cell.scores.map((value) => `<i style="--x:${Math.max(0, Math.min(100, ((value + 3) / 13) * 100))}%"></i>`).join("")}</div>
  <p class="replay-note">The score strip replays preserved values at the fixed historical clean threshold; it does not run a detector.</p>`;
}

async function loadReplay() {
  try {
    const response = await fetch("/site/data/historical_replay.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    replayData = await response.json();
    if (replayData.provenance.verifiedArtifactCount !== 17 || replayData.synthId.status !== "not_run") {
      throw new Error("unexpected evidence contract");
    }
    replayStatus.textContent = `Historical replay · ${replayData.provenance.verifiedArtifactCount} artifacts verified · SynthID not run`;
    renderReplay();
  } catch (error) {
    replayStatus.textContent = "Cached evidence unavailable; no fallback values are shown.";
    replayPanel.innerHTML = `<div class="not-run"><strong>Replay unavailable</strong><p>${error.message}</p></div>`;
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
