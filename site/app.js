const textarea = document.querySelector("#sampleText");
const scoreValue = document.querySelector("#scoreValue");
const verdict = document.querySelector("#verdict");
const wordCount = document.querySelector("#wordCount");
const greenHits = document.querySelector("#greenHits");
const beforeScore = document.querySelector("#beforeScore");
const afterScore = document.querySelector("#afterScore");
const deltaCopy = document.querySelector("#deltaCopy");
const threshold = 2.33;

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
