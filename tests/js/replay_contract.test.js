const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const { loadReplayContract, ratePercent, validateReplayContract } = require("../../site/replay_contract.js");
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, "../../site/data/historical_replay.json"), "utf8"));

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

test("accepts the deterministic tracked contract", () => {
  assert.equal(validateReplayContract(clone(fixture)).schemaVersion, 2);
});

test("rejects a contract that only copies the artifact count", () => {
  const malformed = { provenance: { verifiedArtifactCount: 17 } };
  assert.throws(() => validateReplayContract(malformed), /invalid_contract:schema_version/);
});

test("rejects score values outside the declared shared axis", () => {
  const malformed = clone(fixture);
  malformed.models.gemma.clean.watermarkedScores[0] = malformed.display.scoreAxis.max + 1;
  assert.throws(() => validateReplayContract(malformed), /score_axis/);
});

test("rejects a displayed clean rate that contradicts its score array", () => {
  const malformed = clone(fixture);
  malformed.models.gemma.clean.tpr = 0;
  assert.throws(() => validateReplayContract(malformed), /clean_tpr_consistency/);
});

test("rejects nonnumeric and noninteger sample counts", () => {
  const malformed = clone(fixture);
  malformed.models.gemma.lengthCells[0].nControl = "201";
  assert.throws(() => validateReplayContract(malformed), /length_counts/);
  const fractional = clone(fixture);
  fractional.models.llama.clean.nWatermarked = 99.5;
  assert.throws(() => validateReplayContract(fractional), /clean_counts/);
});

test("rejects missing, duplicate, and unnamed expected cells", () => {
  const missing = clone(fixture);
  missing.models.gemma.editCells.splice(1, 1);
  assert.throws(() => validateReplayContract(missing), /edit_conditions_count/);
  const duplicate = clone(fixture);
  duplicate.models.llama.editCells[1].condition = "baseline";
  assert.throws(() => validateReplayContract(duplicate), /edit_conditions_sequence/);
  const unnamed = clone(fixture);
  unnamed.models.gemma.modelId = "";
  assert.throws(() => validateReplayContract(unnamed), /model_id/);
  const substituted = clone(fixture);
  substituted.models.gemma.modelId = "provider/lookalike-model";
  assert.throws(() => validateReplayContract(substituted), /model_id/);
});

test("rejects contradictory edit summaries and malformed score arrays", () => {
  const summary = clone(fixture);
  summary.models.llama.editCells[1].meanZ = 0;
  assert.throws(() => validateReplayContract(summary), /mean_consistency/);
  const array = clone(fixture);
  array.models.gemma.clean.controlScores = "not-an-array";
  assert.throws(() => validateReplayContract(array), /control_scores/);
  const baseline = clone(fixture);
  baseline.models.llama.editCells[0].scores.pop();
  baseline.models.llama.editCells[0].n -= 1;
  baseline.models.llama.editCells[0].meanZ = baseline.models.llama.editCells[0].scores.reduce((sum, value) => sum + value, 0) / baseline.models.llama.editCells[0].scores.length;
  const sortedBaseline = [...baseline.models.llama.editCells[0].scores].sort((a, b) => a - b);
  baseline.models.llama.editCells[0].medianZ = (sortedBaseline[48] + sortedBaseline[49]) / 2;
  baseline.models.llama.editCells[0].tprAtHistoricalThreshold = baseline.models.llama.editCells[0].scores.filter((value) => value > baseline.models.llama.clean.threshold).length / baseline.models.llama.editCells[0].scores.length;
  assert.throws(() => validateReplayContract(baseline), /baseline_count/);
});

test("rejects not-run cells that contain fabricated numeric evidence", () => {
  const malformed = clone(fixture);
  const cell = malformed.models.gemma.editCells.find((item) => item.status === "not_run");
  cell.n = 1;
  cell.scores = [0];
  assert.throws(() => validateReplayContract(malformed), /not_run_nulls/);
});

test("maps HTTP, malformed JSON, and network failures to unavailable states", async () => {
  await assert.rejects(
    loadReplayContract(async () => ({ ok: false, status: 503 }), "/replay.json"),
    /replay_unavailable:http_503/,
  );
  await assert.rejects(
    loadReplayContract(async () => ({ ok: true, json: async () => { throw new Error("bad"); } }), "/replay.json"),
    /replay_unavailable:malformed_json/,
  );
  await assert.rejects(
    loadReplayContract(async () => { throw new Error("offline"); }, "/replay.json"),
    /replay_unavailable:network/,
  );
});

test("rate geometry keeps ticks at exact zero, one, and one hundred percent", () => {
  assert.equal(ratePercent(0), 0);
  assert.equal(ratePercent(0.01), 1);
  assert.equal(ratePercent(1), 100);
  assert.throws(() => ratePercent(1.01), /rate_geometry/);
});
