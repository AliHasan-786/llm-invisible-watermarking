const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const { loadReplayContract, validateReplayContract } = require("../../site/replay_contract.js");
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
