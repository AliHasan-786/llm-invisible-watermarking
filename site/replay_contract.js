(function exposeReplayContract(root) {
  "use strict";

  function requireValue(condition, message) {
    if (!condition) throw new Error(`invalid_contract:${message}`);
  }

  function finiteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function rate(value) {
    return finiteNumber(value) && value >= 0 && value <= 1;
  }

  function validateScores(values, label, axis) {
    requireValue(Array.isArray(values) && values.length > 0, `${label}_scores`);
    values.forEach((value) => {
      requireValue(finiteNumber(value), `${label}_score_type`);
      requireValue(value >= axis.min && value <= axis.max, `${label}_score_axis`);
    });
  }

  function validateReplayContract(contract) {
    requireValue(contract && typeof contract === "object", "root");
    requireValue(contract.schemaVersion === 2, "schema_version");
    requireValue(contract.status === "historical_coursework_replay", "status");
    requireValue(contract.buildVerification?.status === "verified_by_exporter", "build_verification");
    requireValue(contract.buildVerification?.verifiedArtifactCount === 17, "artifact_count");
    requireValue(typeof contract.buildVerification?.verificationScope === "string", "verification_scope");
    requireValue(contract.provenance?.rawTextIncluded === false, "raw_text_boundary");
    requireValue(/^[a-f0-9]{64}$/.test(contract.provenance?.manifestSha256 || ""), "manifest_digest");
    requireValue(contract.synthId?.status === "not_run", "synthid_status");
    requireValue(contract.synthId?.scores === null, "synthid_scores");
    requireValue(contract.confirmatoryArticle50?.status === "not_run", "confirmatory_status");
    requireValue(contract.confirmatoryArticle50?.results === null, "confirmatory_results");

    const scoreAxis = contract.display?.scoreAxis;
    const rateAxis = contract.display?.rateAxis;
    requireValue(finiteNumber(scoreAxis?.min) && finiteNumber(scoreAxis?.max), "score_axis");
    requireValue(scoreAxis.min < scoreAxis.max && scoreAxis.unit === "z-score", "score_axis_range");
    requireValue(rateAxis?.min === 0 && rateAxis?.max === 1, "rate_axis");

    ["gemma", "llama"].forEach((key) => {
      const model = contract.models?.[key];
      requireValue(model && typeof model.modelId === "string", `${key}_model`);
      const clean = model.clean;
      requireValue(clean?.status === "measured_historical", `${key}_clean_status`);
      requireValue(finiteNumber(clean.threshold) && clean.threshold >= scoreAxis.min && clean.threshold <= scoreAxis.max, `${key}_threshold`);
      requireValue(rate(clean.tpr) && rate(clean.realizedFpr), `${key}_clean_rates`);
      requireValue(finiteNumber(clean.pplRatio) && clean.pplRatio > 0, `${key}_ppl_ratio`);
      requireValue(clean.nWatermarked > 0 && clean.nControl > 0, `${key}_clean_counts`);
      validateScores(clean.watermarkedScores, `${key}_watermarked`, scoreAxis);
      validateScores(clean.controlScores, `${key}_control`, scoreAxis);
      requireValue(clean.watermarkedScores.length === clean.nWatermarked, `${key}_watermarked_count`);
      requireValue(clean.controlScores.length === clean.nControl, `${key}_control_count`);

      requireValue(Array.isArray(model.lengthCells) && model.lengthCells.length > 0, `${key}_length_cells`);
      model.lengthCells.forEach((cell) => {
        requireValue(cell.status === "measured_historical", `${key}_length_status`);
        requireValue(cell.tokens > 0 && cell.nWatermarked > 0 && cell.nControl > 0, `${key}_length_counts`);
        requireValue(finiteNumber(cell.threshold) && cell.threshold >= scoreAxis.min && cell.threshold <= scoreAxis.max, `${key}_length_threshold`);
        requireValue(rate(cell.tpr) && rate(cell.fpr), `${key}_length_values`);
      });

      requireValue(Array.isArray(model.editCells) && model.editCells.length > 0, `${key}_edit_cells`);
      model.editCells.forEach((cell) => {
        requireValue(["measured_historical", "not_run"].includes(cell.status), `${key}_edit_status`);
        if (cell.status === "not_run") {
          requireValue(cell.n === null && cell.scores === null && cell.meanZ === null && cell.medianZ === null && cell.tprAtHistoricalThreshold === null, `${key}_not_run_nulls`);
          requireValue(cell.transformedTextState === "not_run", `${key}_not_run_text_state`);
          requireValue(cell.transformedTextCached === false && cell.transformationReconstructable === false, `${key}_not_run_boundaries`);
        } else {
          requireValue(cell.n > 0 && rate(cell.tprAtHistoricalThreshold), `${key}_edit_values`);
          requireValue(finiteNumber(cell.meanZ) && finiteNumber(cell.medianZ), `${key}_edit_summary`);
          validateScores(cell.scores, `${key}_${cell.condition}`, scoreAxis);
          requireValue(cell.scores.length === cell.n, `${key}_${cell.condition}_count`);
          const expectedTextState = cell.condition === "baseline" ? "not_applicable" : "unavailable";
          requireValue(cell.transformedTextState === expectedTextState, `${key}_text_state`);
          requireValue(cell.transformedTextCached === false, `${key}_text_cache_boundary`);
        }
      });
    });
    return contract;
  }

  async function loadReplayContract(fetchImpl, url) {
    let response;
    try {
      response = await fetchImpl(url, { cache: "no-store" });
    } catch (_error) {
      throw new Error("replay_unavailable:network");
    }
    if (!response || !response.ok) throw new Error(`replay_unavailable:http_${response?.status || "unknown"}`);
    let payload;
    try {
      payload = await response.json();
    } catch (_error) {
      throw new Error("replay_unavailable:malformed_json");
    }
    return validateReplayContract(payload);
  }

  const api = { loadReplayContract, validateReplayContract };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.ReplayContract = api;
}(typeof window !== "undefined" ? window : globalThis));
