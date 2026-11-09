(function exposeReplayContract(root) {
  "use strict";

  const TOLERANCE = 1e-12;
  const MODEL_KEYS = ["gemma", "llama"];
  const MODEL_IDS = {
    gemma: "google/gemma-2-9b-it",
    llama: "meta-llama/Llama-3.1-8B-Instruct",
  };
  const LENGTH_TOKENS = [25, 50, 75, 100, 125, 150, 175, 200];
  const EDIT_CONDITIONS = [
    "baseline",
    "word_sub_5pct",
    "word_sub_10pct",
    "word_sub_15pct",
    "word_sub_20pct",
    "token_deletion_10pct",
    "token_insertion_10pct",
    "llm_paraphrase",
  ];

  function requireValue(condition, message) {
    if (!condition) throw new Error(`invalid_contract:${message}`);
  }

  function finiteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function positiveInteger(value) {
    return Number.isInteger(value) && value > 0;
  }

  function nonEmptyString(value) {
    return typeof value === "string" && value.trim().length > 0;
  }

  function rate(value) {
    return finiteNumber(value) && value >= 0 && value <= 1;
  }

  function close(actual, expected) {
    return Math.abs(actual - expected) <= TOLERANCE;
  }

  function mean(values) {
    return values.reduce((total, value) => total + value, 0) / values.length;
  }

  function median(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  function strictRate(values, threshold) {
    return values.filter((value) => value > threshold).length / values.length;
  }

  function exactSequence(actual, expected, label) {
    requireValue(Array.isArray(actual) && actual.length === expected.length, `${label}_count`);
    requireValue(actual.every((value, index) => value === expected[index]), `${label}_sequence`);
  }

  function validateScores(values, label, axis) {
    requireValue(Array.isArray(values) && values.length > 0, `${label}_scores`);
    values.forEach((value) => {
      requireValue(finiteNumber(value), `${label}_score_type`);
      requireValue(value >= axis.min && value <= axis.max, `${label}_score_axis`);
    });
  }

  function validateReplayContract(contract) {
    requireValue(contract && typeof contract === "object" && !Array.isArray(contract), "root");
    requireValue(contract.schemaVersion === 2, "schema_version");
    requireValue(/^\d{4}-\d{2}-\d{2}$/.test(contract.asOf || ""), "as_of");
    requireValue(contract.status === "historical_coursework_replay", "status");
    requireValue(nonEmptyString(contract.scope), "scope");

    const provenance = contract.provenance;
    requireValue(provenance?.manifestPath === "results/manifest.json", "manifest_path");
    requireValue(/^[a-f0-9]{64}$/.test(provenance?.manifestSha256 || ""), "manifest_digest");
    requireValue(provenance?.verifiedArtifactCount === 17, "provenance_artifact_count");
    requireValue(provenance?.rawTextIncluded === false, "raw_text_boundary");
    requireValue(nonEmptyString(provenance?.source) && nonEmptyString(provenance?.rights), "provenance_copy");

    const verification = contract.buildVerification;
    requireValue(verification?.status === "verified_by_exporter", "build_verification");
    requireValue(verification?.exporter === "scripts/export_historical_replay.py", "exporter");
    requireValue(verification?.verifiedArtifactCount === 17, "artifact_count");
    requireValue(nonEmptyString(verification?.verificationScope), "verification_scope");

    const scoreAxis = contract.display?.scoreAxis;
    const rateAxis = contract.display?.rateAxis;
    requireValue(finiteNumber(scoreAxis?.min) && finiteNumber(scoreAxis?.max), "score_axis");
    requireValue(scoreAxis.min < scoreAxis.max && scoreAxis.unit === "z-score", "score_axis_range");
    requireValue(nonEmptyString(scoreAxis.scope), "score_axis_scope");
    requireValue(rateAxis?.min === 0 && rateAxis?.max === 1 && rateAxis?.unit === "proportion", "rate_axis");
    requireValue(nonEmptyString(rateAxis.scope), "rate_axis_scope");

    requireValue(contract.models && typeof contract.models === "object", "models");
    exactSequence(Object.keys(contract.models).sort(), [...MODEL_KEYS].sort(), "model_keys");

    MODEL_KEYS.forEach((key) => {
      const model = contract.models[key];
      requireValue(model?.modelId === MODEL_IDS[key], `${key}_model_id`);
      requireValue(model.scheme === "kirchenbauer_greenlist", `${key}_scheme`);
      requireValue(model.configuration?.delta === 2 && model.configuration?.gamma === 0.5, `${key}_configuration`);
      requireValue(model.configuration?.seed === 42, `${key}_seed`);

      const clean = model.clean;
      requireValue(clean?.status === "measured_historical", `${key}_clean_status`);
      requireValue(finiteNumber(clean.threshold) && clean.threshold >= scoreAxis.min && clean.threshold <= scoreAxis.max, `${key}_threshold`);
      requireValue(clean.thresholdRule === "historical_empirical_upper_tail", `${key}_threshold_rule`);
      requireValue(nonEmptyString(clean.calibrationLimitation), `${key}_calibration_limitation`);
      requireValue(rate(clean.tpr) && rate(clean.realizedFpr), `${key}_clean_rates`);
      requireValue(finiteNumber(clean.pplRatio) && clean.pplRatio > 0, `${key}_ppl_ratio`);
      requireValue(positiveInteger(clean.nWatermarked) && positiveInteger(clean.nControl), `${key}_clean_counts`);
      validateScores(clean.watermarkedScores, `${key}_watermarked`, scoreAxis);
      validateScores(clean.controlScores, `${key}_control`, scoreAxis);
      requireValue(clean.watermarkedScores.length === clean.nWatermarked, `${key}_watermarked_count`);
      requireValue(clean.controlScores.length === clean.nControl, `${key}_control_count`);
      requireValue(close(clean.tpr, strictRate(clean.watermarkedScores, clean.threshold)), `${key}_clean_tpr_consistency`);
      requireValue(close(clean.realizedFpr, strictRate(clean.controlScores, clean.threshold)), `${key}_clean_fpr_consistency`);

      requireValue(Array.isArray(model.lengthCells), `${key}_length_cells`);
      exactSequence(model.lengthCells.map((cell) => cell.tokens), LENGTH_TOKENS, `${key}_length_tokens`);
      model.lengthCells.forEach((cell) => {
        requireValue(cell.status === "measured_historical", `${key}_length_status`);
        requireValue(positiveInteger(cell.tokens) && positiveInteger(cell.nWatermarked) && positiveInteger(cell.nControl), `${key}_length_counts`);
        requireValue(finiteNumber(cell.threshold) && cell.threshold >= scoreAxis.min && cell.threshold <= scoreAxis.max, `${key}_length_threshold`);
        requireValue(rate(cell.tpr) && rate(cell.fpr), `${key}_length_values`);
        requireValue(cell.calibration === "per_length_same_cache", `${key}_length_calibration`);
      });

      requireValue(Array.isArray(model.editCells), `${key}_edit_cells`);
      exactSequence(model.editCells.map((cell) => cell.condition), EDIT_CONDITIONS, `${key}_edit_conditions`);
      model.editCells.forEach((cell) => {
        requireValue(nonEmptyString(cell.condition), `${key}_condition`);
        requireValue(["measured_historical", "not_run"].includes(cell.status), `${key}_edit_status`);
        if (cell.status === "not_run") {
          requireValue(key === "gemma" && cell.condition === "llm_paraphrase", `${key}_unexpected_not_run`);
          requireValue(cell.n === null && cell.scores === null && cell.meanZ === null && cell.medianZ === null && cell.tprAtHistoricalThreshold === null, `${key}_not_run_nulls`);
          requireValue(cell.transformedTextState === "not_run", `${key}_not_run_text_state`);
          requireValue(cell.transformedTextCached === false && cell.transformationReconstructable === false, `${key}_not_run_boundaries`);
        } else {
          requireValue(positiveInteger(cell.n) && rate(cell.tprAtHistoricalThreshold), `${key}_edit_values`);
          requireValue(finiteNumber(cell.meanZ) && finiteNumber(cell.medianZ), `${key}_edit_summary`);
          validateScores(cell.scores, `${key}_${cell.condition}`, scoreAxis);
          requireValue(cell.scores.length === cell.n, `${key}_${cell.condition}_count`);
          requireValue(close(cell.tprAtHistoricalThreshold, strictRate(cell.scores, clean.threshold)), `${key}_${cell.condition}_tpr_consistency`);
          requireValue(close(cell.meanZ, mean(cell.scores)), `${key}_${cell.condition}_mean_consistency`);
          requireValue(close(cell.medianZ, median(cell.scores)), `${key}_${cell.condition}_median_consistency`);
          const expectedTextState = cell.condition === "baseline" ? "not_applicable" : "unavailable";
          const expectedReconstructable = !["baseline", "llm_paraphrase"].includes(cell.condition);
          requireValue(cell.transformedTextState === expectedTextState, `${key}_text_state`);
          requireValue(cell.transformedTextCached === false, `${key}_text_cache_boundary`);
          requireValue(cell.transformationReconstructable === expectedReconstructable, `${key}_reconstructable`);
        }
      });
      const baseline = model.editCells[0];
      requireValue(baseline.scores.length === clean.watermarkedScores.length, `${key}_baseline_count`);
      requireValue(baseline.scores.every((value, index) => value === clean.watermarkedScores[index]), `${key}_baseline_scores`);
    });

    requireValue(contract.synthId?.status === "not_run", "synthid_status");
    ["threshold", "scores", "lengthCells", "editCells"].forEach((field) => {
      requireValue(contract.synthId[field] === null, `synthid_${field}`);
    });
    requireValue(nonEmptyString(contract.synthId.reason), "synthid_reason");
    requireValue(contract.confirmatoryArticle50?.status === "not_run", "confirmatory_status");
    requireValue(contract.confirmatoryArticle50?.calibratedFpr === null && contract.confirmatoryArticle50?.results === null, "confirmatory_nulls");
    requireValue(nonEmptyString(contract.confirmatoryArticle50.reason), "confirmatory_reason");
    return contract;
  }

  function ratePercent(value) {
    requireValue(rate(value), "rate_geometry");
    return value * 100;
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

  const api = { loadReplayContract, ratePercent, validateReplayContract };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.ReplayContract = api;
}(typeof window !== "undefined" ? window : globalThis));
