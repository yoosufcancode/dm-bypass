/**
 * app.js — top-level state machine for the DM-Bypass pipeline UI.
 *
 * State passed between stages:
 *   stage1Result  → { features_paths: {league: path}, row_counts: {}, leagues_processed: [] }
 *   stage3Result  → { selected_features, ... }
 *   stage4Result  → { models, best_model, scaler_path, feature_count }
 *   stage5Result  → { coefficients, gradient_sensitivity, model_name }
 *   stage6Result  → { league, team, model_selected, spearman_test, squad, recommendations, ... }
 *
 * Stage 6 is self-contained — it does not depend on stage3/stage4 results.
 */

const API = '/api/v1';

const app = (() => {
  let currentStage = 1;
  const unlockedStages = new Set([1]);

  const state = {
    stage1Result: null,
    stage3Result: null,
    stage4Result: null,
    stage5Result: null,
  };

  /**
   * Poll a stage-specific status endpoint by job ID.
   * @deprecated prefer pollUrl which accepts a full URL and is stage-agnostic
   */
  async function pollJob(jobId, onProgress, onDone, onError) {
    const interval = setInterval(async () => {
      try {
        const r = await fetch(`${API}/stage${currentStage}/status/${jobId}`);
        const data = await r.json();
        onProgress(data);
        if (data.status === 'completed') {
          clearInterval(interval);
          onDone(data.result);
        } else if (data.status === 'failed') {
          clearInterval(interval);
          onError(data.error || data.message);
        }
      } catch (e) {
        clearInterval(interval);
        onError(String(e));
      }
    }, 2000);
  }

  /**
   * Poll a full status URL every 2 s until the job completes or fails.
   * @param {string} statusUrl - complete GET endpoint for the job status
   */
  async function pollUrl(statusUrl, onProgress, onDone, onError) {
    const interval = setInterval(async () => {
      try {
        const r = await fetch(statusUrl);
        const data = await r.json();
        onProgress(data);
        if (data.status === 'completed') {
          clearInterval(interval);
          onDone(data.result);
        } else if (data.status === 'failed') {
          clearInterval(interval);
          onError(data.error || data.message);
        }
      } catch (e) {
        clearInterval(interval);
        onError(String(e));
      }
    }, 2000);
  }

  /** Navigate to stage n if it has been unlocked. */
  function goToStage(n) {
    if (!unlockedStages.has(n)) return;
    currentStage = n;

    document.querySelectorAll('.stage-panel').forEach(el => el.classList.add('hidden'));
    document.getElementById(`stage-${n}`).classList.remove('hidden');

    document.querySelectorAll('.stage-tab').forEach(btn => {
      const s = parseInt(btn.dataset.stage);
      btn.classList.remove('active');
      if (s === n) btn.classList.add('active');
    });
  }

  /** Unlock a stage tab so the user can navigate to it. */
  function unlockStage(n) {
    unlockedStages.add(n);
    const btn = document.querySelector(`.stage-tab[data-stage="${n}"]`);
    if (btn) {
      btn.classList.remove('locked');
      btn.classList.add('pending');
    }
  }

  /** Mark a stage tab as completed (applies the done CSS class). */
  function markStageDone(n) {
    const btn = document.querySelector(`.stage-tab[data-stage="${n}"]`);
    if (btn) {
      btn.classList.remove('active', 'pending');
      btn.classList.add('done');
    }
  }

  /** Ping the health endpoint and update the status indicator in the header. */
  async function checkHealth() {
    try {
      const r = await fetch(`${API}/health`);
      const data = await r.json();
      const dot = document.getElementById('health-indicator');
      const label = document.getElementById('health-label');
      if (data.status === 'ok') {
        dot.className = 'w-2 h-2 rounded-full bg-green-400';
        label.textContent = 'API connected';
      }
    } catch {
      const dot = document.getElementById('health-indicator');
      dot.className = 'w-2 h-2 rounded-full bg-red-500';
      document.getElementById('health-label').textContent = 'API unreachable';
    }
  }

  /**
   * Fetch /pipeline/state and restore any stages that are already complete on disk,
   * unlocking tabs and re-initialising stage components accordingly.
   */
  async function resumeFromPipelineState() {
    try {
      const r = await fetch(`${API}/pipeline/state`);
      const data = await r.json();
      const stages = data.stages || {};

      if (stages['1']) {
        state.stage1Result = {
          features_paths: data.features_paths || {},
          row_counts:     data.row_counts || {},
          leagues_processed: data.leagues_processed || [],
        };
        unlockStage(2);
        markStageDone(1);
      }
      if (stages['2']) { unlockStage(3); markStageDone(2); }
      if (stages['3']) {
        state.stage3Result = { selected_features: data.selected_features || [] };
        unlockStage(4);
        markStageDone(3);
      }
      if (stages['4']) {
        const modelPaths = data.model_paths || {};
        // Map file stems (e.g. "mlr_model") to display names used by the UI
        const nameMap = { mlr: 'MLR', ridge: 'Ridge', lasso: 'Lasso' };
        const modelEntries = Object.entries(modelPaths).map(([stem, path]) => {
          const key = stem.replace(/_model$/, '').toLowerCase();
          return { name: nameMap[key] || stem, model_path: path };
        });
        state.stage4Result = {
          models:     modelEntries,
          best_model: modelEntries[0]?.name || '',
          scaler_path: data.scaler_path || '',
          feature_count: 0,
        };
        unlockStage(5);
        markStageDone(4);
      }
      if (stages['5']) unlockStage(6);

      if (window.stage1 && stages['1']) stage1.onResume(state.stage1Result, data);
      if (window.stage3 && stages['3']) {
        if (window.stage4) stage4.init(state.stage1Result, state.stage3Result);
      }
      if (window.stage4 && stages['4']) {
        stage4.onResume(state.stage4Result);
        if (window.stage5) stage5.init(state.stage4Result, state.stage3Result, state.stage1Result);
      }
    } catch (e) {
      console.warn('Could not load pipeline state:', e);
    }
  }

  /** Called by stage 1 on success; unlocks stages 2 and 6. */
  function onStage1Done(result) {
    state.stage1Result = result;
    markStageDone(1);
    unlockStage(2);
    unlockStage(6);  // Stage 6 only needs Wyscout CSVs, available as soon as stage 1 is done
    if (window.stage2) stage2.init(result);
  }

  /** Called by stage 2 on success; unlocks stage 3. */
  function onStage2Done(result) {
    markStageDone(2);
    unlockStage(3);
    if (window.stage3) stage3.init(state.stage1Result);
  }

  /** Called by stage 3 on success; unlocks stage 4. */
  function onStage3Done(result) {
    state.stage3Result = result;
    markStageDone(3);
    unlockStage(4);
    if (window.stage4) stage4.init(state.stage1Result, result);
  }

  /** Called by stage 4 on success; unlocks stage 5. */
  function onStage4Done(result) {
    state.stage4Result = result;
    markStageDone(4);
    unlockStage(5);
    if (window.stage5) stage5.init(result, state.stage3Result, state.stage1Result);
  }

  /** Called by stage 5 on success; unlocks stage 6. */
  function onStage5Done(result) {
    state.stage5Result = result;
    markStageDone(5);
    unlockStage(6);
    if (window.stage6) stage6.init();
  }

  /** Called by stage 6 on success. */
  function onStage6Done(result) {
    markStageDone(6);
  }

  /** Bootstrap the app: check API health, restore pipeline state, and mount all stages. */
  async function init() {
    await checkHealth();
    await resumeFromPipelineState();

    // Mount each stage component
    if (window.stage1) stage1.mount('stage-1', { onDone: onStage1Done, pollUrl, API });
    if (window.stage2) stage2.mount('stage-2', { onDone: onStage2Done, pollUrl, API });
    if (window.stage3) stage3.mount('stage-3', { onDone: onStage3Done, pollUrl, API, stage1Result: state.stage1Result });
    if (window.stage4) stage4.mount('stage-4', { onDone: onStage4Done, pollUrl, API });
    if (window.stage5) stage5.mount('stage-5', { onDone: onStage5Done, pollUrl, API });
    if (window.stage6) stage6.mount('stage-6', { onDone: onStage6Done, pollUrl, API });
  }

  document.addEventListener('DOMContentLoaded', init);

  return { goToStage, state, pollUrl, API };
})();
