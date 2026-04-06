/**
 * Stage 3 — Feature Selection
 * Shows ranked features from 4 methods + consensus, lets user adjust selected set.
 */
const stage3 = (() => {
  let container, config, featuresPath;
  let stageResult = null;

  /**
   * @param {string} containerId
   * @param {object} cfg - shared app config
   */
  function mount(containerId, cfg) {
    container = document.getElementById(containerId);
    config = cfg;
    renderIdle();
  }

  /** Store the features CSV path from stage 1. */
  function init(stage1Result) {
    featuresPath = Object.values(stage1Result?.features_paths || {})[0] || null;
  }

  /** Render the idle state UI before feature selection has been run. */
  function renderIdle() {
    container.innerHTML = `
      <div class="card p-6">
        <h2 class="text-lg font-semibold text-white mb-1">Stage 3 — Feature Selection</h2>
        <p class="text-sm text-slate-400 mb-4">F-regression, Mutual Information, Random Forest, and RFECV combined into a consensus ranking.</p>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
          <div>
            <label>Target column</label>
            <input id="s3-target" type="text" value="bypasses_per_halftime" />
          </div>
          <div>
            <label>Top N features</label>
            <input id="s3-top" type="number" value="10" min="3" max="30" />
          </div>
        </div>
        <button class="btn-primary" onclick="stage3.run()">Run Feature Selection</button>
      </div>
      <div id="s3-progress" class="card p-6 mt-4 hidden">
        <div class="flex justify-between text-sm mb-2">
          <span id="s3-msg" class="text-slate-300">Starting…</span>
          <span id="s3-pct" class="text-blue-400 font-mono">0%</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" id="s3-bar" style="width:0%"></div></div>
      </div>
      <div id="s3-error" class="card p-4 border-red-800 bg-red-950 text-red-300 text-sm mt-4 hidden"></div>
      <div id="s3-results" class="mt-6 space-y-6"></div>
    `;
  }

  /** POST the feature selection job and begin polling. */
  async function run() {
    if (!featuresPath) { alert('Complete Stage 1 first.'); return; }
    const target = document.getElementById('s3-target').value.trim() || 'bypasses_per_halftime';
    const n_top = parseInt(document.getElementById('s3-top').value) || 10;

    document.getElementById('s3-progress').classList.remove('hidden');

    try {
      const r = await fetch(`${config.API}/stage3/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features_path: featuresPath, target_col: target, n_top }),
      });
      const { job_id } = await r.json();

      config.pollUrl(
        `${config.API}/stage3/status/${job_id}`,
        (data) => {
          document.getElementById('s3-msg').textContent = data.message || '…';
          const pct = data.progress || 0;
          document.getElementById('s3-pct').textContent = `${pct}%`;
          document.getElementById('s3-bar').style.width = `${pct}%`;
        },
        (result) => {
          document.getElementById('s3-msg').textContent = 'Complete';
          document.getElementById('s3-pct').textContent = '100%';
          document.getElementById('s3-bar').style.width = '100%';
          stageResult = result;
          renderResults(result);
        },
        (err) => {
          const errEl = document.getElementById('s3-error');
          errEl.textContent = `Error: ${err}`;
          errEl.classList.remove('hidden');
        }
      );
    } catch (e) {
      document.getElementById('s3-error').textContent = `Failed: ${e}`;
      document.getElementById('s3-error').classList.remove('hidden');
    }
  }

  /**
   * Render a ranked feature table card.
   * @param {string} scoreKey - the result object key to use as the score column
   */
  function renderFeatureTable(title, items, scoreKey) {
    if (!items || !items.length) return '';
    return `
      <div class="card p-4">
        <h3 class="text-sm font-semibold text-slate-300 mb-3">${title}</h3>
        <table class="w-full text-xs">
          <thead class="table-header"><tr>
            <th class="px-2 py-1.5 text-left text-slate-400">#</th>
            <th class="px-2 py-1.5 text-left text-slate-400">Feature</th>
            <th class="px-2 py-1.5 text-right text-slate-400">Score</th>
          </tr></thead>
          <tbody>
            ${items.map((it, i) => `
              <tr class="${i%2===0 ? 'bg-slate-800/50' : ''}">
                <td class="px-2 py-1 text-slate-500">${it.rank ?? i+1}</td>
                <td class="px-2 py-1 text-slate-300 font-mono">${it.feature}</td>
                <td class="px-2 py-1 text-right text-blue-400 font-mono">${(it[scoreKey] ?? it.score ?? 0).toFixed(4)}</td>
              </tr>`).join('')}
          </tbody>
        </table>
      </div>`;
  }

  /** Render consensus chips and per-method ranking tables. */
  function renderResults(result) {
    const el = document.getElementById('s3-results');
    el.innerHTML = `
      <!-- Consensus + selected -->
      <div class="card p-6">
        <h3 class="text-sm font-semibold text-slate-300 mb-3">Consensus Selected Features</h3>
        <div class="flex flex-wrap gap-2 mb-4" id="s3-chips">
          ${(result.selected_features || []).map(f => `
            <span class="badge badge-blue flex items-center gap-1">
              ${f}
              <button onclick="stage3.removeFeature('${f}')" class="ml-1 text-blue-200 hover:text-white">×</button>
            </span>`).join('')}
        </div>
        <button class="btn-primary" onclick="stage3.confirm()">Use these features →</button>
      </div>

      <!-- Method tables -->
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
        ${renderFeatureTable('Univariate F-regression', result.univariate, 'score')}
        ${renderFeatureTable('Mutual Information', result.mutual_info, 'score')}
        ${renderFeatureTable('Random Forest Importance', result.random_forest, 'score')}
        ${renderFeatureTable('RFECV Selected', result.rfe, 'score')}
      </div>

      <!-- Consensus ranking -->
      ${renderFeatureTable('Consensus Ranking (avg rank across methods)', result.consensus, 'avg_rank')}
    `;
  }

  /** Remove a feature from the selected set and re-render the chip list. */
  function removeFeature(f) {
    if (!stageResult) return;
    stageResult.selected_features = stageResult.selected_features.filter(x => x !== f);
    const chips = document.getElementById('s3-chips');
    if (chips) chips.innerHTML = (stageResult.selected_features || []).map(feat => `
      <span class="badge badge-blue flex items-center gap-1">
        ${feat}
        <button onclick="stage3.removeFeature('${feat}')" class="ml-1 text-blue-200 hover:text-white">×</button>
      </span>`).join('');
  }

  /** Forward the current stageResult (including any user edits) to onDone. */
  function confirm() {
    if (!stageResult) return;
    config.onDone(stageResult);
  }

  return { mount, init, run, removeFeature, confirm };
})();
