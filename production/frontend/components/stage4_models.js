/**
 * Stage 4 — Model Building
 * Trains MLR, Ridge, Lasso; shows Spearman ρ as primary metric + R² / RMSE / MAE.
 * Best model is selected by LOOCV Spearman ρ.
 */
const stage4 = (() => {
  let container, config, featuresPath, selectedFeatures;
  let charts = [];

  /**
   * @param {string} containerId
   * @param {object} cfg - shared app config
   */
  function mount(containerId, cfg) {
    container = document.getElementById(containerId);
    config = cfg;
    renderIdle();
  }

  /** Store features path and selected feature list needed to build models. */
  function init(stage1Result, stage3Result) {
    const paths = stage1Result?.features_paths || {};
    featuresPath = Object.values(paths)[0] || null;
    selectedFeatures = stage3Result?.selected_features || [];
  }

  /** Show a notice that models were already built, allowing the user to re-run or proceed. */
  function onResume(stage4Result) {
    if (!container) return;
    const note = document.createElement('div');
    note.className = 'card p-4 mb-4 border-green-800 bg-green-950 text-green-300 text-sm';
    note.textContent = 'Models already built. You can re-run or proceed to Stage 5.';
    container.prepend(note);
  }

  /** Render the model-building form UI. */
  function renderIdle() {
    container.innerHTML = `
      <div class="card p-6">
        <h2 class="text-lg font-semibold text-white mb-1">Stage 4 — Model Building</h2>
        <p class="text-sm text-slate-400 mb-4">
          Train MLR, Ridge (CV), and Lasso (CV) on the team's half-match data.
          Best model selected by <strong class="text-white">LOOCV Spearman ρ</strong>.
        </p>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
          <div>
            <label>Target column</label>
            <input id="s4-target" type="text" value="bypasses_per_halftime" />
          </div>
          <div>
            <label>Test size</label>
            <input id="s4-test-size" type="number" value="0.15" step="0.05" min="0.05" max="0.4" />
          </div>
          <div>
            <label>Random state</label>
            <input id="s4-seed" type="number" value="42" />
          </div>
        </div>
        <button class="btn-primary" onclick="stage4.run()">Build Models</button>
      </div>
      <div id="s4-progress" class="card p-6 mt-4 hidden">
        <div class="flex justify-between text-sm mb-2">
          <span id="s4-msg" class="text-slate-300">Starting…</span>
          <span id="s4-pct" class="text-blue-400 font-mono">0%</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" id="s4-bar" style="width:0%"></div></div>
      </div>
      <div id="s4-error" class="card p-4 border-red-800 bg-red-950 text-red-300 text-sm mt-4 hidden"></div>
      <div id="s4-results" class="mt-6 space-y-6"></div>
    `;
  }

  /** POST the model build job and begin polling for results. */
  async function run() {
    if (!featuresPath) { alert('Complete Stage 1 first.'); return; }
    if (!selectedFeatures.length) { alert('Complete Stage 3 first.'); return; }

    const target = document.getElementById('s4-target').value.trim() || 'bypasses_per_halftime';
    const test_size = parseFloat(document.getElementById('s4-test-size').value) || 0.15;
    const random_state = parseInt(document.getElementById('s4-seed').value) || 42;

    document.getElementById('s4-progress').classList.remove('hidden');
    document.getElementById('s4-error').classList.add('hidden');

    try {
      const r = await fetch(`${config.API}/stage4/build`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features_path: featuresPath,
          selected_features: selectedFeatures,
          target_col: target,
          test_size,
          random_state,
        }),
      });
      const { job_id } = await r.json();

      config.pollUrl(
        `${config.API}/stage4/status/${job_id}`,
        (data) => {
          document.getElementById('s4-msg').textContent = data.message || '…';
          const pct = data.progress || 0;
          document.getElementById('s4-pct').textContent = `${pct}%`;
          document.getElementById('s4-bar').style.width = `${pct}%`;
        },
        (result) => {
          document.getElementById('s4-msg').textContent = 'Complete';
          document.getElementById('s4-pct').textContent = '100%';
          document.getElementById('s4-bar').style.width = '100%';
          renderResults(result);
          config.onDone(result);
        },
        (err) => {
          const errEl = document.getElementById('s4-error');
          errEl.textContent = `Error: ${err}`;
          errEl.classList.remove('hidden');
        }
      );
    } catch (e) {
      document.getElementById('s4-error').textContent = `Failed: ${e}`;
      document.getElementById('s4-error').classList.remove('hidden');
    }
  }

  /**
   * Return a colour-coded badge for a Spearman ρ value.
   * Thresholds: ≥ 0.5 = green (strong), ≥ 0.3 = yellow (moderate), < 0.3 = red (weak).
   */
  function spearmanBadge(v) {
    const val = typeof v === 'number' ? v : parseFloat(v);
    if (val >= 0.5)  return `<span class="badge badge-green">${val.toFixed(4)}</span>`;
    if (val >= 0.3)  return `<span class="badge badge-yellow">${val.toFixed(4)}</span>`;
    return `<span class="badge badge-red">${val.toFixed(4)}</span>`;
  }

  /** Render metrics table and Spearman ρ comparison chart for all models. */
  function renderResults(result) {
    charts.forEach(c => c.destroy());
    charts = [];

    const el = document.getElementById('s4-results');
    const models = result.models || [];

    el.innerHTML = `
      <div class="card p-4">
        <div class="flex items-center gap-3 mb-1">
          <span class="text-slate-400 text-sm">Best model (by LOOCV Spearman ρ):</span>
          <span class="badge badge-green text-sm">${result.best_model}</span>
          <span class="text-slate-400 text-sm ml-2">Features used: <strong class="text-white">${result.feature_count}</strong></span>
        </div>
      </div>

      <!-- Metrics table -->
      <div class="card p-6">
        <h3 class="text-sm font-semibold text-slate-300 mb-3">Model Metrics</h3>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead class="table-header">
              <tr>
                <th class="px-3 py-2 text-left text-slate-400">Model</th>
                <th class="px-3 py-2 text-center text-slate-400" colspan="4">LOOCV</th>
                <th class="px-3 py-2 text-center text-slate-400" colspan="4">Test Set</th>
              </tr>
              <tr>
                <th class="px-3 py-1 text-slate-500"></th>
                ${['Spearman ρ','R²','RMSE','MAE','Spearman ρ','R²','RMSE','MAE'].map(h =>
                  `<th class="px-3 py-1 text-slate-500 text-xs">${h}</th>`).join('')}
              </tr>
            </thead>
            <tbody>
              ${models.map((m, i) => `
                <tr class="${i % 2 === 0 ? 'bg-slate-800/50' : ''}">
                  <td class="px-3 py-2 font-semibold ${m.name === result.best_model ? 'text-green-400' : 'text-slate-300'}">${m.name}</td>
                  <td class="px-3 py-2 text-center">${spearmanBadge(m.loocv.spearman)}</td>
                  <td class="px-3 py-2 text-center text-slate-400 font-mono">${m.loocv.r2}</td>
                  <td class="px-3 py-2 text-center text-slate-400 font-mono">${m.loocv.rmse}</td>
                  <td class="px-3 py-2 text-center text-slate-400 font-mono">${m.loocv.mae}</td>
                  <td class="px-3 py-2 text-center">${spearmanBadge(m.test.spearman)}</td>
                  <td class="px-3 py-2 text-center text-slate-400 font-mono">${m.test.r2}</td>
                  <td class="px-3 py-2 text-center text-slate-400 font-mono">${m.test.rmse}</td>
                  <td class="px-3 py-2 text-center text-slate-400 font-mono">${m.test.mae}</td>
                </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Spearman ρ comparison chart -->
      <div class="card p-6">
        <h3 class="text-sm font-semibold text-slate-300 mb-4">Spearman ρ — LOOCV vs Test Set</h3>
        <div style="height:220px"><canvas id="s4-rho-chart"></canvas></div>
      </div>
    `;

    const ctx = document.getElementById('s4-rho-chart').getContext('2d');
    charts.push(new Chart(ctx, {
      type: 'bar',
      data: {
        labels: models.map(m => m.name),
        datasets: [
          { label: 'LOOCV Spearman ρ', data: models.map(m => m.loocv.spearman), backgroundColor: '#3b82f6aa', borderColor: '#3b82f6', borderWidth: 1 },
          { label: 'Test Spearman ρ',  data: models.map(m => m.test.spearman),  backgroundColor: '#10b981aa', borderColor: '#10b981', borderWidth: 1 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#94a3b8' } } },
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } },
          y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' }, min: -0.1, max: 1 },
        },
      },
    }));
  }

  return { mount, init, onResume, run };
})();
