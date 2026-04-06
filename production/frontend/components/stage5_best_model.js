/**
 * Stage 5 — Best Model Analysis
 * Shows coefficients (horizontal bar) and gradient sensitivity.
 */
const stage5 = (() => {
  let container, config, modelPath, scalerPath, featuresPath, selectedFeatures;
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

  /**
   * Resolve paths for the best model, scaler, and features from prior stage results.
   * model_path is read from models[i].model_path, matched by best_model name.
   */
  function init(stage4Result, stage3Result, stage1Result) {
    const best = stage4Result.best_model;
    const models = stage4Result.models || [];
    const match = models.find(m => m.name === best) || models[0];
    modelPath = match?.model_path || null;
    scalerPath = stage4Result.scaler_path;
    featuresPath = Object.values(stage1Result?.features_paths || {})[0] || null;
    selectedFeatures = stage3Result?.selected_features || [];
  }

  /** Render the analysis form, pre-filling paths if available. */
  function renderIdle() {
    container.innerHTML = `
      <div class="card p-6">
        <h2 class="text-lg font-semibold text-white mb-1">Stage 5 — Best Model Analysis</h2>
        <p class="text-sm text-slate-400 mb-4">Extract coefficients and gradient sensitivity from the best linear model.</p>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
          <div>
            <label>Model path (auto-filled)</label>
            <input id="s5-model" type="text" placeholder="models/lasso_model.pkl" />
          </div>
          <div>
            <label>Scaler path (auto-filled)</label>
            <input id="s5-scaler" type="text" placeholder="models/scaler.pkl" />
          </div>
          <div>
            <label>Features path (auto-filled)</label>
            <input id="s5-features" type="text" placeholder="data/processed/…_features.csv" />
          </div>
          <div>
            <label>Target column</label>
            <input id="s5-target" type="text" value="bypasses_per_halftime" />
          </div>
        </div>
        <button class="btn-primary" onclick="stage5.run()">Analyze Model</button>
      </div>
      <div id="s5-progress" class="card p-6 mt-4 hidden">
        <div class="flex justify-between text-sm mb-2">
          <span id="s5-msg" class="text-slate-300">Starting…</span>
          <span id="s5-pct" class="text-blue-400 font-mono">0%</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" id="s5-bar" style="width:0%"></div></div>
      </div>
      <div id="s5-error" class="card p-4 border-red-800 bg-red-950 text-red-300 text-sm mt-4 hidden"></div>
      <div id="s5-results" class="mt-6 space-y-6"></div>
    `;

    if (modelPath) document.getElementById('s5-model').value = modelPath;
    if (scalerPath) document.getElementById('s5-scaler').value = scalerPath;
    if (featuresPath) document.getElementById('s5-features').value = featuresPath;
  }

  /** POST the best-model analysis job and begin polling. */
  async function run() {
    const model_path = document.getElementById('s5-model').value.trim();
    const scaler_path = document.getElementById('s5-scaler').value.trim();
    const features_path = document.getElementById('s5-features').value.trim();
    const target_col = document.getElementById('s5-target').value.trim() || 'bypasses_per_halftime';

    if (!model_path || !scaler_path || !features_path) {
      alert('Model path, scaler path, and features path are required.');
      return;
    }
    if (!selectedFeatures.length) { alert('Complete Stage 3 first.'); return; }

    document.getElementById('s5-progress').classList.remove('hidden');

    try {
      const r = await fetch(`${config.API}/stage5/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path, scaler_path, features_path, selected_features: selectedFeatures, target_col }),
      });
      const { job_id } = await r.json();

      config.pollUrl(
        `${config.API}/stage5/status/${job_id}`,
        (data) => {
          document.getElementById('s5-msg').textContent = data.message || '…';
          const pct = data.progress || 0;
          document.getElementById('s5-pct').textContent = `${pct}%`;
          document.getElementById('s5-bar').style.width = `${pct}%`;
        },
        (result) => {
          document.getElementById('s5-msg').textContent = 'Complete';
          document.getElementById('s5-pct').textContent = '100%';
          document.getElementById('s5-bar').style.width = '100%';
          renderResults(result);
          config.onDone(result);
        },
        (err) => {
          const errEl = document.getElementById('s5-error');
          errEl.textContent = `Error: ${err}`;
          errEl.classList.remove('hidden');
        }
      );
    } catch (e) {
      document.getElementById('s5-error').textContent = `Failed: ${e}`;
      document.getElementById('s5-error').classList.remove('hidden');
    }
  }

  /** Render coefficient bar chart, coefficients table, and gradient sensitivity table. */
  function renderResults(result) {
    charts.forEach(c => c.destroy());
    charts = [];

    const el = document.getElementById('s5-results');
    const coefs = result.coefficients || [];
    const grad = result.gradient_sensitivity || [];

    el.innerHTML = `
      <div class="card p-4">
        <span class="text-slate-400 text-sm">Model: </span>
        <span class="badge badge-blue">${result.model_name || '—'}</span>
      </div>

      <!-- Coefficients bar chart -->
      <div class="card p-6">
        <h3 class="text-sm font-semibold text-slate-300 mb-4">Feature Coefficients (by magnitude)</h3>
        <div style="height:${Math.max(220, coefs.length * 32)}px"><canvas id="s5-coef-chart"></canvas></div>
      </div>

      <!-- Coefficients table -->
      <div class="card p-6">
        <h3 class="text-sm font-semibold text-slate-300 mb-3">Coefficients</h3>
        <table class="w-full text-sm">
          <thead class="table-header"><tr>
            <th class="px-3 py-2 text-left text-slate-400">Feature</th>
            <th class="px-3 py-2 text-right text-slate-400">Coefficient</th>
            <th class="px-3 py-2 text-right text-slate-400">Relative Importance %</th>
          </tr></thead>
          <tbody>
            ${coefs.map((c, i) => `
              <tr class="${i%2===0 ? 'bg-slate-800/50' : ''}">
                <td class="px-3 py-1.5 text-slate-300 font-mono">${c.feature}</td>
                <td class="px-3 py-1.5 text-right font-mono ${c.coefficient >= 0 ? 'text-green-400' : 'text-red-400'}">${c.coefficient}</td>
                <td class="px-3 py-1.5 text-right text-slate-400 font-mono">${c.relative_importance}%</td>
              </tr>`).join('')}
          </tbody>
        </table>
      </div>

      <!-- Gradient sensitivity table -->
      <div class="card p-6">
        <h3 class="text-sm font-semibold text-slate-300 mb-3">Gradient Sensitivity (coef × raw std)</h3>
        <table class="w-full text-sm">
          <thead class="table-header"><tr>
            <th class="px-3 py-2 text-left text-slate-400">Feature</th>
            <th class="px-3 py-2 text-right text-slate-400">Sensitivity</th>
          </tr></thead>
          <tbody>
            ${grad.map((g, i) => `
              <tr class="${i%2===0 ? 'bg-slate-800/50' : ''}">
                <td class="px-3 py-1.5 text-slate-300 font-mono">${g.feature}</td>
                <td class="px-3 py-1.5 text-right font-mono ${g.sensitivity >= 0 ? 'text-green-400' : 'text-red-400'}">${g.sensitivity}</td>
              </tr>`).join('')}
          </tbody>
        </table>
      </div>
    `;

    const ctx = document.getElementById('s5-coef-chart').getContext('2d');
    charts.push(new Chart(ctx, {
      type: 'bar',
      data: {
        labels: coefs.map(c => c.feature),
        datasets: [{
          label: 'Coefficient',
          data: coefs.map(c => c.coefficient),
          backgroundColor: coefs.map(c => c.coefficient >= 0 ? '#10b98180' : '#ef444480'),
          borderColor: coefs.map(c => c.coefficient >= 0 ? '#10b981' : '#ef4444'),
          borderWidth: 1,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
          y: { ticks: { color: '#cbd5e1', font: { size: 11 } }, grid: { color: '#1e293b' } }
        }
      }
    }));
  }

  return { mount, init, run };
})();
