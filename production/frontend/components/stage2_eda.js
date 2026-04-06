/**
 * Stage 2 — Exploratory Data Analysis
 * Shows descriptive stats, correlation heatmap, bypass distribution.
 */
const stage2 = (() => {
  let container, config, featuresPath;
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

  /** Store the features CSV path from stage 1 for use when running EDA. */
  function init(stage1Result) {
    featuresPath = Object.values(stage1Result?.features_paths || {})[0] || null;
  }

  /** Render the idle state UI before any EDA has been run. */
  function renderIdle() {
    container.innerHTML = `
      <div class="card p-6">
        <h2 class="text-lg font-semibold text-white mb-1">Stage 2 — Exploratory Data Analysis</h2>
        <p class="text-sm text-slate-400 mb-4">Compute descriptive statistics, correlation matrix, and bypass distribution.</p>
        <button class="btn-primary" onclick="stage2.run()">Run EDA</button>
      </div>
      <div id="s2-progress" class="card p-6 mt-4 hidden">
        <div class="flex justify-between text-sm mb-2">
          <span id="s2-msg" class="text-slate-300">Starting…</span>
          <span id="s2-pct" class="text-blue-400 font-mono">0%</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" id="s2-bar" style="width:0%"></div></div>
      </div>
      <div id="s2-error" class="card p-4 border-red-800 bg-red-950 text-red-300 text-sm mt-4 hidden"></div>
      <div id="s2-results" class="mt-6 space-y-6"></div>
    `;
  }

  /** POST the EDA job and begin polling for results. */
  async function run() {
    if (!featuresPath) { alert('Complete Stage 1 first.'); return; }
    document.getElementById('s2-progress').classList.remove('hidden');

    try {
      const r = await fetch(`${config.API}/stage2/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features_path: featuresPath }),
      });
      const { job_id } = await r.json();

      config.pollUrl(
        `${config.API}/stage2/status/${job_id}`,
        (data) => {
          document.getElementById('s2-msg').textContent = data.message || '…';
          const pct = data.progress || 0;
          document.getElementById('s2-pct').textContent = `${pct}%`;
          document.getElementById('s2-bar').style.width = `${pct}%`;
        },
        (result) => {
          document.getElementById('s2-msg').textContent = 'Complete';
          document.getElementById('s2-pct').textContent = '100%';
          document.getElementById('s2-bar').style.width = '100%';
          renderResults(result);
          config.onDone(result);
        },
        (err) => {
          const errEl = document.getElementById('s2-error');
          errEl.textContent = `Error: ${err}`;
          errEl.classList.remove('hidden');
        }
      );
    } catch (e) {
      document.getElementById('s2-error').textContent = `Failed: ${e}`;
      document.getElementById('s2-error').classList.remove('hidden');
    }
  }

  /** Render summary cards, bypass distribution chart, heatmap, and stats table. */
  function renderResults(result) {
    charts.forEach(c => c.destroy());
    charts = [];

    const el = document.getElementById('s2-results');

    el.innerHTML = `
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div class="card p-4 text-center">
          <div class="text-2xl font-bold text-blue-400">${result.row_count ?? '—'}</div>
          <div class="text-xs text-slate-400 mt-1">Player-match rows</div>
        </div>
        <div class="card p-4 text-center">
          <div class="text-2xl font-bold text-blue-400">${result.column_count ?? '—'}</div>
          <div class="text-xs text-slate-400 mt-1">Columns</div>
        </div>
        <div class="card p-4 text-center">
          <div class="text-2xl font-bold text-blue-400">${result.bypass_distribution?.mean?.toFixed(2) ?? '—'}</div>
          <div class="text-xs text-slate-400 mt-1">Mean bypasses/half</div>
        </div>
        <div class="card p-4 text-center">
          <div class="text-2xl font-bold text-blue-400">${Object.keys(result.missing_values || {}).length}</div>
          <div class="text-xs text-slate-400 mt-1">Cols w/ missing vals</div>
        </div>
      </div>

      <!-- Bypass distribution chart -->
      ${result.bypass_distribution?.counts ? `
      <div class="card p-6">
        <h3 class="text-sm font-semibold text-slate-300 mb-4">Bypass Distribution (bypasses per halftime)</h3>
        <div style="height:260px"><canvas id="s2-dist-chart"></canvas></div>
      </div>` : ''}

      <!-- Correlation heatmap (top features) -->
      <div class="card p-6">
        <h3 class="text-sm font-semibold text-slate-300 mb-4">Correlation Matrix (top 15 features)</h3>
        <div id="s2-heatmap" class="overflow-x-auto text-xs"></div>
      </div>

      <!-- Descriptive stats table -->
      <div class="card p-6">
        <h3 class="text-sm font-semibold text-slate-300 mb-4">Descriptive Statistics</h3>
        <div id="s2-desc" class="overflow-x-auto"></div>
      </div>
    `;

    if (result.bypass_distribution?.counts) {
      const bd = result.bypass_distribution;
      const labels = bd.bin_edges.slice(0, -1).map(v => v.toFixed(2));
      const ctx = document.getElementById('s2-dist-chart').getContext('2d');
      charts.push(new Chart(ctx, {
        type: 'bar',
        data: {
          labels,
          datasets: [{ label: 'Count', data: bd.counts, backgroundColor: '#3b82f680', borderColor: '#3b82f6', borderWidth: 1 }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: '#94a3b8', maxTicksLimit: 10 }, grid: { color: '#1e293b' } },
            y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
          }
        }
      }));
    }

    const corrEl = document.getElementById('s2-heatmap');
    const corrMatrix = result.correlation_matrix || {};
    const allCols = Object.keys(corrMatrix).slice(0, 15);
    if (allCols.length) {
      const cellSize = 40;
      let table = `<div style="display:grid;grid-template-columns:120px ${allCols.map(() => `${cellSize}px`).join(' ')};gap:1px;font-size:10px">`;
      table += `<div></div>`;
      allCols.forEach(c => {
        table += `<div style="writing-mode:vertical-rl;transform:rotate(180deg);color:#94a3b8;height:90px;text-align:left;padding:4px 2px;overflow:hidden;text-overflow:ellipsis">${c}</div>`;
      });
      allCols.forEach(row => {
        table += `<div style="color:#cbd5e1;padding:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:120px">${row}</div>`;
        allCols.forEach(col => {
          const v = (corrMatrix[row] || {})[col] ?? 0;
          const abs = Math.abs(v);
          const bg = v > 0 ? `rgba(59,130,246,${abs.toFixed(2)})` : `rgba(239,68,68,${abs.toFixed(2)})`;
          table += `<div title="${row} × ${col}: ${v}" style="background:${bg};width:${cellSize}px;height:${cellSize}px;display:flex;align-items:center;justify-content:center;color:${abs>0.5?'#fff':'#94a3b8'}">${v.toFixed(1)}</div>`;
        });
      });
      table += `</div>`;
      corrEl.innerHTML = table;
    }

    const descEl = document.getElementById('s2-desc');
    const desc = result.descriptive_stats || {};
    const statKeys = Object.keys(desc);
    if (statKeys.length) {
      const statCols = Object.keys(desc[statKeys[0]] || {});
      let html = `<table class="w-full text-xs text-left"><thead class="table-header"><tr>
        <th class="px-3 py-2 text-slate-400">Stat</th>
        ${statCols.slice(0,12).map(c => `<th class="px-3 py-2 text-slate-400 font-mono">${c}</th>`).join('')}
      </tr></thead><tbody>`;
      statKeys.forEach((stat, i) => {
        html += `<tr class="${i%2===0 ? 'bg-slate-800/50' : ''}">
          <td class="px-3 py-1.5 text-slate-300 font-medium">${stat}</td>
          ${statCols.slice(0,12).map(c => `<td class="px-3 py-1.5 text-slate-400 font-mono">${(desc[stat][c] ?? '—').toString().slice(0,8)}</td>`).join('')}
        </tr>`;
      });
      html += `</tbody></table>`;
      descEl.innerHTML = html;
    }
  }

  return { mount, init, run };
})();
