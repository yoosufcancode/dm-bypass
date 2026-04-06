/**
 * Stage 1 — Data Ingestion
 * Lets user select one or more Wyscout leagues to process, or use existing CSVs.
 */
const stage1 = (() => {
  let container, config;

  const ALL_LEAGUES = ['Spain', 'England', 'France', 'Germany', 'Italy'];

  /**
   * @param {string} containerId
   * @param {object} cfg - shared app config (API base URL, callbacks)
   */
  function mount(containerId, cfg) {
    container = document.getElementById(containerId);
    config = cfg;
    render();
  }

  /** Reload available data when the pipeline resumes mid-session. */
  function onResume(result, pipelineState) {
    if (container) loadAvailableData();
  }

  /** Render the full stage 1 UI into the container. */
  function render() {
    container.innerHTML = `
      <div class="space-y-6">
        <div class="card p-6">
          <h2 class="text-lg font-semibold text-white mb-1">Stage 1 — Data Ingestion</h2>
          <p class="text-sm text-slate-400 mb-4">Compute midfielder features from the Wyscout Open Dataset (2017/18) for one or more leagues.</p>

          <!-- Existing data -->
          <div id="s1-existing" class="mb-6"></div>

          <!-- League selection form -->
          <div class="border-t border-slate-700 pt-4">
            <h3 class="text-sm font-semibold text-slate-300 mb-3">Select leagues to process</h3>
            <div class="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-4" id="s1-league-checks">
              ${ALL_LEAGUES.map(lg => `
                <label class="flex items-center gap-2 bg-slate-800 rounded-lg px-3 py-2 cursor-pointer hover:bg-slate-700">
                  <input type="checkbox" value="${lg}" class="s1-league-cb w-4 h-4 accent-blue-500" />
                  <span class="text-sm text-slate-300">${lg}</span>
                </label>`).join('')}
            </div>
            <div class="flex items-center gap-2 mb-4">
              <input id="s1-skip" type="checkbox" class="w-4 h-4 accent-blue-500" />
              <label class="mb-0">Skip feature engineering if CSV already exists</label>
            </div>
            <button class="btn-primary" onclick="stage1.startIngest()">Run Ingestion</button>
          </div>
        </div>

        <!-- Progress -->
        <div id="s1-progress" class="card p-6 hidden">
          <div class="flex justify-between text-sm mb-2">
            <span id="s1-msg" class="text-slate-300">Starting…</span>
            <span id="s1-pct" class="text-blue-400 font-mono">0%</span>
          </div>
          <div class="progress-bar"><div class="progress-fill" id="s1-bar" style="width:0%"></div></div>
        </div>

        <!-- Error -->
        <div id="s1-error" class="card p-4 border-red-800 bg-red-950 text-red-300 text-sm hidden"></div>
      </div>
    `;

    loadAvailableData();
  }

  /** Fetch already-processed CSVs from the API and render them. */
  async function loadAvailableData() {
    try {
      const r = await fetch(`${config.API}/stage1/available-data`);
      const data = await r.json();
      renderAvailableData(data);
    } catch { /* ignore */ }
  }

  /** Render the list of existing CSVs and pre-check their league checkboxes. */
  function renderAvailableData(rows) {
    const el = document.getElementById('s1-existing');
    if (!el || !rows.length) return;

    el.innerHTML = `
      <h3 class="text-sm font-semibold text-slate-300 mb-2">Existing processed data</h3>
      <div class="space-y-2">
        ${rows.map(d => `
          <div class="flex items-center justify-between bg-slate-800 rounded-lg px-4 py-2">
            <div>
              <span class="text-sm text-slate-300 font-semibold">${d.league}</span>
              <span class="text-xs text-slate-500 ml-2">${d.row_count?.toLocaleString()} rows</span>
            </div>
            <button class="btn-secondary text-sm" onclick="stage1.useExisting(${JSON.stringify(d)})">Use →</button>
          </div>`).join('')}
      </div>
      <div class="flex gap-2 mt-3">
        <button class="btn-secondary text-sm" onclick="stage1.useAllExisting()">Use all existing data →</button>
      </div>
    `;

    rows.forEach(d => {
      const cb = document.querySelector(`.s1-league-cb[value="${d.league}"]`);
      if (cb) cb.checked = true;
    });
  }

  /** Pass a single existing dataset row directly to the onDone callback. */
  function useExisting(d) {
    config.onDone({
      features_paths: { [d.league]: d.features_path },
      row_counts: { [d.league]: d.row_count },
      leagues_processed: [d.league],
    });
  }

  /** Fetch all existing CSVs and pass the combined result to onDone. */
  async function useAllExisting() {
    try {
      const r = await fetch(`${config.API}/stage1/available-data`);
      const data = await r.json();
      if (!data.length) { alert('No existing data found.'); return; }
      const features_paths = {};
      const row_counts = {};
      data.forEach(d => {
        features_paths[d.league] = d.features_path;
        row_counts[d.league] = d.row_count;
      });
      config.onDone({
        features_paths,
        row_counts,
        leagues_processed: data.map(d => d.league),
      });
    } catch (e) {
      alert(`Failed to load existing data: ${e}`);
    }
  }

  /** Read selected leagues, POST the ingest job, and begin polling. */
  async function startIngest() {
    const leagues = [...document.querySelectorAll('.s1-league-cb:checked')].map(cb => cb.value);
    const skip = document.getElementById('s1-skip').checked;

    if (!leagues.length) { alert('Select at least one league.'); return; }

    document.getElementById('s1-progress').classList.remove('hidden');
    document.getElementById('s1-error').classList.add('hidden');

    try {
      const r = await fetch(`${config.API}/stage1/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ leagues, skip_download: skip }),
      });
      const { job_id } = await r.json();

      config.pollUrl(
        `${config.API}/stage1/status/${job_id}`,
        (data) => {
          document.getElementById('s1-msg').textContent = data.message || '…';
          const pct = data.progress || 0;
          document.getElementById('s1-pct').textContent = `${pct}%`;
          document.getElementById('s1-bar').style.width = `${pct}%`;
        },
        (result) => {
          document.getElementById('s1-msg').textContent = 'Complete';
          document.getElementById('s1-pct').textContent = '100%';
          document.getElementById('s1-bar').style.width = '100%';
          config.onDone(result);
        },
        (err) => {
          document.getElementById('s1-progress').classList.add('hidden');
          const errEl = document.getElementById('s1-error');
          errEl.textContent = `Error: ${err}`;
          errEl.classList.remove('hidden');
        }
      );
    } catch (e) {
      const errEl = document.getElementById('s1-error');
      errEl.textContent = `Failed to start job: ${e}`;
      errEl.classList.remove('hidden');
    }
  }

  return { mount, onResume, startIngest, useExisting, useAllExisting };
})();
