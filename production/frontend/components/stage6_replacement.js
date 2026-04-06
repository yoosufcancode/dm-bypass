/**
 * Stage 6 — Replacement Finder
 *
 * User selects: league → team → top_n → min_matches
 * Pipeline runs scouting evaluation + model selection + tactical role clustering
 * and returns the full squad profile + ranked replacement candidates.
 */
const stage6 = (() => {
  let container, config;
  let teamsByLeague = {};

  const ALL_LEAGUES = ['Spain', 'England', 'France', 'Germany', 'Italy'];

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
   * No-op initialiser: Stage 6 is self-contained and does not depend on
   * stage 3 or stage 4 results — it reads directly from the processed CSVs.
   */
  function init() {}

  /** Render the league/team selection form and load available teams. */
  async function renderIdle() {
    container.innerHTML = `
      <div class="card p-6">
        <h2 class="text-lg font-semibold text-white mb-1">Stage 6 — Replacement Finder</h2>
        <p class="text-sm text-slate-400 mb-4">
          Team-specific model selection + tactical role clustering + cross-league scouting.
          Select a league and team — the pipeline analyzes the full squad automatically.
        </p>

        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
          <div>
            <label>League</label>
            <select id="s6-league" onchange="stage6.onLeagueChange()">
              <option value="">— select league —</option>
              ${ALL_LEAGUES.map(lg => `<option value="${lg}">${lg}</option>`).join('')}
            </select>
          </div>
          <div>
            <label>Team</label>
            <select id="s6-team">
              <option value="">— select league first —</option>
            </select>
          </div>
          <div>
            <label>Top N replacements per player</label>
            <input id="s6-topn" type="number" value="5" min="1" max="20" />
          </div>
          <div>
            <label>Min half-match rows per player <span class="text-slate-500">(10 ≈ 5 full matches)</span></label>
            <input id="s6-minmatches" type="number" value="10" min="3" max="50" />
          </div>
        </div>

        <button class="btn-primary" onclick="stage6.run()">Find Replacements</button>
      </div>

      <div id="s6-progress" class="card p-6 mt-4 hidden">
        <div class="flex justify-between text-sm mb-2">
          <span id="s6-msg" class="text-slate-300">Starting…</span>
          <span id="s6-pct" class="text-blue-400 font-mono">0%</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" id="s6-bar" style="width:0%"></div></div>
      </div>
      <div id="s6-error" class="card p-4 border-red-800 bg-red-950 text-red-300 text-sm mt-4 hidden"></div>
      <div id="s6-results" class="mt-6 space-y-6"></div>
    `;

    await loadTeams();
  }

  /** Fetch teams by league from the API and populate the local cache. */
  async function loadTeams() {
    try {
      const r = await fetch(`${config.API}/stage1/teams`);
      const data = await r.json();
      teamsByLeague = data.teams_by_league || {};
    } catch { /* ignore */ }
  }

  /** Repopulate the team dropdown when the selected league changes. */
  function onLeagueChange() {
    const league = document.getElementById('s6-league').value;
    const teamSel = document.getElementById('s6-team');
    const teams = teamsByLeague[league] || [];
    teamSel.innerHTML = teams.length
      ? teams.map(t => `<option value="${t}">${t}</option>`).join('')
      : `<option value="">— no data for ${league} —</option>`;
  }

  /** POST the replacement-finder job for the selected league and team. */
  async function run() {
    const league      = document.getElementById('s6-league').value;
    const team        = document.getElementById('s6-team').value;
    const top_n       = parseInt(document.getElementById('s6-topn').value) || 5;
    const min_matches = parseInt(document.getElementById('s6-minmatches').value) || 10;

    if (!league) { alert('Select a league.'); return; }
    if (!team)   { alert('Select a team.'); return; }

    document.getElementById('s6-progress').classList.remove('hidden');
    document.getElementById('s6-error').classList.add('hidden');
    document.getElementById('s6-results').innerHTML = '';

    try {
      const r = await fetch(`${config.API}/stage6/find-replacements`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ league, team, top_n, min_matches }),
      });
      const { job_id } = await r.json();

      config.pollUrl(
        `${config.API}/stage6/status/${job_id}`,
        (data) => {
          document.getElementById('s6-msg').textContent = data.message || '…';
          const pct = data.progress || 0;
          document.getElementById('s6-pct').textContent = `${pct}%`;
          document.getElementById('s6-bar').style.width = `${pct}%`;
        },
        (result) => {
          document.getElementById('s6-msg').textContent = 'Complete';
          document.getElementById('s6-pct').textContent = '100%';
          document.getElementById('s6-bar').style.width = '100%';
          renderResults(result);
          config.onDone(result);
        },
        (err) => {
          document.getElementById('s6-progress').classList.add('hidden');
          const errEl = document.getElementById('s6-error');
          errEl.textContent = `Error: ${err}`;
          errEl.classList.remove('hidden');
        }
      );
    } catch (e) {
      document.getElementById('s6-error').textContent = `Failed: ${e}`;
      document.getElementById('s6-error').classList.remove('hidden');
    }
  }

  /** Return a colour-coded badge element for a player's tactical role. */
  function roleBadge(role) {
    const colors = {
      'Anchor 6':       'badge-blue',
      'Ball-winning 8': 'badge-yellow',
      'Hybrid 6/8':     'badge-green',
    };
    return `<span class="badge ${colors[role] || 'badge-blue'}">${role || 'Unknown'}</span>`;
  }

  /**
   * Render a colour-graded progress bar for a bypass score percentile.
   * Green < 33 (low bypass rate), amber 33–65, red ≥ 66 (high bypass rate).
   */
  function scorebar(v) {
    const pct = Math.round(Math.max(0, Math.min(100, v)));
    const color = pct < 33 ? '#10b981' : pct < 66 ? '#f59e0b' : '#ef4444';
    return `
      <div class="flex items-center gap-2">
        <div class="flex-1 progress-bar">
          <div class="progress-fill" style="width:${pct}%;background:${color}"></div>
        </div>
        <span class="text-xs font-mono w-8 text-right" style="color:${color}">${pct}</span>
      </div>`;
  }

  /** Render the full results view: header, scouting features, squad table, and recommendations. */
  function renderResults(result) {
    const el = document.getElementById('s6-results');

    const modelHtml = `
      <div class="card p-4 flex flex-wrap items-center gap-4">
        <div>
          <span class="text-slate-400 text-sm">Team:</span>
          <span class="text-white font-semibold ml-1">${result.team}</span>
          <span class="text-slate-500 text-sm ml-1">(${result.league})</span>
        </div>
        <div>
          <span class="text-slate-400 text-sm">Model selected:</span>
          <span class="badge badge-green ml-1">${result.model_selected}</span>
        </div>
        <div>
          <span class="text-slate-400 text-sm">Spearman ρ test:</span>
          <span class="font-mono font-semibold ml-1 ${result.spearman_test >= 0.3 ? 'text-green-400' : 'text-yellow-400'}">${result.spearman_test?.toFixed(4)}</span>
        </div>
        <div>
          <span class="text-slate-400 text-sm">train:</span>
          <span class="font-mono text-slate-300 ml-1">${result.spearman_train?.toFixed(4)}</span>
        </div>
      </div>`;

    const feats = result.scouting_features || [];
    const featHtml = `
      <div class="card p-6">
        <h3 class="text-sm font-semibold text-slate-300 mb-3">Scouting Features (${feats.length})</h3>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead class="table-header">
              <tr>
                <th class="px-3 py-2 text-left text-slate-400">Feature</th>
                <th class="px-3 py-2 text-center text-slate-400">Gradient</th>
                <th class="px-3 py-2 text-center text-slate-400">Direction</th>
                <th class="px-3 py-2 text-center text-slate-400">p-value</th>
                <th class="px-3 py-2 text-center text-slate-400">Sign stable</th>
                <th class="px-3 py-2 text-left text-slate-400">Confidence</th>
              </tr>
            </thead>
            <tbody>
              ${feats.map((f, i) => `
                <tr class="${i % 2 === 0 ? 'bg-slate-800/50' : ''}">
                  <td class="px-3 py-2 font-mono text-slate-300">${f.feature}</td>
                  <td class="px-3 py-2 text-center font-mono ${f.gradient > 0 ? 'text-red-400' : 'text-green-400'}">${f.gradient?.toFixed(4)}</td>
                  <td class="px-3 py-2 text-center text-xs text-slate-400">${f.direction}</td>
                  <td class="px-3 py-2 text-center font-mono ${f.p_value < 0.05 ? 'text-green-400' : f.p_value < 0.15 ? 'text-yellow-400' : 'text-red-400'}">${f.p_value?.toFixed(4)}</td>
                  <td class="px-3 py-2 text-center">${f.sign_stable ? '<span class="text-green-400">✓</span>' : '<span class="text-yellow-400">~</span>'}</td>
                  <td class="px-3 py-2 text-xs text-slate-500">${f.confidence_tier}</td>
                </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>`;

    const squad = result.squad || [];
    const squadHtml = `
      <div class="card p-6">
        <h3 class="text-sm font-semibold text-slate-300 mb-3">${result.team} — Squad Bypass Scores</h3>
        <p class="text-xs text-slate-500 mb-3">Score = league percentile rank. Lower = better bypass prevention.</p>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead class="table-header">
              <tr>
                <th class="px-3 py-2 text-left text-slate-400">Player</th>
                <th class="px-3 py-2 text-left text-slate-400">Role</th>
                <th class="px-3 py-2 text-center text-slate-400">Pos</th>
                <th class="px-3 py-2 text-center text-slate-400">AvgX</th>
                <th class="px-3 py-2 text-center text-slate-400" style="min-width:140px">Bypass Score</th>
                <th class="px-3 py-2 text-center text-slate-400">Halves</th>
                <th class="px-3 py-2 text-center text-slate-400">Bypasses/half</th>
              </tr>
            </thead>
            <tbody>
              ${squad.map((p, i) => `
                <tr class="${i % 2 === 0 ? 'bg-slate-800/50' : ''}">
                  <td class="px-3 py-2 font-semibold text-white">${p.player_name}</td>
                  <td class="px-3 py-2">${roleBadge(p.tactical_role)}</td>
                  <td class="px-3 py-2 text-center text-slate-400">${p.position_bucket}</td>
                  <td class="px-3 py-2 text-center font-mono text-slate-400">${p.average_position_x?.toFixed(1) ?? '—'}</td>
                  <td class="px-3 py-2">${scorebar(p.bypass_score)}</td>
                  <td class="px-3 py-2 text-center text-slate-400">${p.halves_played}</td>
                  <td class="px-3 py-2 text-center font-mono ${p.bypasses_per_half > 6 ? 'text-red-400' : 'text-slate-300'}">${p.bypasses_per_half?.toFixed(2)}</td>
                </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>`;

    const recs = result.recommendations || [];
    const recsHtml = recs.map(rec => {
      const p = rec.target_player;
      const replacements = rec.replacements || [];

      return `
        <div class="card p-6">
          <div class="flex flex-wrap items-center gap-3 mb-4">
            <span class="text-base font-semibold text-white">${p.player_name}</span>
            ${roleBadge(p.tactical_role)}
            <span class="badge badge-blue">${p.position_bucket}</span>
            <span class="text-slate-400 text-sm">AvgX ${p.average_position_x?.toFixed(1) ?? '—'}</span>
            <div class="ml-auto flex gap-4 text-right">
              <div>
                <div class="text-xl font-bold ${p.bypass_score > 66 ? 'text-red-400' : p.bypass_score > 33 ? 'text-yellow-400' : 'text-green-400'}">${p.bypass_score?.toFixed(1)}</div>
                <div class="text-xs text-slate-500">bypass score</div>
              </div>
              <div>
                <div class="text-xl font-bold text-orange-400">${p.bypasses_per_half?.toFixed(2)}</div>
                <div class="text-xs text-slate-500">bypasses/half</div>
              </div>
            </div>
          </div>

          <div class="text-xs text-slate-500 mb-3">Match filter: <span class="text-slate-400">${rec.match_filter}</span></div>

          ${replacements.length === 0
            ? `<p class="text-slate-500 text-sm">No candidates found with lower bypass score in this role + position.</p>`
            : `<div class="space-y-3">
                ${replacements.map(c => {
                  const fc = c.feature_comparison || {};
                  const fcKeys = Object.keys(fc);
                  return `
                    <div class="bg-slate-800 rounded-lg p-4">
                      <div class="flex flex-wrap items-start justify-between gap-2 mb-2">
                        <div>
                          <div class="flex items-center gap-2 flex-wrap">
                            <span class="text-slate-400 text-xs font-mono">#${c.rank}</span>
                            <span class="font-semibold text-white">${c.player_name}</span>
                            <span class="badge badge-blue text-xs">${c.team}</span>
                            <span class="text-xs text-slate-500">${c.league}</span>
                            ${roleBadge(c.tactical_role)}
                            <span class="badge badge-blue">${c.position_bucket}</span>
                          </div>
                          <div class="text-xs text-slate-500 mt-1">AvgX ${c.average_position_x?.toFixed(1) ?? '—'}</div>
                        </div>
                        <div class="flex gap-4 text-right">
                          <div>
                            <div class="text-lg font-bold text-green-400">${c.bypass_score?.toFixed(1)}</div>
                            <div class="text-xs text-slate-500">score</div>
                          </div>
                          <div>
                            <div class="text-lg font-bold text-blue-400">+${c.improvement?.toFixed(3)}</div>
                            <div class="text-xs text-slate-500">improvement</div>
                          </div>
                          <div>
                            <div class="text-lg font-bold text-orange-400">${c.bypasses_per_half?.toFixed(2)}</div>
                            <div class="text-xs text-slate-500">bypasses/half</div>
                          </div>
                        </div>
                      </div>
                      ${fcKeys.length > 0 ? `
                        <details class="mt-2">
                          <summary class="text-xs text-slate-500 cursor-pointer hover:text-slate-300">Feature comparison vs ${p.player_name}</summary>
                          <table class="w-full text-xs mt-2">
                            <thead><tr>
                              <th class="text-left text-slate-500 pb-1">Feature</th>
                              <th class="text-right text-slate-500 pb-1">Candidate</th>
                              <th class="text-right text-slate-500 pb-1">Target</th>
                              <th class="text-right text-slate-500 pb-1">Δ</th>
                              <th class="text-left text-slate-500 pb-1 pl-2">Direction</th>
                            </tr></thead>
                            <tbody>
                              ${fcKeys.map(f => {
                                const v = fc[f];
                                return `<tr>
                                  <td class="font-mono text-slate-400 py-0.5">${f}</td>
                                  <td class="text-right font-mono text-slate-300">${v.candidate?.toFixed(3)}</td>
                                  <td class="text-right font-mono text-slate-500">${v.target?.toFixed(3)}</td>
                                  <td class="text-right font-mono ${(v.delta ?? 0) > 0 ? 'text-green-400' : 'text-red-400'}">${(v.delta ?? 0) > 0 ? '+' : ''}${v.delta?.toFixed(3)}</td>
                                  <td class="pl-2 text-slate-500">${v.direction || ''}</td>
                                </tr>`;
                              }).join('')}
                            </tbody>
                          </table>
                        </details>` : ''}
                    </div>`;
                }).join('')}
              </div>`}
        </div>`;
    }).join('');

    el.innerHTML = modelHtml + featHtml + squadHtml + recsHtml;
  }

  return { mount, init, run, onLeagueChange };
})();
