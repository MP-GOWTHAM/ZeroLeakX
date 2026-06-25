/* ============================================================================
 * ZeroLeakX — frontend client (v3)
 *
 * Captures real keystroke dynamics + mouse dynamics in the browser, sends
 * per-keystroke TIMING arrays (no key content) to the FastAPI risk engine, and
 * renders the live Dynamic Trust Score, 10-feature SHAP, real model outputs and
 * the step-up flow. Multi-user login with persistent baselines, paste capture,
 * and a live SOC console over WebSocket.
 * ==========================================================================*/

const API = '';
let SID = null, deviceFP = null, lastState = null, currentUser = null;

// Behavioural buffers
const keyDown = {};
let keyLog = [], lastSentIdx = 0;
const pointer = { last: null, speeds: [], accels: [], lastSpeed: null };

// Flow flags
let attackActive = false, simTimer = null, otpOpen = false, stepupShown = false;
let otpContext = 'attack', currentOtpCode = '', blocked = false, blockedCount = 0;
let socWS = null;

// Channel: phones/tablets emit touch; PCs emit a mouse. Detected once.
const IS_TOUCH = ('ontouchstart' in window) || (navigator.maxTouchPoints > 0);
const CHANNEL = IS_TOUCH ? 'mobile' : 'web';

const FEATS = ['mean_dwell', 'std_dwell', 'mean_flight', 'std_flight', 'typing_speed',
  'backspace_rate', 'dwell_cv', 'flight_cv', 'flight_p90', 'rhythm_entropy'];
const LABELS = {
  mean_dwell: 'Mean dwell', std_dwell: 'Dwell var', mean_flight: 'Mean flight',
  std_flight: 'Flight var', typing_speed: 'Typing speed', backspace_rate: 'Backspace',
  dwell_cv: 'Dwell CV', flight_cv: 'Flight CV', flight_p90: 'Long pauses', rhythm_entropy: 'Rhythm entropy'
};

/* ── helpers ───────────────────────────────────────────────────────────────*/
const $ = id => document.getElementById(id);
const mean = a => a.reduce((x, y) => x + y, 0) / (a.length || 1);
const std = a => { const m = mean(a); return Math.sqrt(mean(a.map(x => (x - m) ** 2))); };
const col = v => v > 70 ? 'var(--green)' : v > 40 ? 'var(--amber)' : 'var(--red)';
const r2 = x => Math.round(x * 10) / 10;
const lerp = (a, b, p) => a + (b - a) * p;
function gauss(m, s) { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return m + s * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

async function api(path, body) {
  const r = await fetch(API + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  if (!r.ok) throw new Error(path + ' -> ' + r.status);
  return r.json();
}

function computeFP() {
  let ch = '';
  try { const c = document.createElement('canvas'); const x = c.getContext('2d'); x.textBaseline = 'top'; x.font = "14px 'Arial'"; x.fillStyle = '#069'; x.fillText('ZeroLeakX◆bob', 2, 2); ch = c.toDataURL(); } catch (e) {}
  const raw = [navigator.userAgent, navigator.language, screen.width + 'x' + screen.height + 'x' + screen.colorDepth, new Date().getTimezoneOffset(), navigator.hardwareConcurrency || 0, ch].join('|');
  let h = 2166136261 >>> 0;
  for (let i = 0; i < raw.length; i++) { h ^= raw.charCodeAt(i); h = Math.imul(h, 16777619); }
  return 'fp_' + (h >>> 0).toString(16);
}

/* ── keystroke capture → timing arrays ─────────────────────────────────────*/
// Accept soft-keyboard keys too ('Unidentified'/'Process') so taps on a phone
// still yield dwell/flight timing.
const trackable = k => k.length === 1 || k === 'Backspace' || k === 'Unidentified' || k === 'Process';
document.addEventListener('keydown', e => { const k = e.key; if (trackable(k) && keyDown[k] == null) keyDown[k] = performance.now(); });
document.addEventListener('keyup', e => {
  const k = e.key;
  if (keyDown[k] != null) {
    keyLog.push({ char: k, down: keyDown[k], up: performance.now(), dwell: performance.now() - keyDown[k] });
    keyDown[k] = null;
    if (keyLog.length > 500) keyLog = keyLog.slice(-250);
    if ($('panel-calibrate').classList.contains('active')) renderLiveFeatures();
  }
});

function extractWindow(slice) {
  const s = [...slice].sort((a, b) => a.down - b.down);
  const dwell = s.map(x => r2(x.dwell));
  const flight = [];
  for (let i = 1; i < s.length; i++) flight.push(r2(s[i].down - s[i - 1].up));
  const back = s.filter(x => x.char === 'Backspace').length / (s.length || 1);
  return { dwell, flight, backspace_rate: +back.toFixed(3), key_count: s.length };
}

/* ── mouse dynamics → pointer signal ───────────────────────────────────────*/
document.addEventListener('mousemove', e => {
  const now = performance.now();
  if (pointer.last) {
    const dt = now - pointer.last.t;
    if (dt > 0 && dt < 400) {
      const sp = Math.hypot(e.clientX - pointer.last.x, e.clientY - pointer.last.y) / dt;
      pointer.speeds.push(sp);
      if (pointer.lastSpeed != null) pointer.accels.push(Math.abs(sp - pointer.lastSpeed) / dt);
      pointer.lastSpeed = sp;
      if (pointer.speeds.length > 240) pointer.speeds.shift();
      if (pointer.accels.length > 240) pointer.accels.shift();
    }
  }
  pointer.last = { x: e.clientX, y: e.clientY, t: now };
});

// Touch / swipe dynamics (mobile) — feeds the SAME pointer buffers, so the
// "swipe" signal is real on phones (where mousemove never fires).
let touchLast = null;
document.addEventListener('touchmove', e => {
  const t = (e.touches && e.touches[0]) || (e.changedTouches && e.changedTouches[0]); if (!t) return;
  const now = performance.now();
  if (touchLast) {
    const dt = now - touchLast.t;
    if (dt > 0 && dt < 400) {
      const sp = Math.hypot(t.clientX - touchLast.x, t.clientY - touchLast.y) / dt;
      pointer.speeds.push(sp);
      if (pointer.lastSpeed != null) pointer.accels.push(Math.abs(sp - pointer.lastSpeed) / dt);
      pointer.lastSpeed = sp;
      if (pointer.speeds.length > 240) pointer.speeds.shift();
      if (pointer.accels.length > 240) pointer.accels.shift();
    }
  }
  touchLast = { x: t.clientX, y: t.clientY, t: now };
}, { passive: true });
document.addEventListener('touchend', () => { touchLast = null; }, { passive: true });

function pointerStats() {
  if (pointer.speeds.length < 6) return null;
  const sp = pointer.speeds.slice(-80), ac = pointer.accels.slice(-80);
  return [r2(mean(sp) * 100), r2(std(sp) * 100), r2(std(ac) * 1000)];
}

/* ── login / multi-user ────────────────────────────────────────────────────*/
async function loadUsers() {
  try {
    const r = await (await fetch('/api/users')).json();
    $('login-users').innerHTML = r.users.length
      ? r.users.map(u => `<button class="btn btn-ghost btn-sm" onclick="quickLogin('${u.username}')">${u.username}${u.enrolled ? ' ✓' : ''}</button>`).join('')
      : '<span style="font-size:11px;color:var(--muted)">No users yet — type a name above</span>';
  } catch (e) {}
}
function quickLogin(u) { $('login-username').value = u; doLogin(); }

async function doLogin() {
  const u = ($('login-username').value || 'guest').trim() || 'guest';
  deviceFP = computeFP();
  try {
    const st = await api('/api/login', { username: u });
    SID = st.session_id; currentUser = st.user;
    $('login-overlay').classList.remove('show');
    setConn(true);
    applyUser(st);
    render(st);
    await score({ device_fp: deviceFP });
    loadHealth();
  } catch (e) { setConn(false); }
}

function applyUser(st) {
  $('user-name').textContent = st.user;
  $('user-avatar').textContent = (st.user || 'GU').slice(0, 2).toUpperCase();
  $('user-geo').textContent = '📍 ' + (st.geo_city || '—');
  const b = $('calib-badge');
  if (st.enrolled) { b.textContent = 'Personal model active'; b.style.cssText = 'background:var(--green-dim);color:var(--green);border:1px solid rgba(0,201,122,.25)'; }
  else { b.textContent = 'Not enrolled'; b.style.cssText = 'background:var(--amber-dim);color:var(--amber);border:1px solid rgba(255,176,32,.25)'; }
}

function logout() {
  SID = null; currentUser = null; lastState = null;
  if (socWS) { socWS.close(); socWS = null; }
  keyLog = []; lastSentIdx = 0; calibWindows = []; calibLines = 0;
  $('login-overlay').classList.add('show');
  $('login-username').value = '';
  loadUsers();
}

async function loadHealth() {
  try {
    const h = await (await fetch('/api/health')).json();
    const el = $('calib-metric'); if (!el) return;
    const m = h.metrics || {};
    if (h.data_backed && m.auc_mean != null) {
      el.innerHTML = `🎓 <strong style="color:var(--text)">Validated on real data:</strong> CMU Keystroke Dynamics ` +
        `(${m.subjects} subjects, ${m.features}-feature model). Impostor detection AUC ` +
        `<strong style="color:var(--teal)">${m.auc_mean}</strong>, EER <strong style="color:var(--teal)">${m.eer_mean}</strong> ` +
        `on unseen identities. Models live: XGBoost ${h.xgboost ? '✓' : '✗'} · LSTM ${h.lstm ? '✓' : '✗'} · real impostors ${h.impostor_pool_size.toLocaleString()}.`;
    }
  } catch (e) {}
}

let scoring = false;
async function score(extra) {
  if (!SID || scoring) return null;
  scoring = true;
  try {
    const st = await api(`/api/session/${SID}/score`, Object.assign({ device_fp: deviceFP, channel: CHANNEL }, extra || {}));
    render(st); handleAction(st); setConn(true); return st;
  } catch (e) { setConn(false); return null; } finally { scoring = false; }
}

function handleAction(st) {
  blocked = st.blocked;
  if (st.blocked) { showBlocked(st); return; }
  if ((st.action === 'stepup' || st.action === 'block') && !otpOpen && !stepupShown) { stepupShown = true; triggerStepup(st.action === 'block', 'attack'); }
  if (st.action === 'allow') stepupShown = false;
}

/* ── continuous monitoring loop ────────────────────────────────────────────*/
setInterval(() => {
  if (!SID || attackActive || otpOpen || blocked) return;
  const fresh = keyLog.length - lastSentIdx;
  const extra = {};
  if (fresh >= 6) {
    extra.window = extractWindow(keyLog.slice(-16));
    const ps = pointerStats(); if (ps) extra.pointer = ps;
    lastSentIdx = keyLog.length;
  }
  score(extra);
}, 1400);

/* ── calibration ───────────────────────────────────────────────────────────*/
const SENTENCES = [
  'the quick brown fox jumps over the lazy dog near the river bank',
  'banking security depends on continuous behavioural validation',
  'my account password alone can no longer prove who i really am',
  'zeroleakx watches how i type not just what i type today',
  'transfer five thousand rupees to the savings account tomorrow',
  'every keystroke has a rhythm that is uniquely my own signature',
];
let calibIdx = 0, calibStartIdx = 0, calibWindows = [], calibLines = 0;

function calibBegin() { calibStartIdx = keyLog.length; $('calib-sentence').textContent = SENTENCES[calibIdx % SENTENCES.length]; $('calib-input').value = ''; $('calib-input').focus(); }
function calibCommit() {
  const slice = keyLog.slice(calibStartIdx);
  if (slice.length >= 8) {
    calibWindows.push(extractWindow(slice)); calibLines++;
    $('calib-count').textContent = Math.min(calibLines, 6);
    $('calib-windows').textContent = calibWindows.length;
    $('calib-fill').style.width = Math.min(100, calibLines / 4 * 100) + '%';
    if (calibLines >= 3) $('calib-enroll-btn').disabled = false;
  }
  calibIdx++; calibBegin();
}
function calibSkip() { calibIdx++; calibBegin(); }

async function enrollBaseline() {
  if (calibWindows.length < 3) return;
  const ps = pointerStats();
  const st = await api(`/api/session/${SID}/enroll`, { windows: calibWindows, pointer: ps ? [ps] : null, device_fp: deviceFP });
  render(st);
  if (st.enroll && st.enroll.enrolled) applyUser(st);
}

function renderLiveFeatures() {
  if (keyLog.length < 4) return;
  const w = extractWindow(keyLog.slice(-16));
  const md = r2(mean(w.dwell)), mf = r2(mean(w.flight.length ? w.flight : [0]));
  $('calib-features').innerHTML = `
    <div>keys captured <span style="color:var(--blue)">${w.dwell.length}</span></div>
    <div>mean dwell&nbsp;&nbsp; <span style="color:var(--blue)">${md} ms</span></div>
    <div>mean flight&nbsp; <span style="color:var(--teal)">${mf} ms</span></div>
    <div>dwell array&nbsp; <span style="color:var(--muted2);font-size:10px">[${w.dwell.slice(0, 8).join(', ')}…]</span></div>
    <div>backspace&nbsp;&nbsp;&nbsp; <span style="color:var(--purple)">${(w.backspace_rate * 100).toFixed(1)} %</span></div>`;
}

/* ── scenario simulator (synthetic timing → real model) ────────────────────*/
function synthWindow(p) {
  const md = lerp(95, 188, p), sdd = lerp(8, 55, p), mf = lerp(120, 255, p), sdf = lerp(14, 120, p), back = lerp(0.04, 0.26, p);
  const n = 14, dwell = [], flight = [];
  for (let i = 0; i < n; i++) dwell.push(r2(Math.max(20, gauss(md, sdd))));
  for (let i = 0; i < n - 1; i++) flight.push(r2(Math.max(0, gauss(mf, sdf))));
  return { dwell, flight, backspace_rate: +back.toFixed(3), key_count: n };
}
function banner(on) { $('attacker-banner').classList.toggle('show', on); }

async function simReset() {
  attackActive = false; if (simTimer) clearInterval(simTimer); banner(false); stepupShown = false;
  if (!SID) return; render(await api(`/api/session/${SID}/reset`));
}

async function simAttack() {
  if (attackActive) return; await simReset();
  attackActive = true; banner(true); stepupShown = false;
  setShap('Behavioural drift initiated — model scoring synthetic attacker input…');
  let step = 0;
  simTimer = setInterval(async () => {
    step++; const p = Math.min(1, step / 8);
    const st = await api(`/api/session/${SID}/score`, { window: synthWindow(p), device_ok: false, geo_km: 380, swipe: Math.max(6, 88 - p * 100), nav_score: Math.max(40, 95 - p * 70), label: 'sim' });
    render(st);
    if (step >= 8) { clearInterval(simTimer); attackActive = false; handleAction(st); }
  }, 850);
}

async function simBehaviour() {
  if (attackActive) return; await simReset();
  attackActive = true; banner(true); stepupShown = false;
  setShap('Same device, different typist — pure behavioural-biometric drift…');
  let step = 0;
  simTimer = setInterval(async () => {
    step++; const p = Math.min(1, step / 7);
    const st = await api(`/api/session/${SID}/score`, { window: synthWindow(p), swipe: Math.max(12, 88 - p * 80), nav_score: Math.max(44, 95 - p * 55), label: 'sim' });
    render(st);
    if (step >= 7) { clearInterval(simTimer); attackActive = false; handleAction(st); }
  }, 850);
}

async function simGeo() { await simReset(); const st = await score({ geo_km: 380, label: 'sim' }); if (st) setShap('Geo context: 380 km shift (weight 10%). Behaviour & device nominal. Risk: moderate — monitoring tightened, no block.'); }

async function simNewDevice() {
  await simReset(); stepupShown = true;
  const st = await score({ device_ok: false, label: 'sim' });
  if (st) setShap('Device fingerprint: unknown device (weight 18%). Behaviour consistent. Policy: step-up required for new device.');
  setTimeout(() => triggerStepup(false, 'new-device'), 900);
}

/* ── step-up OTP ───────────────────────────────────────────────────────────*/
async function triggerStepup(final, ctx) {
  if (otpOpen) return; otpOpen = true; otpContext = ctx || 'attack';
  ['o1', 'o2', 'o3', 'o4', 'o5', 'o6'].forEach(id => $(id).value = '');
  try {
    const r = await api(`/api/session/${SID}/stepup/request`);
    currentOtpCode = r.demo_code;
    $('otp-hint').innerHTML = `Demo OTP (would arrive by SMS): <strong style="color:var(--text)">${r.demo_code}</strong> → legitimate user, session resumes<br>Any other code → attacker confirmed, account locked`;
  } catch (e) { currentOtpCode = ''; }
  $('modal-title').textContent = final ? 'Final verification required' : 'Identity verification required';
  $('modal-desc').textContent = otpContext === 'new-device' ? 'Login from an unrecognised device. Enter the OTP sent to •••• 7291 to add this device.' : 'ZeroLeakX detected anomalous behaviour. Enter the OTP sent to •••• 7291 to continue.';
  $('otp-overlay').classList.add('show');
  setTimeout(() => $('o1').focus(), 120);
}
function otpNext(el, n) { if (el.value.length === 1) $(n).focus(); }
function getOtp() { return ['o1', 'o2', 'o3', 'o4', 'o5', 'o6'].map(id => $(id).value).join(''); }
function checkOtp() { if (getOtp().length === 6) verifyOtp(); }
async function verifyOtp() {
  const code = getOtp(); $('otp-overlay').classList.remove('show'); otpOpen = false; stepupShown = false;
  banner(false);
  const st = await api(`/api/session/${SID}/stepup/verify`, { code });
  render(st); if (!st.verified) handleAction(st);
}
function cancelSession() { $('otp-overlay').classList.remove('show'); otpOpen = false; stepupShown = false; }

function showBlocked(st) {
  blocked = true; blockedCount++; $('sessions-blocked').textContent = blockedCount;
  if (st.incident) $('incident-id').textContent = 'Incident #' + st.incident;
  setTimeout(() => $('blocked-overlay').classList.add('show'), 350);
}
async function resetAll() {
  $('blocked-overlay').classList.remove('show'); blocked = false; otpOpen = false; stepupShown = false;
  render(await api(`/api/session/${SID}/reset`));
}

/* ── transfer ──────────────────────────────────────────────────────────────*/
async function doTransfer() {
  if (lastState && (lastState.action === 'stepup' || lastState.action === 'block' || lastState.dts < 70)) { triggerStepup(lastState.dts < 25, 'transfer'); return; }
  const amt = $('txn-amt').value, bene = $('bene-name').value || 'Beneficiary';
  if (!amt || isNaN(amt) || +amt <= 0) return;
  $('balance-display').textContent = '₹' + (482150 - parseFloat(amt)).toLocaleString('en-IN') + '.00';
  ['bene-name', 'bene-acct', 'bene-ifsc', 'txn-amt', 'txn-remarks'].forEach(id => $(id).value = '');
  await score({ txn: true });
}

/* ── rendering ─────────────────────────────────────────────────────────────*/
const SIGS = ['key', 'swipe', 'nav', 'device', 'geo', 'txn'];
function setText(id, v) { const e = $(id); if (e) e.textContent = v; }

function render(st) {
  if (!st) return; lastState = st;
  const dts = st.dts, c = col(dts);
  setText('dts-mini-num', dts); $('dts-mini-num').style.color = c;
  $('dts-mini-fill').style.width = dts + '%'; $('dts-mini-fill').style.background = c;
  const msx = $('dts-mini-status'); msx.style.color = c; msx.textContent = dts > 70 ? '● Session trusted' : dts > 40 ? '◐ Moderate risk' : '⚠ High risk';
  const fab = $('fab-dts'); if (fab) { fab.textContent = dts; fab.style.color = dts > 70 ? '#bfffe0' : dts > 40 ? '#ffe6b0' : '#ffd0d0'; }

  ['dts-big', 'dts-xfer'].forEach(id => { const e = $(id); if (e) { e.textContent = dts; e.style.color = c; } });
  ['dts-fill-big', 'dts-xfer-bar'].forEach(id => { const e = $(id); if (e) { e.style.width = dts + '%'; e.style.background = c; } });
  if ($('risk-needle')) $('risk-needle').style.left = dts + '%';

  const verdict = $('dts-verdict'), vmap = { low: ['Low risk', 'green'], moderate: ['Moderate risk', 'amber'], high: ['High risk', 'red'], critical: ['Critical risk', 'red'] };
  const [vt, vc] = vmap[st.risk] || vmap.low;
  verdict.textContent = vt; verdict.style.background = `var(--${vc}-dim)`; verdict.style.color = `var(--${vc})`; verdict.style.border = `1px solid var(--${vc})`;

  const pill = $('trust-pill'), txt = $('trust-text');
  if (dts > 70) { pill.className = 'trust-pill trust-high'; txt.textContent = 'Trusted'; pill.querySelector('.trust-dot').style.background = 'var(--green)'; }
  else if (dts > 40) { pill.className = 'trust-pill trust-med'; txt.textContent = 'Moderate risk'; pill.querySelector('.trust-dot').style.background = 'var(--amber)'; }
  else { pill.className = 'trust-pill trust-low'; txt.textContent = 'High risk'; pill.querySelector('.trust-dot').style.background = 'var(--red)'; }

  SIGS.forEach(k => {
    const v = st.signals[k], cc = col(v);
    [['ss', 'sn'], ['ms', 'mn']].forEach(([fp, np]) => { const f = $(fp + '-' + k), n = $(np + '-' + k); if (f) { f.style.width = v + '%'; f.style.background = cc; } if (n) n.textContent = v; });
  });
  ['key', 'device', 'geo'].forEach(k => { const v = st.signals[k], cc = col(v); const f = $('xs-' + k), n = $('xn-' + k); if (f) { f.style.width = v + '%'; f.style.background = cc; } if (n) n.textContent = v; });

  const m = st.models;
  setText('mo-lstm', m.lstm_real != null ? `${m.lstm} (${m.lstm_real})` : m.lstm);
  $('mo-lstm').style.color = m.lstm === 'Normal' ? 'var(--green)' : m.lstm.includes('Drift') ? 'var(--amber)' : 'var(--red)';
  setText('mo-iso', 'Score ' + Number(m.iso_score).toFixed(2));
  setText('mo-xgb', m.xgb_dts + ' / 100');
  setText('mo-ato', Number(m.ato_prob).toFixed(1) + '%');
  ['mo-iso', 'mo-xgb', 'mo-ato'].forEach(id => $(id).style.color = c);

  renderShap(st.contrib || {}, st.risk, m);
  setText('dts-avg', st.avg_dts);
  setText('model-mode', st.enrolled ? 'Personal' : 'Pop.');
  setText('model-mode-sub', st.enrolled ? `personal model · ${st.n_samples} samples` : 'cold-start baseline');

  updateChart(st.history, dts);
  renderTransferAdvice(st);
  renderEvents(st.events);
}

function renderShap(contrib, risk, m) {
  const grid = $('shap-grid'); if (!grid) return;
  let top = null, topv = 0;
  grid.innerHTML = FEATS.map(f => {
    const pct = contrib[f] != null ? contrib[f] : 0;
    if (pct > topv) { topv = pct; top = f; }
    const cc = pct > 28 ? 'var(--red)' : pct > 14 ? 'var(--amber)' : 'var(--teal)';
    return `<div class="shap-row"><span class="shap-label">${LABELS[f]}</span><div class="shap-track"><div class="shap-fill" style="width:${Math.min(100, pct * 1.6)}%;background:${cc}"></div></div><span class="shap-pct">${pct.toFixed(0)}%</span></div>`;
  }).join('');
  if (risk !== 'low' && top) setShap(`Top anomaly driver: ${LABELS[top]} (${topv.toFixed(0)}%). LSTM: ${m.lstm}. Isolation Forest: ${Number(m.iso_score).toFixed(2)}. XGBoost ATO: ${Number(m.ato_prob).toFixed(1)}%.`);
  else if (risk === 'low') setShap('All signals nominal — no anomaly contribution above threshold.');
}

function renderTransferAdvice(st) {
  const adv = $('xfer-advice'), badge = $('transfer-badge'), notice = $('risk-notice'); if (!adv) return;
  if (st.dts > 70) { adv.textContent = 'Session trust is high. Transfer can proceed without additional verification.'; badge.textContent = 'Secure session'; badge.style.cssText = 'background:var(--green-dim);color:var(--green);border:1px solid rgba(0,201,122,.25)'; notice.classList.remove('show'); }
  else if (st.dts > 40) { adv.textContent = 'Moderate risk detected. Transfer is allowed but flagged; step-up may be requested.'; badge.textContent = '⚠ Moderate risk'; badge.style.cssText = 'background:var(--amber-dim);color:var(--amber);border:1px solid rgba(255,176,32,.25)'; notice.classList.remove('show'); }
  else { adv.textContent = 'High-risk session. Transfer blocked — identity verification required.'; badge.textContent = '⛔ Blocked'; badge.style.cssText = 'background:var(--red-dim);color:var(--red);border:1px solid rgba(255,77,77,.25)'; notice.classList.add('show'); }
}

function renderEvents(events) {
  const html = (events || []).map(e => `<div class="alert-item alert-${e.level}"><div class="alert-time">${e.t}</div><div class="alert-msg">${e.msg}</div></div>`).join('');
  ['alert-log', 'soc-full-log'].forEach(id => { const el = $(id); if (el) el.innerHTML = html; });
}
function setShap(msg) { const e = $('shap-verdict'); if (e) e.textContent = msg; }
function setConn(on) { const p = $('conn-pill'); if (!p) return; p.classList.toggle('online', on); $('conn-text').textContent = on ? 'Engine online' : 'Reconnecting…'; }

/* ── SOC console (WebSocket) ───────────────────────────────────────────────*/
function connectSOC() {
  if (socWS) return;
  socWS = new WebSocket((location.protocol === 'https:' ? 'wss' : 'ws') + '://' + location.host + '/ws/soc');
  socWS.onopen = () => { $('soc-ws').textContent = '● connected'; $('soc-ws').style.color = 'var(--green)'; };
  socWS.onmessage = e => renderSOC(JSON.parse(e.data));
  socWS.onclose = () => { socWS = null; $('soc-ws').textContent = 'disconnected'; $('soc-ws').style.color = 'var(--red)'; };
}
function renderSOC(d) {
  const sess = d.sessions || [];
  setText('soc-active', sess.length);
  setText('soc-atrisk', sess.filter(s => s.dts < 50).length);
  setText('soc-blocked', sess.filter(s => s.blocked).length);
  $('soc-sessions-body').innerHTML = sess.map(s => {
    const cc = col(s.dts), rc = { low: 'green', moderate: 'amber', high: 'red', critical: 'red' }[s.risk] || 'green';
    return `<tr style="border-bottom:1px solid var(--border)">
      <td style="padding:9px 10px;font-weight:600">${s.user}${s.session_id === SID ? ' <span style="color:var(--blue);font-size:9px">(you)</span>' : ''}</td>
      <td style="padding:9px 10px;font-family:var(--mono);font-size:10px;color:var(--muted2)">${s.session_id}</td>
      <td style="padding:9px 10px;color:var(--muted2)">${s.geo_city}</td>
      <td style="padding:9px 10px">${s.channel === 'mobile' ? '📱 mobile' : '💻 web'}</td>
      <td style="padding:9px 10px">${s.enrolled ? '<span style="color:var(--green)">Personal</span>' : '<span style="color:var(--muted2)">Population</span>'}</td>
      <td style="padding:9px 10px;text-align:right;font-weight:700;color:${cc}">${s.dts}</td>
      <td style="padding:9px 10px;text-align:right"><span style="font-size:10px;padding:2px 8px;border-radius:10px;background:var(--${rc}-dim);color:var(--${rc})">${s.blocked ? 'BLOCKED' : s.risk}</span></td>
    </tr>`;
  }).join('') || '<tr><td colspan="7" style="padding:14px;color:var(--muted2);text-align:center">No active sessions</td></tr>';
  $('soc-audit-log').innerHTML = (d.audit || []).map(a => `<div class="alert-item alert-${a.level}"><div class="alert-time">${a.t} · ${a.username} · ${a.session_id}</div><div class="alert-msg">${a.msg}</div></div>`).join('');
}
function exportAudit() { window.open('/api/soc/export', '_blank'); }
function toggleSOC() { document.querySelector('.soc').classList.toggle('soc-open'); }

/* ── chart ─────────────────────────────────────────────────────────────────*/
let dtsChart;
function initChart() {
  dtsChart = new Chart($('dts-chart').getContext('2d'), {
    type: 'line',
    data: { labels: Array(30).fill(''), datasets: [{ data: Array(30).fill(94), borderColor: '#00C97A', borderWidth: 1.5, fill: true, backgroundColor: c => { const g = c.chart.ctx.createLinearGradient(0, 0, 0, 120); g.addColorStop(0, 'rgba(0,201,122,.2)'); g.addColorStop(1, 'rgba(0,201,122,0)'); return g; }, tension: .4, pointRadius: 0 }] },
    options: { responsive: true, maintainAspectRatio: false, animation: { duration: 300 }, plugins: { legend: { display: false }, tooltip: { enabled: false } }, scales: { x: { display: false }, y: { display: false, min: 0, max: 100 } } }
  });
}
function updateChart(history, dts) {
  if (!dtsChart || !history) return;
  const data = history.slice(-30); while (data.length < 30) data.unshift(data[0] ?? 94);
  dtsChart.data.datasets[0].data = data;
  dtsChart.data.datasets[0].borderColor = dts > 70 ? '#00C97A' : dts > 40 ? '#FFB020' : '#FF4D4D';
  dtsChart.update('none');
}

/* ── navigation + clock + statements ───────────────────────────────────────*/
function switchNav(el, panel) {
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  if (el) el.classList.add('active');
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  const t = $('panel-' + panel); if (t) t.classList.add('active');
  const titles = { dashboard: 'Dashboard', transfer: 'Transfer funds', statements: 'Statements', calibrate: 'Calibrate', 'soc-view': 'SOC view', 'soc-console': 'SOC console', simulate: 'Simulate' };
  setText('page-title', titles[panel] || panel); setText('page-crumb', titles[panel] || panel);
  if (panel === 'calibrate') calibBegin();
  if (panel === 'soc-console') connectSOC();
  if (SID && !attackActive && !otpOpen && !blocked) score({ nav: true });
}
function clock() { setText('clock', new Date().toLocaleTimeString('en-IN', { hour12: false })); }

const STMT = [
  { date: '22 Jun', desc: 'Swiggy Food', type: 'Debit', amt: -348, bal: 482150 },
  { date: '22 Jun', desc: 'UPI Transfer — Rajesh K', type: 'Debit', amt: -5000, bal: 482498 },
  { date: '20 Jun', desc: 'Salary — PSG College', type: 'Credit', amt: 85000, bal: 487498 },
  { date: '18 Jun', desc: 'BESCOM Electricity', type: 'Debit', amt: -2140, bal: 402498 },
  { date: '15 Jun', desc: 'Amazon Pay', type: 'Debit', amt: -1299, bal: 404638 },
  { date: '12 Jun', desc: 'Apollo Pharmacy', type: 'Debit', amt: -860, bal: 405937 },
];
function fillStatements() {
  $('stmt-body').innerHTML = STMT.map(r => `<tr style="border-bottom:1px solid var(--border)">
    <td style="padding:9px 10px;color:var(--muted2)">${r.date}</td><td style="padding:9px 10px">${r.desc}</td>
    <td style="padding:9px 10px"><span style="font-size:10px;padding:2px 8px;border-radius:10px;${r.type === 'Credit' ? 'background:var(--green-dim);color:var(--green)' : 'background:var(--red-dim);color:var(--red)'}">${r.type}</span></td>
    <td style="padding:9px 10px;text-align:right;font-weight:600;color:${r.amt > 0 ? 'var(--green)' : 'var(--red)'}">${r.amt > 0 ? '+' : ''}₹${Math.abs(r.amt).toLocaleString('en-IN')}</td>
    <td style="padding:9px 10px;text-align:right;font-family:var(--mono);font-size:11px">₹${r.bal.toLocaleString('en-IN')}</td></tr>`).join('');
}

document.addEventListener('DOMContentLoaded', () => {
  initChart(); fillStatements(); clock(); setInterval(clock, 1000);
  $('calib-input').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); calibCommit(); } });
  document.querySelectorAll('.bio').forEach(inp => inp.addEventListener('paste', () => { if (SID && !attackActive && !otpOpen && !blocked) score({ paste: inp.id }); }));
  loadUsers();
  setTimeout(() => $('login-username').focus(), 200);
});
