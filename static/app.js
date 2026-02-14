/* ── Fluid-Geometric Dashboard — Frontend Logic ── */

const API = "";
let pollTimer = null;

// ──────────────────────── Init ───────────────────────
document.addEventListener("DOMContentLoaded", () => {
    loadHoldings();
    loadSignals();
    checkStatus();
});

// ──────────────────────── Holdings ───────────────────
async function loadHoldings() {
    const res = await fetch(`${API}/api/holdings`);
    const data = await res.json();
    renderHoldings(data);
}

function renderHoldings(holdings) {
    const el = document.getElementById("holdingsTable");
    const entries = Object.entries(holdings);

    if (entries.length === 0) {
        el.innerHTML = '<p class="empty-state">No positions — click + Add Position</p>';
        return;
    }

    // Sort by confidence descending
    entries.sort((a, b) => (b[1].confidence || 0) - (a[1].confidence || 0));

    let html = `<table>
    <thead><tr>
      <th>Ticker</th>
      <th>Direction</th>
      <th>Signal</th>
      <th>Confidence</th>
      <th>Source</th>
      <th></th>
    </tr></thead><tbody>`;

    for (const [ticker, h] of entries) {
        const dir = h.direction || "LONG";
        const dirClass = dir === "LONG" ? "dir-long" : "dir-short";
        const source = h.manual ? "Manual" : "Bot";
        const conf = h.confidence || 0;
        const signal = h.signal || "—";

        let signalClass = "";
        if (signal.includes("SNIPER")) signalClass = "label-sniper";
        else if (signal.includes("STRONG") || signal.includes("BUY")) signalClass = "label-strong";
        else if (signal === "MANUAL") signalClass = "";

        html += `<tr>
      <td class="ticker">${ticker}</td>
      <td><span class="${dirClass}">${dir}</span></td>
      <td><span class="signal-label ${signalClass}">${signal}</span></td>
      <td><span style="font-family:var(--mono)">${conf}</span>/10</td>
      <td style="color:var(--text-muted);font-size:12px">${source}</td>
      <td><button class="btn btn-danger" onclick="removeHolding('${ticker}')">✕</button></td>
    </tr>`;
    }

    html += "</tbody></table>";
    el.innerHTML = html;
}

async function removeHolding(ticker) {
    await fetch(`${API}/api/holdings/${ticker}`, { method: "DELETE" });
    loadHoldings();
}

// ──────────────────────── Signals ────────────────────
async function loadSignals() {
    const res = await fetch(`${API}/api/signals`);
    const signals = await res.json();
    renderSignals(signals);
    renderTrending(signals);
}

function renderSignals(signals) {
    const el = document.getElementById("signalsContainer");
    const countEl = document.getElementById("signalCount");

    const actionable = signals.filter(s => s.is_buy || s.is_exit);
    countEl.textContent = actionable.length;

    if (actionable.length === 0) {
        el.innerHTML = '<p class="empty-state">Run inference to generate signals</p>';
        return;
    }

    let html = "";
    for (const s of actionable) {
        let cardClass = "signal-card";
        let labelClass = "";
        let labelText = s.signal;

        if (s.signal === "SNIPER BUY") {
            cardClass += " sniper";
            labelClass = "label-sniper";
        } else if (s.is_buy) {
            cardClass += " buy";
            labelClass = "label-strong";
        } else if (s.is_exit) {
            cardClass += " exit";
            labelClass = "label-exit";
            labelText = "EXIT";
        }

        const marker = s.is_new ? " ★ New" : s.is_held ? " ↔ Held" : s.is_exit ? " ⬇ Drop" : "";

        html += `
      <div class="${cardClass}">
        <div class="signal-info">
          <span class="signal-ticker">${s.ticker}</span>
          <span class="signal-label ${labelClass}">${labelText}${marker}</span>
        </div>
        <div class="signal-conf">${s.confidence}<small>/10</small></div>
      </div>`;
    }

    el.innerHTML = html;
}

function renderTrending(signals) {
    const el = document.getElementById("trendingContainer");

    // Show all stocks sorted by confidence (top 25)
    const sorted = [...signals].sort((a, b) => b.confidence - a.confidence).slice(0, 25);

    if (sorted.length === 0) {
        el.innerHTML = '<p class="empty-state">Run inference to see trends</p>';
        return;
    }

    const maxConf = Math.max(...sorted.map(s => s.confidence), 1);

    let html = "";
    sorted.forEach((s, i) => {
        const pct = (s.confidence / 10) * 100;
        let color;
        if (s.confidence > 6) color = "var(--amber)";
        else if (s.confidence > 3) color = "var(--green)";
        else color = "var(--text-dim)";

        html += `
      <div class="trend-row">
        <span class="trend-rank">${i + 1}</span>
        <span class="trend-ticker">${s.ticker}</span>
        <div class="trend-bar-container">
          <div class="trend-bar" style="width:${pct}%;background:${color}"></div>
        </div>
        <span class="trend-value">${s.confidence}</span>
      </div>`;
    });

    el.innerHTML = html;
}

// ──────────────────────── Inference Runner ───────────
async function runInference(mode) {
    const btns = document.querySelectorAll(".controls-panel .btn");
    btns.forEach(b => b.disabled = true);

    const progressEl = document.getElementById("inferenceProgress");
    const logEl = document.getElementById("inferenceLog");
    progressEl.style.display = "block";
    logEl.style.display = "block";
    logEl.innerHTML = "";

    try {
        const res = await fetch(`${API}/api/run/inference`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode }),
        });

        const data = await res.json();
        if (data.error) {
            logEl.innerHTML += `<div style="color:var(--red)">${data.error}</div>`;
            btns.forEach(b => b.disabled = false);
            return;
        }

        // Start polling
        startPolling();
    } catch (e) {
        logEl.innerHTML += `<div style="color:var(--red)">Error: ${e.message}</div>`;
        btns.forEach(b => b.disabled = false);
    }
}

function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(pollStatus, 2000);
}

async function pollStatus() {
    try {
        const res = await fetch(`${API}/api/run/status`);
        const st = await res.json();

        const dot = document.getElementById("statusDot");
        const text = document.getElementById("statusText");
        const progressFill = document.getElementById("progressFill");
        const progressText = document.getElementById("progressText");
        const logEl = document.getElementById("inferenceLog");

        if (st.running) {
            dot.className = "status-dot running";
            text.textContent = st.progress || "Running …";
            progressText.textContent = st.progress || "Running …";

            // Parse progress for bar fill
            const match = st.progress?.match(/(\d+)\/(\d+)/);
            if (match) {
                const pct = (parseInt(match[1]) / parseInt(match[2])) * 100;
                progressFill.style.width = pct + "%";
            }

            // Update logs
            if (st.logs?.length) {
                logEl.innerHTML = st.logs.map(l => `<div>${l}</div>`).join("");
                logEl.scrollTop = logEl.scrollHeight;
            }
        } else {
            dot.className = "status-dot done";
            text.textContent = "Complete";
            progressFill.style.width = "100%";
            progressText.textContent = "✓ Complete";

            if (st.logs?.length) {
                logEl.innerHTML = st.logs.map(l => `<div>${l}</div>`).join("");
            }

            clearInterval(pollTimer);
            pollTimer = null;

            // Re-enable buttons after short delay
            setTimeout(() => {
                document.querySelectorAll(".controls-panel .btn").forEach(b => b.disabled = false);
            }, 500);

            // Refresh data
            loadHoldings();
            loadSignals();
        }
    } catch (e) {
        console.error("Poll error:", e);
    }
}

async function checkStatus() {
    try {
        const res = await fetch(`${API}/api/run/status`);
        const st = await res.json();
        const dot = document.getElementById("statusDot");
        const text = document.getElementById("statusText");

        if (st.running) {
            dot.className = "status-dot running";
            text.textContent = st.progress || "Running …";
            document.getElementById("inferenceProgress").style.display = "block";
            document.getElementById("inferenceLog").style.display = "block";
            startPolling();
        } else if (st.completed_at) {
            dot.className = "status-dot done";
            text.textContent = `Last: ${new Date(st.completed_at).toLocaleTimeString()}`;
        }
    } catch (e) { }
}

// ──────────────────────── Modal ──────────────────────
function showAddModal() {
    document.getElementById("modal").style.display = "flex";
    document.getElementById("modalTicker").focus();
}

function hideModal() {
    document.getElementById("modal").style.display = "none";
}

function closeModal(e) {
    if (e.target === document.getElementById("modal")) hideModal();
}

async function addPosition(e) {
    e.preventDefault();
    const ticker = document.getElementById("modalTicker").value.trim().toUpperCase();
    const direction = document.getElementById("modalDirection").value;

    if (!ticker) return;

    await fetch(`${API}/api/holdings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            ticker,
            direction,
            confidence: 0,
            signal: "MANUAL",
            action: 0,
        }),
    });

    hideModal();
    document.getElementById("modalTicker").value = "";
    loadHoldings();
}
