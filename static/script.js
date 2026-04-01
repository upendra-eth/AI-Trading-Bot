document.addEventListener('DOMContentLoaded', () => {
    // Tab Navigation
    const navBtns = document.querySelectorAll('.nav-btn');
    const tabs = document.querySelectorAll('.tab-content');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            navBtns.forEach(b => b.classList.remove('active'));
            tabs.forEach(t => t.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.add('active');
        });
    });

    const formatCurr = (val) => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR' }).format(val);
    const formatPct = (val) => (val > 0 ? '+' : '') + val.toFixed(2) + '%';

    // -----------------------------------------------------------------------------
    // API configuration
    // -----------------------------------------------------------------------------
    // GitHub Pages hosts only static files, so /api/* endpoints will 404.
    // If you have the FastAPI backend running elsewhere, set:
    //   localStorage.setItem('AIQUANT_API_BASE_URL', 'http://127.0.0.1:8000')
    // then refresh.
    const IS_GITHUB_PAGES = window.location.hostname.endsWith('github.io');
    const API_BASE_URL = (localStorage.getItem('AIQUANT_API_BASE_URL') || '').trim();
    const BACKEND_ENABLED = !IS_GITHUB_PAGES || API_BASE_URL.length > 0;

    function apiUrl(path) {
        if (!BACKEND_ENABLED) return null;
        if (!path.startsWith('/')) path = '/' + path;
        const base = API_BASE_URL.length > 0 ? API_BASE_URL.replace(/\/+$/, '') : '';
        return `${base}${path}`;
    }

    function ensureBackendOrAlert(featureName) {
        if (BACKEND_ENABLED) return true;
        alert(
            `${featureName} needs the FastAPI backend.\n\n` +
            `You're viewing the static GitHub Pages demo, so /api/* is unavailable.\n\n` +
            `Run locally: python app.py\n` +
            `Then (optional) in browser console:\n` +
            `localStorage.setItem('AIQUANT_API_BASE_URL','http://127.0.0.1:8000')\n` +
            `and refresh.`
        );
        return false;
    }

    async function fetchJson(path, options) {
        const url = apiUrl(path);
        if (!url) throw new Error('Backend not enabled');

        const res = await fetch(url, options);
        const ct = (res.headers.get('content-type') || '').toLowerCase();

        if (!res.ok) {
            // Avoid JSON parse errors when server returns HTML error pages
            const text = await res.text().catch(() => '');
            throw new Error(`HTTP ${res.status} ${res.statusText}${text ? `: ${text.slice(0, 200)}` : ''}`);
        }

        if (!ct.includes('application/json')) {
            const text = await res.text().catch(() => '');
            throw new Error(`Expected JSON but got ${ct || 'unknown content-type'}${text ? `: ${text.slice(0, 200)}` : ''}`);
        }

        return await res.json();
    }

    function injectStaticNotice() {
        if (BACKEND_ENABLED) return;
        const container = document.querySelector('.content');
        if (!container) return;

        const notice = document.createElement('div');
        notice.className = 'panel glass';
        notice.style.marginBottom = '16px';
        notice.innerHTML = `
            <h3 style="margin-bottom:6px;">Static Demo Mode</h3>
            <div style="color:var(--text-muted); font-size:13px; line-height:1.6;">
                This GitHub Pages site hosts only the UI. API features (portfolio, scan, paper trading, backtest) require the FastAPI backend.
                <br/>
                To connect a running backend, open DevTools and set:
                <br/>
                <code style="display:inline-block; margin-top:8px; padding:6px 10px; background:rgba(0,0,0,0.25); border:1px solid var(--border); border-radius:8px;">
                    localStorage.setItem('AIQUANT_API_BASE_URL','http://127.0.0.1:8000')
                </code>
                <br/>
                Then refresh the page.
            </div>
        `;
        container.prepend(notice);
    }

    injectStaticNotice();

    // Fetch Portfolio Data
    async function loadPortfolio() {
        try {
            if (!BACKEND_ENABLED) return;
            const data = await fetchJson('/api/portfolio');

            document.getElementById('balance-value').textContent = formatCurr(data.balance);

            const activeTbody = document.querySelector('#active-trades-table tbody');
            activeTbody.innerHTML = '';
            data.active_trades.forEach(t => {
                activeTbody.innerHTML += `<tr>
                    <td><strong>${t.symbol}</strong></td>
                    <td>${formatCurr(t.entry_price)}</td>
                    <td>${t.quantity}</td>
                </tr>`;
            });

            const recentTbody = document.querySelector('#recent-trades-table tbody');
            recentTbody.innerHTML = '';
            data.recent_closed_trades.forEach(t => {
                const pnlClass = t.pnl >= 0 ? 'positive' : 'negative';
                recentTbody.innerHTML += `<tr>
                    <td><strong>${t.symbol}</strong></td>
                    <td>${formatCurr(t.entry)}</td>
                    <td>${formatCurr(t.exit)}</td>
                    <td class="${pnlClass}">${formatCurr(t.pnl)}</td>
                </tr>`;
            });
        } catch (e) {
            console.error("Failed to load portfolio", e);
        }
    }

    // Initial Load
    loadPortfolio();

    // Charting Setup using Lightweight Charts (Wrapped in try/catch to prevent script failure if CDN is blocked)
    let chart, lineSeries;
    try {
        const chartContainer = document.getElementById('tv-chart');
        chart = LightweightCharts.createChart(chartContainer, {
            layout: {
                background: { type: 'solid', color: 'transparent' },
                textColor: '#94a3b8',
            },
            watermark: {
                color: 'rgba(255, 255, 255, 0.1)',
                visible: true,
                text: 'AI Trading Engine',
                fontSize: 48,
                horzAlign: 'center',
                vertAlign: 'center',
            },
            grid: { vertLines: { color: 'rgba(255,255,255,0.05)' }, horzLines: { color: 'rgba(255,255,255,0.05)' } },
            rightPriceScale: { borderVisible: false },
            timeScale: { borderVisible: false },
        });

        new ResizeObserver(entries => {
            if (entries.length === 0 || entries[0].target !== chartContainer) return;
            const newRect = entries[0].contentRect;
            chart.applyOptions({ height: newRect.height, width: newRect.width });
        }).observe(chartContainer);

        const candleOpts = { upColor: '#10b981', downColor: '#ef4444', borderVisible: false, wickUpColor: '#10b981', wickDownColor: '#ef4444' };
        if (typeof chart.addSeries === 'function' && LightweightCharts.CandlestickSeries) {
            lineSeries = chart.addSeries(LightweightCharts.CandlestickSeries, candleOpts);
        } else if (typeof chart.addCandlestickSeries === 'function') {
            lineSeries = chart.addCandlestickSeries(candleOpts);
        } else {
            console.warn("Could not find addSeries or addCandlestickSeries on chart object.");
        }
    } catch (e) {
        console.warn("Failed to initialize TradingView charts. CDN might be blocked.", e);
    }

    // Backtest function
    document.getElementById('run-bt-btn').addEventListener('click', async () => {
        if (!ensureBackendOrAlert('Backtest')) return;
        const symbol = document.getElementById('bt-symbol').value;
        const start = document.getElementById('bt-start').value;
        const end = document.getElementById('bt-end').value;
        const interval = document.getElementById('bt-interval').value;

        document.getElementById('bt-loading').classList.remove('hidden');
        document.getElementById('bt-results').classList.add('hidden');
        document.getElementById('chart-panel').classList.add('hidden');

        try {
            const data = await fetchJson('/api/backtest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol, start_date: start, end_date: end, interval: interval })
            });

            document.getElementById('bt-loading').classList.add('hidden');

            if (data.error || data.detail) {
                alert(data.error || data.detail);
                return;
            }

            document.getElementById('bt-results').classList.remove('hidden');

            const ret = document.getElementById('bt-return');
            ret.textContent = formatPct(data.total_return_pct);
            ret.className = data.total_return_pct >= 0 ? 'metric positive' : 'metric negative';

            document.getElementById('bt-winrate').textContent = data.win_rate_pct.toFixed(1) + '%';
            document.getElementById('bt-trades').textContent = data.total_trades;

            if (lineSeries && chart) {
                try {
                    document.getElementById('chart-panel').classList.remove('hidden');

                    // Map OHLC
                    const candlestickData = data.equity_curve.map(d => ({
                        time: d.time,
                        open: d.open,
                        high: d.high,
                        low: d.low,
                        close: d.close
                    }));

                    lineSeries.setData(candlestickData);
                    chart.timeScale().fitContent();

                    chart.applyOptions({
                        watermark: {
                            text: `${symbol.toUpperCase()} (${interval})`
                        }
                    });

                    const markers = (data.trade_history || []).map(t => {
                        if (t.type === 'BUY') {
                            return { time: t.time, position: 'belowBar', color: '#10b981', shape: 'arrowUp', text: 'BUY' };
                        } else if (t.type === 'SELL') {
                            const pnlText = t.pnl >= 0 ? `+₹${t.pnl.toFixed(0)}` : `-₹${Math.abs(t.pnl).toFixed(0)}`;
                            return { time: t.time, position: 'aboveBar', color: '#ef4444', shape: 'arrowDown', text: `SELL (${pnlText})` };
                        }
                    }).filter(m => m !== undefined);

                    // Fix Lightweight Charts duplicate time bug
                    const uniqueMarkers = [];
                    const seenTimes = new Set();
                    markers.sort((a, b) => new Date(a.time) - new Date(b.time)).forEach(m => {
                        if (!seenTimes.has(m.time)) {
                            seenTimes.add(m.time);
                            uniqueMarkers.push(m);
                        }
                    });

                    lineSeries.setMarkers(uniqueMarkers);
                } catch (err) {
                    console.error("Charting Error:", err);
                }
            }

            // Render detailed trade log
            try {
                const logTbody = document.querySelector('#bt-trade-log-table tbody');
                if (logTbody) {
                    logTbody.innerHTML = '';
                    document.getElementById('trade-log-panel').classList.remove('hidden');

                    (data.trade_history || []).forEach(t => {
                        const isBuy = t.type === 'BUY';
                        const sig = t.signal_details || {};
                        const weightObj = Number(sig.weighted_score) || 0;

                        const formatDir = (dir) => {
                            if (!dir) return '-';
                            if (dir === 'BULLISH') return `<span class="positive">BULLISH</span>`;
                            if (dir === 'BEARISH') return `<span class="negative">BEARISH</span>`;
                            return `<span class="neutral">NEUTRAL</span>`;
                        };

                        let openPriceStr = isBuy ? formatCurr(t.price) : (t.entry_price ? formatCurr(t.entry_price) : '-');
                        let closePriceStr = isBuy ? '-' : formatCurr(t.price);
                        let pnlStr = isBuy ? '-' : (t.pnl >= 0 ? `<span class="positive">+${formatCurr(t.pnl)}</span>` : `<span class="negative">-${formatCurr(Math.abs(t.pnl))}</span>`);
                        let reasoning = sig.explanation || '-';

                        logTbody.innerHTML += `<tr>
                            <td>${t.date}</td>
                            <td class="${isBuy ? 'positive' : 'negative'}"><strong>${t.type}</strong></td>
                            <td>${openPriceStr}</td>
                            <td>${closePriceStr}</td>
                            <td><strong>${pnlStr}</strong></td>
                            <td style="font-size: 11px; max-width: 250px; white-space: normal; line-height: 1.4;">${reasoning}</td>
                            <td style="font-size: 11px;">
                                XGB: ${formatDir(sig.xgb)}<br>
                                LSTM: ${formatDir(sig.lstm)}<br>
                                FB: ${formatDir(sig.finbert)}
                            </td>
                        </tr>`;
                    });
                }
            } catch (err) {
                console.error("Trade Log Render Error:", err);
            }

        } catch (e) {
            console.error("Global Backtest Error:", e);
            alert("Error: " + (e.message || "Failed to parse backtest results"));
            document.getElementById('bt-loading').classList.add('hidden');
        }
    });

    // Scanner Logic
    document.getElementById('refresh-scan-btn').addEventListener('click', async () => {
        if (!ensureBackendOrAlert('Market scan')) return;
        const btn = document.getElementById('refresh-scan-btn');
        const text = document.getElementById('scan-btn-text');
        const loading = document.getElementById('scan-loading');
        const container = document.getElementById('scan-results');

        btn.disabled = true;
        text.textContent = 'Scanning...';
        loading.classList.remove('hidden');
        container.innerHTML = '';

        try {
            const data = await fetchJson('/api/scan');

            loading.classList.add('hidden');
            btn.disabled = false;
            text.textContent = 'Run Scan Now';

            if (data.scan_results) {
                data.scan_results.forEach(stock => {
                    const sig = stock.signal;
                    const finbert = sig.model_details.finbert;

                    let headlinesHtml = '';
                    if (finbert.headline_details && finbert.headline_details.length > 0) {
                        headlinesHtml = finbert.headline_details.map(h => {
                            const badgeCls = h.sentiment === 'POSITIVE' ? 'badge-pos' : (h.sentiment === 'NEGATIVE' ? 'badge-neg' : 'badge-neu');
                            return `<div class="headline-item">
                                <span class="badge ${badgeCls}">${h.sentiment} (${h.score > 0 ? '+' : ''}${h.score.toFixed(2)})</span>
                                <div class="headline-text">${h.headline}</div>
                            </div>`;
                        }).join('');
                    } else {
                        headlinesHtml = `<div class="score-explain">No recent news found.</div>`;
                    }

                    const card = document.createElement('div');
                    card.className = 'signal-card glass w-50';
                    card.innerHTML = `
                        <div class="signal-header">
                            <div class="header-titles">
                                <h2>${stock.symbol}</h2>
                                <div class="price-tag">${formatCurr(stock.price)}</div>
                            </div>
                            <div class="overall-signal signal-${sig.final_signal}">${sig.final_signal}</div>
                        </div>
                        
                        <div class="model-breakdown">
                            <div class="model-box">
                                <span class="lbl">XGBoost (35%)</span>
                                <span class="val ${sig.model_details.xgboost.direction === 'BULLISH' ? 'positive' : (sig.model_details.xgboost.direction === 'BEARISH' ? 'negative' : 'neutral')}">${sig.model_details.xgboost.direction}</span>
                                <div style="font-size: 11px; color: #94a3b8; margin-top: 6px;">
                                    <b>${(sig.model_details.xgboost.predicted_change_pct || 0).toFixed(2)}%</b> predicted<br>
                                    Conf: ${sig.model_details.xgboost.confidence}%<br>
                                    Data: ${sig.model_details.xgboost.data_points_used || '100+'} days
                                </div>
                            </div>
                            <div class="model-box">
                                <span class="lbl">LSTM (35%)</span>
                                <span class="val ${sig.model_details.lstm.direction === 'BULLISH' ? 'positive' : (sig.model_details.lstm.direction === 'BEARISH' ? 'negative' : 'neutral')}">${sig.model_details.lstm.direction}</span>
                                <div style="font-size: 11px; color: #94a3b8; margin-top: 6px;">
                                    <b>${(sig.model_details.lstm.predicted_vs_current || 0).toFixed(4)}</b> scale ∆<br>
                                    Conf: ${sig.model_details.lstm.confidence}%<br>
                                    Data: ${sig.model_details.lstm.data_points_used || '100+'} seqs
                                </div>
                            </div>
                            <div class="model-box">
                                <span class="lbl">FinBERT (30%)</span>
                                <span class="val ${finbert.direction === 'BULLISH' ? 'positive' : (finbert.direction === 'BEARISH' ? 'negative' : 'neutral')}">${finbert.direction}</span>
                                <div style="font-size: 11px; color: #94a3b8; margin-top: 6px;">
                                    <b>${finbert.score > 0 ? '+' : ''}${(finbert.score || 0).toFixed(2)}</b> avg score<br>
                                    Conf: ${finbert.confidence}%<br>
                                    Data: ${finbert.headlines_analyzed} headlines
                                </div>
                            </div>
                        </div>
                        <div class="score-explain">${sig.explanation}</div>
                        
                        <div class="news-section">
                            <h4>News Analysis</h4>
                            ${headlinesHtml}
                        </div>
                    `;
                    container.appendChild(card);
                });
            }
        } catch (e) {
            console.error(e);
            alert("Error running market scan");
            loading.classList.add('hidden');
            btn.disabled = false;
            text.textContent = 'Run Scan Now';
        }
    });

    // ─── Paper Trading Functions ─────────────────────────────────────────────────

    let paperStatusPoller = null;

    function updatePaperStatusBadge(enabled, running) {
        const badge = document.getElementById('pt-status-badge');
        const startBtn = document.getElementById('pt-start-btn');
        const stopBtn = document.getElementById('pt-stop-btn');
        if (!badge) return;
        if (running) {
            badge.textContent = '● RUNNING';
            badge.style.background = 'rgba(16,185,129,0.15)';
            badge.style.color = '#10b981';
        } else if (enabled) {
            badge.textContent = '● ACTIVE (Waiting)';
            badge.style.background = 'rgba(59,130,246,0.15)';
            badge.style.color = '#60a5fa';
        } else {
            badge.textContent = '● OFFLINE';
            badge.style.background = 'rgba(100,116,139,0.2)';
            badge.style.color = '#64748b';
        }
        startBtn.style.display = enabled ? 'none' : 'inline-block';
        stopBtn.style.display = enabled ? 'inline-block' : 'none';
    }

    async function loadPaperStatus() {
        try {
            if (!BACKEND_ENABLED) return;
            const [statusRes, portRes] = await Promise.all([
                fetchJson('/api/paper-trading/status'),
                fetchJson('/api/portfolio')
            ]);
            const status = statusRes;
            const port = portRes;

            updatePaperStatusBadge(status.enabled, status.running);

            document.getElementById('pt-last-run').textContent = status.last_run
                ? new Date(status.last_run).toLocaleString('en-IN')
                : 'Never';
            document.getElementById('pt-next-run').textContent = status.next_run
                ? new Date(status.next_run).toLocaleString('en-IN')
                : '—';

            // Portfolio metrics
            document.getElementById('pt-balance').textContent = formatCurr(port.balance);
            const pnlEl = document.getElementById('pt-total-pnl');
            pnlEl.textContent = (port.total_pnl >= 0 ? '+' : '') + formatCurr(port.total_pnl || 0);
            pnlEl.className = (port.total_pnl || 0) >= 0 ? 'positive' : 'negative';
            document.getElementById('pt-win-rate').textContent = port.win_rate_pct > 0 ? port.win_rate_pct.toFixed(1) + '%' : '—';
            document.getElementById('pt-trade-count').textContent = port.total_closed_trades || 0;

            // Active positions
            const activeTbody = document.querySelector('#pt-active-table tbody');
            if (port.active_trades && port.active_trades.length > 0) {
                activeTbody.innerHTML = port.active_trades.map(t => `<tr>
                    <td><strong>${t.symbol}</strong></td>
                    <td>${formatCurr(t.entry_price)}</td>
                    <td>${t.quantity}</td>
                    <td style="font-size:11px; color:var(--text-muted);">${t.entry_time ? new Date(t.entry_time).toLocaleDateString('en-IN') : '-'}</td>
                </tr>`).join('');
            } else {
                activeTbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text-muted);">No open positions</td></tr>';
            }

            // Closed trades
            const closedTbody = document.querySelector('#pt-closed-table tbody');
            if (port.recent_closed_trades && port.recent_closed_trades.length > 0) {
                closedTbody.innerHTML = port.recent_closed_trades.map(t => {
                    const pnlClass = (t.pnl || 0) >= 0 ? 'positive' : 'negative';
                    return `<tr>
                        <td><strong>${t.symbol}</strong></td>
                        <td>${formatCurr(t.entry)}</td>
                        <td>${t.exit ? formatCurr(t.exit) : '-'}</td>
                        <td>${t.quantity}</td>
                        <td class="${pnlClass}"><strong>${(t.pnl >= 0 ? '+' : '') + formatCurr(t.pnl || 0)}</strong></td>
                    </tr>`;
                }).join('');
            } else {
                closedTbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);">No closed trades yet</td></tr>';
            }

            // Cycle log
            if (status.last_cycle_log && status.last_cycle_log.length > 0) {
                const logEl = document.getElementById('pt-log');
                logEl.innerHTML = status.last_cycle_log.map(line => {
                    let color = '#94a3b8';
                    if (line.includes('BUY EXECUTED')) color = '#10b981';
                    else if (line.includes('SELL EXECUTED')) color = '#f59e0b';
                    else if (line.includes('ERROR')) color = '#ef4444';
                    else if (line.includes('Cycle')) color = '#60a5fa';
                    return `<div style="color:${color};">${line}</div>`;
                }).join('');
                logEl.scrollTop = logEl.scrollHeight;
            }
        } catch (e) {
            console.error('Paper status error:', e);
        }
    }

    // Auto-poll when on paper trading tab
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.dataset.tab === 'paper-trading') {
                loadPaperStatus();
                if (!paperStatusPoller) {
                    paperStatusPoller = setInterval(loadPaperStatus, 10000);
                }
            } else {
                if (paperStatusPoller) {
                    clearInterval(paperStatusPoller);
                    paperStatusPoller = null;
                }
            }
        });
    });

    window.startPaperEngine = async function() {
        if (!ensureBackendOrAlert('Paper trading')) return;
        const interval = document.getElementById('pt-interval').value;
        try {
            await fetchJson(`/api/paper-trading/start?interval_minutes=${interval}`, { method: 'POST' });
            loadPaperStatus();
            if (!paperStatusPoller) paperStatusPoller = setInterval(loadPaperStatus, 10000);
        } catch (e) { alert('Failed to start engine: ' + e.message); }
    };

    window.stopPaperEngine = async function() {
        if (!ensureBackendOrAlert('Paper trading')) return;
        try {
            await fetchJson('/api/paper-trading/stop', { method: 'POST' });
            loadPaperStatus();
        } catch (e) { alert('Failed to stop engine: ' + e.message); }
    };

    window.runPaperNow = async function() {
        if (!ensureBackendOrAlert('Paper trading')) return;
        const logEl = document.getElementById('pt-log');
        logEl.innerHTML = '<div style="color:#60a5fa;">⚡ Running manual cycle... please wait (30–90 seconds).</div>';
        try {
            const data = await fetchJson('/api/paper-trading/run-now', { method: 'POST' });
            if (data.cycle_log) {
                logEl.innerHTML = data.cycle_log.map(line => {
                    let color = '#94a3b8';
                    if (line.includes('BUY EXECUTED')) color = '#10b981';
                    else if (line.includes('SELL EXECUTED')) color = '#f59e0b';
                    else if (line.includes('ERROR')) color = '#ef4444';
                    else if (line.includes('Cycle')) color = '#60a5fa';
                    return `<div style="color:${color};">${line}</div>`;
                }).join('');
                logEl.scrollTop = logEl.scrollHeight;
            }
            loadPaperStatus();
        } catch (e) { alert('Cycle failed: ' + e.message); }
    };

    window.resetPortfolio = async function() {
        if (!ensureBackendOrAlert('Paper trading')) return;
        if (!confirm('Reset paper portfolio to ₹1,00,000? This will close all open positions.')) return;
        try {
            await fetchJson('/api/paper-trading/reset', { method: 'POST' });
            loadPaperStatus();
        } catch (e) { alert('Reset failed: ' + e.message); }
    };

});
