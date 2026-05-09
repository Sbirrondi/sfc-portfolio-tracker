"""UI helpers for the Streamlit portfolio dashboard."""

from html import escape

import streamlit as st


CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .main .block-container {
        padding-top: 0.5rem;
        max-width: 1600px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    #MainMenu, footer,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stHeaderActionElements"],
    .stDeployButton {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
    }

    .sidebar-toggle-btn {
        position: fixed;
        top: 0.6rem;
        left: 0.6rem;
        z-index: 999999;
        background: rgba(99,102,241,0.2);
        border: 1px solid rgba(99,102,241,0.4);
        border-radius: 10px;
        padding: 0.5rem 0.6rem;
        cursor: pointer;
        backdrop-filter: blur(10px);
        box-shadow: 0 2px 12px rgba(0,0,0,0.4);
        transition: all 0.2s;
        display: none;
    }
    .sidebar-toggle-btn:hover {
        background: rgba(99,102,241,0.4);
        border-color: rgba(99,102,241,0.6);
    }
    .sidebar-toggle-btn svg { width:22px; height:22px; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080810 0%, #0d0d1a 100%);
        border-right: 1px solid rgba(99,102,241,0.12);
    }

    .fund-banner {
        background: linear-gradient(135deg, #0a0a14 0%, #12122a 50%, #0f1a2e 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        color: white;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        border: 1px solid rgba(99,102,241,0.18);
        margin-bottom: 0.6rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .fund-banner img { width: 52px; height: 52px; border-radius: 50%; background: white; padding: 3px; }
    .fund-banner h1 { margin:0; font-size:1.3rem; font-weight:700; color:#e2e8f0; }
    .fund-banner p { margin:0.1rem 0 0; color:#64748b; font-size:0.75rem; letter-spacing:0.3px; }

    .kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:0.7rem; margin-bottom:0.8rem; }
    .kpi-card {
        background: linear-gradient(135deg, #0d0d1a 0%, #13132a 100%);
        border: 1px solid rgba(99,102,241,0.10);
        border-radius:10px;
        padding: 0.9rem 1.1rem;
        transition: border-color 0.2s;
        min-width: 0;
    }
    .kpi-card:hover { border-color: rgba(99,102,241,0.3); }
    .kpi-label { font-size:0.65rem; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:0.35rem; }
    .kpi-value { font-size:1.45rem; font-weight:700; color:#e2e8f0; line-height:1.2; overflow-wrap:anywhere; }
    .kpi-delta { font-size:0.72rem; font-weight:500; margin-top:0.25rem; color:#64748b; }
    .kpi-delta .pos { color:#22c55e; font-weight:600; }
    .kpi-delta .neg { color:#ef4444; font-weight:600; }
    .accent-purple { border-left:3px solid #6366f1; }
    .accent-green { border-left:3px solid #22c55e; }
    .accent-blue { border-left:3px solid #3b82f6; }
    .accent-amber { border-left:3px solid #f59e0b; }

    .section-header {
        font-size:0.8rem;
        font-weight:600;
        color:#94a3b8;
        text-transform:uppercase;
        letter-spacing:0.8px;
        margin:1.2rem 0 0.5rem;
        padding-bottom:0.4rem;
        border-bottom:1px solid rgba(99,102,241,0.10);
    }

    .perf-table {
        width:100%;
        border-collapse:separate;
        border-spacing:0;
        font-size:0.8rem;
        border-radius:8px;
        overflow:hidden;
        border:1px solid rgba(99,102,241,0.10);
    }
    .perf-table thead th {
        background:#0d0d1a;
        color:#94a3b8;
        font-weight:600;
        font-size:0.65rem;
        text-transform:uppercase;
        letter-spacing:0.5px;
        padding:0.55rem 0.7rem;
        text-align:right;
        border-bottom:1px solid rgba(99,102,241,0.12);
    }
    .perf-table thead th:first-child { text-align:left; }
    .perf-table tbody td {
        padding:0.5rem 0.7rem;
        text-align:right;
        color:#cbd5e1;
        border-bottom:1px solid rgba(99,102,241,0.05);
    }
    .perf-table tbody td:first-child { text-align:left; font-weight:500; color:#e2e8f0; }
    .perf-table tbody tr:hover { background:rgba(99,102,241,0.04); }
    .perf-table .pos { color:#22c55e; font-weight:600; }
    .perf-table .neg { color:#ef4444; font-weight:600; }

    .stat-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.45rem; }
    .stat-item {
        background:#0d0d1a;
        border:1px solid rgba(99,102,241,0.07);
        border-radius:8px;
        padding:0.6rem 0.8rem;
    }
    .stat-label { font-size:0.6rem; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; }
    .stat-value { font-size:1.05rem; font-weight:700; color:#e2e8f0; margin-top:0.15rem; }

    .mover-item {
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:0.4rem 0;
        border-bottom:1px solid rgba(99,102,241,0.05);
        font-size:0.78rem;
    }
    .mover-item:last-child { border-bottom:none; }
    .mover-name { color:#cbd5e1; font-weight:500; max-width:68%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .mover-pnl { font-weight:700; }
    .mover-pnl.pos { color:#22c55e; }
    .mover-pnl.neg { color:#ef4444; }
    .mover-section { font-size:0.6rem; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; margin:0.5rem 0 0.3rem; }
    .mover-section:first-child { margin-top:0; }

    .stTabs [data-baseweb="tab-list"] {
        gap:0.2rem;
        background:rgba(13,13,26,0.6);
        padding:0.25rem;
        border-radius:10px;
        border:1px solid rgba(99,102,241,0.08);
    }
    .stTabs [data-baseweb="tab"] { border-radius:8px; padding:0.4rem 1rem; font-weight:500; font-size:0.8rem; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d0d1a 0%, #13132a 100%);
        border:1px solid rgba(99,102,241,0.08);
        border-radius:10px;
        padding:0.7rem 0.9rem;
    }
    [data-testid="stMetricLabel"] { font-size:0.68rem !important; text-transform:uppercase; letter-spacing:0.4px; }
    [data-testid="stExpander"] {
        border:1px solid rgba(99,102,241,0.08);
        border-radius:10px;
        background:rgba(13,13,26,0.3);
    }

    .position-kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.7rem;
        margin: 0.2rem 0 1.1rem;
    }
    .position-kpi-card {
        min-width: 0;
        background: linear-gradient(135deg, #0d0d1a 0%, #13132a 100%);
        border: 1px solid rgba(99,102,241,0.10);
        border-left: 3px solid var(--accent, #6366f1);
        border-radius: 10px;
        padding: 0.8rem 0.95rem;
        overflow: hidden;
    }
    .position-kpi-label {
        color: #94a3b8;
        font-size: 0.64rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        line-height: 1.2;
        margin-bottom: 0.38rem;
        text-transform: uppercase;
    }
    .position-kpi-value {
        color: #f8fafc;
        font-size: clamp(1.22rem, 1.7vw, 1.65rem);
        font-weight: 750;
        line-height: 1.1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .position-kpi-detail {
        color: #64748b;
        font-size: 0.69rem;
        line-height: 1.25;
        margin-top: 0.33rem;
        overflow-wrap: anywhere;
    }

    .access-panel {
        background: linear-gradient(135deg, rgba(13,13,26,0.95) 0%, rgba(19,19,42,0.95) 100%);
        border: 1px solid rgba(99,102,241,0.12);
        border-left: 3px solid #6366f1;
        border-radius: 10px;
        margin: 0.2rem 0 0.85rem;
        padding: 0.95rem 1rem;
    }
    .access-panel-title {
        color: #e2e8f0;
        font-size: 0.92rem;
        font-weight: 750;
        line-height: 1.25;
        margin-bottom: 0.25rem;
    }
    .access-panel-copy {
        color: #94a3b8;
        font-size: 0.78rem;
        line-height: 1.45;
        max-width: 780px;
    }
    .access-panel-status {
        color: #64748b;
        font-size: 0.66rem;
        font-weight: 650;
        letter-spacing: 0.4px;
        margin-top: 0.55rem;
        text-transform: uppercase;
    }

    @media(max-width: 768px) {
        .main .block-container {
            padding-left: 0.6rem;
            padding-right: 0.6rem;
        }
        .kpi-grid{grid-template-columns:repeat(2,1fr); gap:0.4rem;}
        .kpi-value { font-size:1.1rem; }
        .kpi-label { font-size:0.58rem; }
        .kpi-delta { font-size:0.65rem; }
        .fund-banner { flex-direction:column; text-align:center; padding:0.8rem; gap:0.6rem; }
        .fund-banner img { width:44px; height:44px; }
        .fund-banner h1 { font-size:1.05rem; }
        .section-header { font-size:0.72rem; }
        .stat-grid { grid-template-columns:1fr; }
        .perf-table { font-size:0.7rem; }
        .perf-table thead th { font-size:0.58rem; padding:0.4rem; }
        .perf-table tbody td { padding:0.35rem 0.4rem; }
        .mover-name { max-width:55%; font-size:0.72rem; }
        .position-kpi-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.5rem;
        }
        .position-kpi-card {
            padding: 0.7rem 0.75rem;
        }
        .position-kpi-value {
            font-size: 1.12rem;
        }
    }
    @media(max-width: 430px) {
        .kpi-grid { grid-template-columns:1fr; }
        .kpi-value { font-size:1rem; }
        .position-kpi-grid {
            grid-template-columns: 1fr;
        }
    }

    ::-webkit-scrollbar { width:5px; }
    ::-webkit-scrollbar-track { background:#0a0a14; }
    ::-webkit-scrollbar-thumb { background:#2a2a4a; border-radius:3px; }
</style>
"""


SIDEBAR_TOGGLE_HTML = """
<div class="sidebar-toggle-btn" id="sidebarToggle" onclick="
    var btn = window.parent.document.querySelector('[data-testid=\\'stSidebarCollapsedControl\\'] button')
        || window.parent.document.querySelector('[data-testid=\\'collapsedControl\\'] button')
        || window.parent.document.querySelector('button[aria-label*=\\'sidebar\\']')
        || window.parent.document.querySelector('header button');
    if(btn) btn.click();
    else { window.parent.document.querySelector('[data-testid=\\'stSidebar\\']').style.display='block'; }
">
    <svg viewBox="0 0 24 24" fill="none" stroke="#e2e8f0" stroke-width="2.5" stroke-linecap="round">
        <line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/>
    </svg>
</div>
<script>
(function() {
    function polishChrome() {
        var root = window.parent.document;
        root.querySelectorAll('[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], [data-testid="stHeaderActionElements"], .stDeployButton')
            .forEach(function(el) { el.style.display = 'none'; el.style.visibility = 'hidden'; });
        root.querySelectorAll('header button').forEach(function(btn) {
            if ((btn.textContent || '').trim().toLowerCase() === 'deploy') {
                btn.style.display = 'none';
                btn.style.visibility = 'hidden';
            }
        });
    }
    function checkSidebar() {
        var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        var toggle = document.getElementById('sidebarToggle');
        if (!sidebar || !toggle) return;
        var collapsed = sidebar.getAttribute('aria-expanded') === 'false'
                     || sidebar.offsetWidth < 50
                     || sidebar.style.display === 'none'
                     || sidebar.classList.contains('st-emotion-cache-1cypcdb');
        toggle.style.display = collapsed ? 'block' : 'none';
    }
    setInterval(function() {
        polishChrome();
        checkSidebar();
    }, 500);
    polishChrome();
    checkSidebar();
})();
</script>
"""


def render_position_kpis(items: list[dict]) -> None:
    """Render compact responsive KPI cards for the positions page."""
    cards = []
    for item in items:
        label = escape(str(item.get("label", "")))
        value = escape(str(item.get("value", "")))
        detail = escape(str(item.get("detail", "")))
        accent = escape(str(item.get("accent", "#6366f1")))
        title = escape(str(item.get("title", detail or value)))
        cards.append(
            f"""<div class="position-kpi-card" style="--accent:{accent}" title="{title}">
                <div class="position-kpi-label">{label}</div>
                <div class="position-kpi-value">{value}</div>
                <div class="position-kpi-detail">{detail}</div>
            </div>"""
        )
    st.markdown(f'<div class="position-kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_management_gate(key: str, title: str, description: str) -> None:
    """Render a consistent password gate for management pages."""
    correct_pw = st.secrets.get("management", {}).get("password") if hasattr(st, "secrets") else None
    status = "Accesso protetto attivo" if correct_pw else "Password non configurata"
    st.markdown(
        f"""<div class="access-panel">
            <div class="access-panel-title">{escape(title)}</div>
            <div class="access-panel-copy">{escape(description)}</div>
            <div class="access-panel-status">{escape(status)}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    mgmt_pw = st.text_input("🔒 Password richiesta", type="password", key=f"pw_{key}")
    if not correct_pw:
        st.error("Password di gestione non configurata. Imposta [management] password in Streamlit Secrets.")
        st.stop()
    if mgmt_pw != correct_pw:
        if mgmt_pw:
            st.warning("Password errata")
        st.stop()
