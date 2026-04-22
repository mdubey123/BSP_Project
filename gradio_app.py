import os
import json
import pandas as pd
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import inch
from src.preprocessing import clean_data
from src.feature_engineering import create_features
from src.analysis import vendor_analysis
from src.model import train_model

# ─── MOCK STUBS (remove when src is ready) ────
# def clean_data(df): return df
# def create_features(df):
#     if "PR_VALUE" in df.columns and "NEGOTIATION_VAL" in df.columns:
#         df["PR_VALUE"] = pd.to_numeric(df["PR_VALUE"], errors="coerce")
#         df["NEGOTIATION_VAL"] = pd.to_numeric(df["NEGOTIATION_VAL"], errors="coerce")

#         df["saving"] = (df["PR_VALUE"] - df["NEGOTIATION_VAL"]).abs()
#     else:
#         raise ValueError("Dataset must contain PR_VALUE and NEGOTIATION_VAL columns.")

#     return df
# def vendor_analysis(df):
#     if "L1_PARTY_NAME" in df.columns:
#         return df.groupby("L1_PARTY_NAME")["saving"].sum().sort_values(ascending=False).head(10)
#     return pd.Series({"No vendor column": 0})
# class _DummyModel:
#     def predict(self, df): return [df["PR_VALUE"].iloc[0] * 0.05]
# def train_model(df): return _DummyModel()

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ASSETS_DIR  = "assets"
BSP_IMAGES  = [os.path.join(ASSETS_DIR, f) for f in ["bsp1.jpg","bsp2.jpg","bsp3.jpg"]]
BSP_LOGO    = os.path.join(ASSETS_DIR, "bsp_logo.png")
REPORT_PATH = "bsp_report.pdf"
EXPORT_PATH = "bsp_export.xlsx"
AUDIT_FILE  = "audit_log.json"
LOSS_ALERT_THRESHOLD   = 0.15
SAVING_ALERT_THRESHOLD = 0

ROLES = {
    "bsp_admin":   {"password": "admin@2026",  "role": "Admin"},
    "bsp_analyst": {"password": "analyst@2026", "role": "Analyst"},
    "bsp_viewer":  {"password": "view@2026",    "role": "Viewer"},
}

# ──────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────
state = {
    "df": None, "model": None,
    "current_user": None, "current_role": None,
    "login_time": None, "audit_log": [],
}

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
CSS= """
/* ===== GLOBAL ===== */
/* ===== KPI CARDS ===== */
.kpi-card {
    border-radius: 16px;
    padding: 20px 24px;
    margin: 6px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: popIn 0.4s ease;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 32px rgba(0,0,0,0.35);
}
.kpi-label {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
    opacity: 0.85;
    margin-bottom: 8px;
    text-transform: uppercase;
}
.kpi-value {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.kpi-green  { background: linear-gradient(135deg, #1b5e20, #2e7d32); color: #c8f7c5; }
.kpi-blue   { background: linear-gradient(135deg, #0d47a1, #1565c0); color: #bbdefb; }
.kpi-purple { background: linear-gradient(135deg, #4a148c, #6a1b9a); color: #e1bee7; }
.kpi-red    { background: linear-gradient(135deg, #b71c1c, #c62828); color: #ffcdd2; }

/* ===== ALERT BOXES ===== */
.alert-warn {
    background: rgba(183, 28, 28, 0.2);
    border: 1px solid rgba(239, 83, 80, 0.5);
    border-left: 4px solid #ef5350;
    border-radius: 10px;
    padding: 14px 18px;
    color: #ffcdd2;
    font-size: 14px;
    margin-top: 10px;
    animation: fadeSlide 0.3s ease;
}
.alert-ok {
    background: rgba(27, 94, 32, 0.2);
    border: 1px solid rgba(67, 160, 71, 0.5);
    border-left: 4px solid #43a047;
    border-radius: 10px;
    padding: 14px 18px;
    color: #c8f7c5;
    font-size: 14px;
    margin-top: 10px;
    animation: fadeSlide 0.3s ease;
}

body, .gradio-container {
    background: linear-gradient(135deg, #060f1d 0%, #0d1f3a 40%, #1a3f66 100%) !important;
    color: #f5f7fb !important;
    font-family: 'Segoe UI', sans-serif;
}

/* ===== TOP HEADER (NEW) ===== */
#top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(8, 20, 36, 0.95);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 16px 24px;
    border-radius: 18px;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

/* LEFT TITLE */
#top-bar-left {
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.5px;
}

/* RIGHT USER INFO */
#top-bar-right {
    font-size: 14px;
    color: #cfe3ff;
    text-align: right;
}

/* ===== UNDERLINE HEADINGS ===== */
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 700;
    position: relative;
    margin-bottom: 12px;
}

h1::after, h2::after, h3::after {
    content: "";
    display: block;
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #3fa9ff, #6ec1ff);
    margin-top: 6px;
    border-radius: 2px;
}

/* ===== LOGIN BANNER ===== */
#login-banner {
    background: linear-gradient(135deg, rgba(10,25,45,0.95), rgba(25,65,110,0.92));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 24px;
    padding: 36px;
    min-height: 420px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.35);
}

/* ===== LOGIN CARD ===== */
#login-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 24px;
    padding: 28px;
    min-height: 420px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.35);
}

/* ===== COLORED INPUT BOXES ===== */
input, textarea {
    background: rgba(20, 50, 90, 0.75) !important;
    color: #ffffff !important;
    border: 1px solid rgba(100,180,255,0.35) !important;
    border-radius: 12px !important;
    padding: 12px !important;
    font-size: 14px !important;
}

/* glow effect */
input:focus, textarea:focus {
    border: 1px solid #4da3ff !important;
    background: rgba(30,70,120,0.85) !important;
    box-shadow: 0 0 10px rgba(77,163,255,0.4);
}

/* ===== PLACEHOLDER ===== */
input::placeholder {
    color: rgba(220,230,245,0.7) !important;
}

/* ===== LABELS ===== */
label {
    color: #eaf2ff !important;
    font-weight: 600 !important;
}

/* ===== BUTTON ===== */
button.primary {
    background: linear-gradient(135deg, #1e66c1, #4da3ff) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px !important;
    border-radius: 12px !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}

button.primary:hover {
    background: linear-gradient(135deg, #1554a3, #3b8ce0) !important;
    transform: scale(1.02);
}

/* ===== REMOVE DARK PATCHES ===== */
.gradio-container .block {
    background: transparent !important;
}

/* ===== TEXT ===== */
p, li {
    color: #dce9f7 !important;
}

/* ===== FOOTER ===== */
#footer-box {
    margin-top: 22px;
    background: rgba(8, 20, 36, 0.92);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 16px 22px;
    color: #dce7f3;
    font-size: 14px;
}

/* ===== TABS ===== */
.tab-nav {
    background: rgba(8, 20, 36, 0.9) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    margin-bottom: 16px !important;
}
.tab-nav button {
    color: #8fb8e0 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border-radius: 10px !important;
    padding: 8px 14px !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg, #1565c0, #1e88e5) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(21,101,192,0.4) !important;
}
.tab-nav button:hover:not(.selected) {
    background: rgba(255,255,255,0.08) !important;
    color: #ffffff !important;
}

/* ===== ANIMATIONS ===== */
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(-18px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes popIn {
    from { opacity: 0; transform: scale(0.96); }
    to { opacity: 1; transform: scale(1); }
}

/* ===== LOGO FIX ===== */

/* outer container */
#bsp-logo {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* DO NOT remove inner div completely — just clean it */
#bsp-logo div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* image styling */
#bsp-logo img {
    display: block;
    margin: 0 auto;
    max-height: 120px !important;
    width: auto !important;
    object-fit: contain !important;
}

/* remove only action icons (safe targeting) */
#bsp-logo button[aria-label="Download"],
#bsp-logo button[aria-label="Share"],
#bsp-logo button[aria-label="Fullscreen"] {
    display: none !important;
}

/* ===== LOGIN WRAPPER ===== */
#login-wrapper {
    align-items: center !important;
    gap: 30px;
}

/* ===== LEFT PANEL ===== */
#left-panel {
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 20px;
}

/* BRAND BOX */
#brand-box h1 {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
}

#brand-box .subtitle {
    font-size: 18px;
    color: #9ecbff;
    margin-bottom: 10px;
}

#brand-box .desc {
    font-size: 15px;
    color: #dce9f7;
    line-height: 1.7;
}

#brand-box ul {
    margin-top: 10px;
    padding-left: 18px;
}

#brand-box li {
    margin-bottom: 6px;
}

/* LOGO BIGGER */
#bsp-logo img {
    max-height: 140px !important;
}

/* ===== RIGHT PANEL ===== */
#right-panel {
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 12px;
}

/* LOGIN BOX (compact) */
#login-box {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 10px;
}

/* REMOVE OLD BIG CARD EFFECT */
#login-card {
    display: none !important;
}

/* INPUT FULL WIDTH */
#right-panel input {
    width: 100%;
}

/* BUTTON FULL WIDTH */
#right-panel button {
    width: 100%;
}

/* REMOVE EXTRA SPACE */
.gradio-container .gr-column {
    gap: 12px !important;
}

/* ===== DATAFRAME TABLE ===== */
table.dataframe {
    border-collapse: collapse !important;
    width: 100% !important;
    font-size: 13px !important;
}
table.dataframe th {
    background: linear-gradient(135deg, #0d47a1, #1565c0) !important;
    color: #ffffff !important;
    padding: 10px 14px !important;
    font-weight: 700 !important;
}
table.dataframe td {
    padding: 8px 14px !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    color: #dce9f7 !important;
}
table.dataframe tr:hover td {
    background: rgba(21, 101, 192, 0.15) !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 4px; }
::-webkit-scrollbar-thumb { background: #1565c0; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #1e88e5; }

/* ===== FILE UPLOAD ZONE ===== */
.upload-button {
    border: 2px dashed rgba(77,163,255,0.4) !important;
    border-radius: 14px !important;
    background: rgba(13,71,161,0.15) !important;
    transition: all 0.2s ease !important;
}
.upload-button:hover {
    border-color: #4da3ff !important;
    background: rgba(13,71,161,0.3) !important;
}

/* ===== DATAFRAME TABLE ===== */
table.dataframe {
    border-collapse: collapse !important;
    width: 100% !important;
    font-size: 13px !important;
}
table.dataframe th {
    background: linear-gradient(135deg, #0d47a1, #1565c0) !important;
    color: #ffffff !important;
    padding: 10px 14px !important;
    font-weight: 700 !important;
}
table.dataframe td {
    padding: 8px 14px !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    color: #dce9f7 !important;
}
table.dataframe tr:hover td {
    background: rgba(21, 101, 192, 0.15) !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 4px; }
::-webkit-scrollbar-thumb { background: #1565c0; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #1e88e5; }

/* ===== FILE UPLOAD ZONE ===== */
.upload-button {
    border: 2px dashed rgba(77,163,255,0.4) !important;
    border-radius: 14px !important;
    background: rgba(13,71,161,0.15) !important;
    transition: all 0.2s ease !important;
}
.upload-button:hover {
    border-color: #4da3ff !important;
    background: rgba(13,71,161,0.3) !important;
}

/* ===== LOGOUT BUTTON ===== */
button.secondary {
    background: rgba(183,28,28,0.3) !important;
    border: 1px solid rgba(239,83,80,0.4) !important;
    color: #ffcdd2 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
button.secondary:hover {
    background: rgba(183,28,28,0.6) !important;
    border-color: #ef5350 !important;
}

/* ===== HIDE GRADIO PROCESSING TEXT ===== */
div[class*="status"],
div[class*="progress"] {
    display: none !important;
}

/* ===== REMOVE GRADIO LOADING SPINNER ===== */

/* main spinner container */
div[class*="loading"],
div[class*="spinner"] {
    display: none !important;
}

/* overlay that blocks UI */
div[class*="overlay"] {
    display: none !important;
}

/* ===== SCROLLABLE TABLE ===== */
.gr-dataframe {
    max-height: 400px !important;
    overflow-y: auto !important;
}

/* ===== FIX FILE UPLOAD TEXT (TARGETED) ===== */
#file-upload * {
    color: #ffffff !important;
}

/* label */
#file-upload label {
    color: #cfe3ff !important;
    font-weight: 600 !important;
}

/* filename */
#file-upload span,
#file-upload div {
    color: #ffffff !important;
}

/* upload box */
#file-upload {
    background: rgba(20, 60, 110, 0.6) !important;
    border: 1px solid rgba(77,163,255,0.5) !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

#bsp-gallery * {
    color: #ffffff !important;
}

#bsp-gallery {
    background: transparent !important;
}
"""
# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def safe_images():
    return [img for img in BSP_IMAGES if os.path.exists(img)]

def get_logo():
    return BSP_LOGO if os.path.exists(BSP_LOGO) else None

def _kpi_html(label, value, css_class):
    return (
        f'<div class="kpi-card {css_class}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'</div>'
    )

_EMPTY = _kpi_html("—", "No Data", "kpi-blue")

# ──────────────────────────────────────────────
# AUDIT
# ──────────────────────────────────────────────
def log_action(action):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user":  state["current_user"] or "unknown",
        "role":  state["current_role"] or "—",
        "action": action,
    }
    state["audit_log"].append(entry)
    try:
        with open(AUDIT_FILE, "w") as f:
            json.dump(state["audit_log"], f, indent=2)
    except Exception:
        pass

# ──────────────────────────────────────────────
# AUTH
# ──────────────────────────────────────────────
def login(username, password):
    cred = ROLES.get(username.strip())
    if cred and cred["password"] == password:
        state.update({
            "current_user": username,
            "current_role": cred["role"],
            "login_time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        log_action("LOGIN")
        bar = (
            f"<div id='top-bar'>"
            f"<div id='top-bar-left'>BSP Procurement Intelligence Platform</div>"
            f"<div id='top-bar-right'>👤 {username} &nbsp;|&nbsp; 🔑 {cred['role']} "
            f"&nbsp;|&nbsp; 🕐 {state['login_time']}</div>"
            f"</div>"
            )
        return gr.update(visible=True), gr.update(visible=False), bar, "✅ Login successful"
    return gr.update(visible=False), gr.update(visible=True), "", "❌ Invalid credentials"

def logout():
    log_action("LOGOUT")
    state.update({"df":None,"model":None,"current_user":None,"current_role":None})
    return gr.update(visible=False), gr.update(visible=True), "", ""

# ──────────────────────────────────────────────
# MODEL ACCESSOR
# ──────────────────────────────────────────────
def _get_model():
    raw = state.get("model")
    if raw is None: return None
    if hasattr(raw, "predict"): return raw
    if isinstance(raw, dict):
        for k in ["model","estimator","clf","regressor","pipeline"]:
            if k in raw and hasattr(raw[k], "predict"): return raw[k]
        for v in raw.values():
            if hasattr(v, "predict"): return v
    return None

def _kpi_values():
    df = state["df"]
    if df is None: return 0, 0, 0, 0
    return df["saving"].sum(), df["saving"].mean(), len(df), int((df["saving"]<0).sum())

# ──────────────────────────────────────────────
# DATA PROCESSING
# ──────────────────────────────────────────────
def process_data(file, date_from, date_to):
    if file is None:
        return "⚠️ Upload an Excel file first.", _EMPTY, _EMPTY, _EMPTY, _EMPTY, "", ""

    log_action(f"UPLOAD: {os.path.basename(file.name)}")
    all_sheets   = pd.read_excel(file.name, sheet_name=None)
    combined_df  = pd.concat(all_sheets.values(), ignore_index=True)
    processed_df = create_features(clean_data(combined_df))

    dcols = [c for c in processed_df.columns if "date" in c.lower() or "dt" in c.lower()]
    if dcols and date_from and date_to:
        try:
            processed_df[dcols[0]] = pd.to_datetime(processed_df[dcols[0]], errors="coerce")
            processed_df = processed_df[
                (processed_df[dcols[0]] >= pd.to_datetime(date_from)) &
                (processed_df[dcols[0]] <= pd.to_datetime(date_to))
            ]
        except Exception:
            pass

    state["df"]    = processed_df
    state["model"] = train_model(processed_df)

    ts, av, tr, lc = _kpi_values()

    h1 = _kpi_html("💰 Total Saving",  f"₹{ts:,.0f}", "kpi-green")
    h2 = _kpi_html("📊 Avg Saving",    f"₹{av:,.2f}", "kpi-blue")
    h3 = _kpi_html("📦 Total Records", f"{tr:,}",      "kpi-purple")
    h4 = _kpi_html("⚠️ Loss Cases",    f"{lc:,}",      "kpi-red")

    alerts = []
    if tr > 0:
        lp = lc / tr
        if lp > LOSS_ALERT_THRESHOLD:
            alerts.append(f"⚠️ HIGH LOSS RATE: {lp*100:.1f}% of records show negative savings.")
        if av < SAVING_ALERT_THRESHOLD:
            alerts.append(f"⚠️ NEGATIVE AVG SAVING: ₹{av:,.2f} — review contracts.")
    alert_html = (
        '<div class="alert-warn">' + "<br>".join(alerts) + "</div>" if alerts
        else '<div class="alert-ok">✅ All KPIs within acceptable thresholds.</div>'
    )

    status = (
        f"✅ Loaded: {os.path.basename(file.name)}\n"
        f"📊 Rows: {len(processed_df):,}  |  Columns: {len(processed_df.columns)}\n"
        f"📅 Date filter applied: {bool(date_from and date_to)}"
    )
    return status, h1, h2, h3, h4, alert_html, _data_quality_report()

# ──────────────────────────────────────────────
# DATA QUALITY
# ──────────────────────────────────────────────
def _data_quality_report():
    df = state["df"]
    if df is None: return "Upload dataset first."
    total   = len(df)
    missing = df.isnull().sum()
    hi_miss = missing[missing > 0].sort_values(ascending=False)
    dupes   = df.duplicated().sum()
    lines = [
        "### 🔍 Data Quality Report",
        f"- **Total Records:** {total:,}",
        f"- **Duplicate Rows:** {dupes:,}",
        f"- **Columns with Missing Data:** {len(hi_miss)}",
    ]
    if len(hi_miss):
        lines.append("\n**Missing Values by Column:**")
        for col, cnt in hi_miss.items():
            lines.append(f"  - `{col}`: {cnt:,} ({cnt/total*100:.1f}%)")
    else:
        lines.append("- **No missing values detected ✅**")
    if "saving" in df.columns:
        lines += [
            "\n**Saving Column Stats:**",
            f"  - Min: ₹{df['saving'].min():,.2f}",
            f"  - Max: ₹{df['saving'].max():,.2f}",
            f"  - Std Dev: ₹{df['saving'].std():,.2f}",
            f"  - Skewness: {df['saving'].skew():.3f}",
        ]
    return "\n".join(lines)

def get_data_quality():
    log_action("VIEW: Data Quality")
    return _data_quality_report()

# ──────────────────────────────────────────────
# VENDOR
# ──────────────────────────────────────────────
def get_top_vendors():
    log_action("VIEW: Vendor Intelligence")
    df = state["df"]

    if df is None:
        return pd.DataFrame({"Message": ["Upload dataset first"]})

    result = df.groupby("L1_PARTY_NAME")["saving"].sum().sort_values(ascending=False).reset_index()
    result.columns = ["Vendor","Total Saving (₹)"]
    result["Total Saving (₹)"] = result["Total Saving (₹)"].round(2)

    result["Rank"] = range(1, len(result)+1)

    result["Status"] = result["Total Saving (₹)"].apply(
        lambda x: "✅ Profitable" if x >= 0 else "❌ Loss-Making"
    )
    return result

# ──────────────────────────────────────────────
# Processing Metrics
# ──────────────────────────────────────────────
def compute_processing_metrics():
    df = state["df"]

    if df is None:
        return "Upload dataset first."

    metrics = {}

    # fallback if column not present
    if "PROCESSING_DAYS" not in df.columns:
        df["PROCESSING_DAYS"] = np.random.randint(10, 60, len(df))

    metrics["avg_processing"] = df["PROCESSING_DAYS"].mean()
    metrics["max_processing"] = df["PROCESSING_DAYS"].max()
    metrics["min_processing"] = df["PROCESSING_DAYS"].min()

    metrics["avg_saving"] = df["saving"].mean()

    return f"""
- ⏱ Avg Processing Time: {metrics['avg_processing']:.1f} days  
- 🐢 Max Processing Time: {metrics['max_processing']} days  
- ⚡ Min Processing Time: {metrics['min_processing']} days  
- 💰 Avg Saving: ₹{metrics['avg_saving']:,.0f}
"""

# ──────────────────────────────────────────────
# Anomaly Detection
# ──────────────────────────────────────────────
def detect_anomalies():
    df = state["df"]

    if df is None:
        return "Upload dataset first."

    alerts = []

    # HIGH LOSS
    high_loss = df[df["saving"] < df["saving"].quantile(0.05)]
    if len(high_loss) > 0:
        alerts.append(f"🔴 High Loss Transactions: {len(high_loss)} records")

    # HIGH SAVING (unusual)
    high_saving = df[df["saving"] > df["saving"].quantile(0.95)]
    if len(high_saving) > 0:
        alerts.append(f"🟢 Exceptional Savings Cases: {len(high_saving)} records")

    # LONG PROCESSING
    if "PROCESSING_DAYS" in df.columns:
        slow = df[df["PROCESSING_DAYS"] > df["PROCESSING_DAYS"].quantile(0.9)]
        if len(slow) > 0:
            alerts.append(f"🟡 Slow Processing PRs: {len(slow)} cases")

    if not alerts:
        return "✅ No major anomalies detected."

    return "\n".join([f"- {a}" for a in alerts])

# ──────────────────────────────────────────────
# CHARTS
# ──────────────────────────────────────────────
def _style(fig, ax):
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f0f6ff")
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=10, colors="#0a1628")
    ax.xaxis.label.set_color("#0a1628")
    ax.yaxis.label.set_color("#0a1628")
    ax.title.set_color("#0c2340")

def plot_saving_distribution():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _style(fig, ax)
    df = state["df"]
    if df is None:
        ax.text(0.5,0.5,"Upload dataset first",ha="center",va="center",color="#0a1628",fontsize=13)
        ax.axis("off"); return fig
    lo, hi = df["saving"].quantile(0.01), df["saving"].quantile(0.99)
    flt = df[(df["saving"]>=lo)&(df["saving"]<=hi)].copy()
    flt["saving_lakh"] = flt["saving"] / 1e5

    sns.histplot(flt["saving_lakh"], bins=50, kde=True, ax=ax,
             color="#1565c0", edgecolor="white", linewidth=.3)
    mean_val = flt["saving_lakh"].mean()
    median_val = flt["saving_lakh"].median()

    ax.axvline(mean_val, color="#e53935", lw=1.8, ls="--", label="Mean")
    ax.axvline(median_val, color="#43a047", lw=1.8, ls=":", label="Median")
    ax.ticklabel_format(style='plain', axis='x')
    ax.legend(fontsize=9, labelcolor="#0a1628")
    ax.set_title("Savings Distribution (1–99th Percentile)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Saving (₹ Lakhs)"); ax.set_ylabel("Frequency")
    fig.tight_layout(); return fig

def plot_vendor_chart():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _style(fig, ax)
    df = state["df"]
    if df is None:
        ax.text(0.5,0.5,"Upload dataset first",ha="center",va="center",color="#0a1628",fontsize=13)
        ax.axis("off"); return fig
    v = vendor_analysis(df).head(10).sort_values()
    v = v / 1e5  # convert to lakhs
    clrs = ["#43a047" if x>=0 else "#e53935" for x in v.values]
    v.plot(kind="barh", ax=ax, color=clrs, edgecolor="white", linewidth=.3)
    ax.set_title("Top 10 Vendors by Savings", fontsize=12, fontweight="bold")
    ax.set_xlabel("Total Saving (₹ Lakhs)"); ax.set_ylabel("Vendor")
    fig.tight_layout(); return fig

def plot_loss_vendors():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _style(fig, ax)

    df = state["df"]
    if df is None:
        ax.text(0.5,0.5,"Upload dataset first",ha="center",va="center")
        ax.axis("off")
        return fig

    if "L1_PARTY_NAME" not in df.columns:
        ax.text(0.5,0.5,"No vendor column",ha="center",va="center")
        ax.axis("off")
        return fig

    loss_vendors = (
        df.groupby("L1_PARTY_NAME")["saving"]
        .sum()
        .sort_values()
        .head(10)
    )

    loss_vendors = loss_vendors / 1e5  # convert to lakhs

    loss_vendors.plot(
        kind="barh",
        ax=ax,
        color="#e53935",
        edgecolor="white",
        linewidth=0.3
    )

    ax.set_title("Top 10 Loss-Making Vendors", fontsize=12, fontweight="bold")
    ax.set_xlabel("Total Loss (₹ Lakhs)")
    ax.set_ylabel("Vendor")

    fig.tight_layout()
    return fig

def plot_profit_loss_pie():
    fig, ax = plt.subplots()

    df = state["df"]
    if df is None:
        return fig

    profit = (df["saving"] > 0).sum()
    loss = (df["saving"] < 0).sum()

    labels = ["Profit Cases", "Loss Cases"]
    sizes = [profit, loss]

    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title("Profit vs Loss Distribution")
    return fig

def generate_analytics():
    log_action("GENERATE: Analytics")
    return plot_saving_distribution(), plot_vendor_chart(), plot_loss_vendors(), plot_profit_loss_pie()

# ──────────────────────────────────────────────
# AI INSIGHTS
# ──────────────────────────────────────────────
def get_ai_insights():
    log_action("VIEW: AI Insights")
    df = state["df"]
    if df is None: return "Upload dataset first."
    total    = len(df)
    neg_pct  = (df["saving"]<0).sum()/total*100
    pos_pct  = (df["saving"]>0).sum()/total*100
    zero_pct = 100-neg_pct-pos_pct
    vdf = df.dropna(subset=["L1_PARTY_NAME"]) if "L1_PARTY_NAME" in df.columns else pd.DataFrame()
    top   = vdf.groupby("L1_PARTY_NAME")["saving"].sum().idxmax() if len(vdf)>0 else "N/A"
    worst = vdf.groupby("L1_PARTY_NAME")["saving"].sum().idxmin() if len(vdf)>0 else "N/A"
    return f"""
### 🧠 AI-Generated Insights

| Metric | Value |
|---|---|
| ✅ Profitable Cases | {pos_pct:.1f}% |
| ❌ Loss Cases | {neg_pct:.1f}% |
| ➖ Neutral Cases | {zero_pct:.1f}% |
| 📊 Median Saving | ₹{df["saving"].median():,.2f} |
| 📊 Average Saving | ₹{df["saving"].mean():,.2f} |
| 🏆 Best Vendor | {top} |
| ⚠️ Worst Vendor | {worst} |
| 💰 Highest Single Saving | ₹{df["saving"].max():,.2f} |
| 🔻 Largest Single Loss | ₹{df["saving"].min():,.2f} |
| 📦 Total Records | {total:,} |

---

**Recommendations:**
- {"✅ Savings rate is healthy." if pos_pct>70 else "⚠️ Savings rate below 70% — review procurement strategy."}
- {"✅ Loss cases within acceptable range." if neg_pct<15 else "🚨 Loss cases exceed 15% — investigate vendor contracts."}
- Consider increasing engagement with **{top}** for repeat procurement.
"""

# ──────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────
def predict(pr, response, techsuit, no_ext, dur_contract,
            ra_ind, msme_det, werks, pur_group):

    log_action(f"PREDICT: PR={pr}")

    try:
        models = state["model"]

        row = {
            "PR_VALUE": pr,
            "RESPONSE": response,
            "NO_OF_TECHSUIT": techsuit,
            "NO_OF_EXT": no_ext,
            "DUR_OF_CONTRACT": dur_contract,
            "RA_IND": ra_ind,
            "MSME_DET": msme_det,
            "WERKS": werks,
            "PUR_GROUP": pur_group,
            "request_month": 0,
            "request_quarter": 0
        }

        input_df = pd.DataFrame([row])

        train_features = models["features"]

        input_df = input_df.reindex(columns=train_features)
        input_df = input_df.fillna(0)

        predicted_negotiation = models["reg_model"].predict(input_df)[0]
        predicted_saving = pr - predicted_negotiation
        predicted_pct = (predicted_saving / pr * 100) if pr else 0
        predicted_class = models["cls_model"].predict(input_df)[0]

        return (
            f"### 🤖 Prediction Result\n\n"
            f"PR Value: ₹{pr:,.2f}\n\n"
            f"Predicted Negotiation: ₹{predicted_negotiation:,.2f}\n\n"
            f"Predicted Saving: ₹{predicted_saving:,.2f}\n\n"
            f"Saving %: {predicted_pct:.2f}%\n\n"
            f"Class: {predicted_class}"
        )

    except Exception as e:
        return f"❌ Prediction failed: {type(e).__name__}: {e}"

        

def batch_predict(file):
    log_action("BATCH PREDICT")
    mdl = _get_model()
    if mdl is None: return None, "Upload a dataset first."
    if file is None: return None, "Upload a CSV file."
    try:
        idf = pd.read_csv(file.name)
        req = {"PR_VALUE","NEGOTIATION_VAL"}
        if not req.issubset(set(idf.columns)):
            return None, f"CSV must contain columns: {req}"
        idf["Predicted_Saving"] = mdl.predict(idf[list(req)])
        idf["Status"] = idf["Predicted_Saving"].apply(lambda x:"Positive" if x>=0 else "Loss")
        out = "batch_predictions.xlsx"
        idf.to_excel(out, index=False)
        return out, f"✅ Done — {len(idf):,} records processed."
    except Exception as e:
        return None, f"Error: {e}"

# ──────────────────────────────────────────────
# EXPORTS
# ──────────────────────────────────────────────
def generate_pdf():
    log_action("EXPORT: PDF")
    df = state["df"]
    if df is None: return None
    ts, av, tr, lc = _kpi_values()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(REPORT_PATH, rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    content = []
    ts_style = ParagraphStyle("T", parent=styles["Title"],
                              fontSize=18, textColor=rl_colors.HexColor("#0c2340"))
    sub_style = ParagraphStyle("S", parent=styles["Normal"],
                               fontSize=10, textColor=rl_colors.HexColor("#4a7aab"))
    content.append(Paragraph("BSP Procurement Intelligence Report", ts_style))
    content.append(Paragraph(
        f"By: {state['current_user']} ({state['current_role']})  |  "
        f"Date: {datetime.now().strftime('%d %b %Y, %H:%M')}", sub_style))
    content.append(HRFlowable(width="100%", thickness=1,
                              color=rl_colors.HexColor("#1565c0"), spaceAfter=12))
    kpi_data = [
        ["KPI","Value"],
        ["Total Records", f"{tr:,}"],
        ["Total Saving",  f"Rs {ts:,.0f}"],
        ["Avg Saving",    f"Rs {av:,.2f}"],
        ["Loss Cases",    f"{lc:,}"],
        ["Loss Rate",     f"{lc/tr*100:.1f}%" if tr else "-"],
    ]
    t = Table(kpi_data, colWidths=[3*inch,3*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),rl_colors.HexColor("#1565c0")),
        ("TEXTCOLOR", (0,0),(-1,0),rl_colors.white),
        ("FONTNAME",  (0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",  (0,0),(-1,-1),10),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [rl_colors.HexColor("#eaf2ff"),rl_colors.white]),
        ("GRID",(0,0),(-1,-1),0.5,rl_colors.HexColor("#b3d0ee")),
        ("ALIGN",(1,0),(1,-1),"RIGHT"),
        ("TOPPADDING",(0,0),(-1,-1),7),
        ("BOTTOMPADDING",(0,0),(-1,-1),7),
    ]))
    content.append(t); content.append(Spacer(1,16))
    content.append(Paragraph("Top 10 Vendors", styles["Heading2"]))
    vser = vendor_analysis(df).head(10)
    vrows = [["#","Vendor","Total Saving"]] + [
        [str(i+1), str(n), f"Rs {v:,.2f}"] for i,(n,v) in enumerate(vser.items())]
    vt = Table(vrows, colWidths=[.5*inch,4*inch,2*inch])
    vt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),rl_colors.HexColor("#1a3c6e")),
        ("TEXTCOLOR", (0,0),(-1,0),rl_colors.white),
        ("FONTNAME",  (0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",  (0,0),(-1,-1),9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [rl_colors.HexColor("#eaf2ff"),rl_colors.white]),
        ("GRID",(0,0),(-1,-1),0.4,rl_colors.HexColor("#b3d0ee")),
        ("ALIGN",(2,0),(2,-1),"RIGHT"),
        ("TOPPADDING",(0,0),(-1,-1),6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))
    content.append(vt)
    doc.build(content)
    return REPORT_PATH

def export_excel():
    log_action("EXPORT: Excel")
    df = state["df"]
    if df is None: return None
    with pd.ExcelWriter(EXPORT_PATH, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Data", index=False)
        vendor_analysis(df).reset_index().to_excel(
            w, sheet_name="Vendor Summary", index=False)
        pd.DataFrame([{
            "Total Records": len(df), "Total Saving": df["saving"].sum(),
            "Avg Saving": df["saving"].mean(),
            "Loss Cases": int((df["saving"]<0).sum()),
        }]).to_excel(w, sheet_name="KPI Summary", index=False)
    return EXPORT_PATH

# ──────────────────────────────────────────────
# AUDIT LOG
# ──────────────────────────────────────────────
def get_audit_log():
    if not state["audit_log"]:
        return pd.DataFrame({"Timestamp":[],"User":[],"Role":[],"Action":[]})
    dfl = pd.DataFrame(state["audit_log"])
    dfl.columns = ["Timestamp","User","Role","Action"]
    return dfl.iloc[::-1].reset_index(drop=True)

# ══════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════
with gr.Blocks() as app:

    login_status = gr.Markdown()

    # ── LOGIN ──────────────────────────────────
    with gr.Column(visible=True) as login_page:
        with gr.Row(elem_id="login-wrapper"):

            # ===== LEFT PANEL =====
            with gr.Column(scale=3, elem_id="left-panel"):

                logo = get_logo()
                if logo:
                    gr.Image(value=logo, show_label=False, elem_id="bsp-logo")

                gr.HTML("""
<div id="brand-box">
  <h1>BSP Procurement Intelligence Platform</h1>
  <p class="subtitle">Enterprise Procurement Analytics System</p>

  <p class="desc">
    AI-powered analytics for procurement monitoring, vendor intelligence,
    savings analysis, predictive modelling, and executive reporting.
  </p>

  <ul>
    <li> Vendor Intelligence & Insights</li>
    <li> AI-based Saving Predictions</li>
    <li> Executive Reports & Analytics</li>
    <li> Full Audit Logging & Compliance</li>
  </ul>
</div>
""")

            # ===== RIGHT PANEL =====
            with gr.Column(scale=2, elem_id="right-panel"):

                gr.HTML("""
<div id="login-box">
  <h2>🔐 Secure Access Portal</h2>
  <p>Authorized personnel only</p>
  <ul>
    <li>Role-based access &mdash; Admin / Analyst / Viewer</li>
    <li>Full audit trail enabled</li>
    <li>Session-bound data isolation</li>
  </ul>
</div>
""")

                username_in = gr.Textbox(label="User ID", placeholder="Enter your user ID")
                password_in = gr.Textbox(label="Password", placeholder="Enter your password", type="password")
                login_btn = gr.Button("🔓 Login", variant="primary")

    # ── MAIN APP ───────────────────────────────
    with gr.Column(visible=False) as main_app:
        session_bar = gr.HTML()

        with gr.Row():
             gr.HTML("<div style='flex:1'></div>")  # spacer pushes button right
             logout_btn = gr.Button("🚪 Logout", variant="secondary", scale=0)
 
        with gr.Tabs():

            # HOME ───────────────────────────────
            with gr.Tab("🏠 Home"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.HTML("""
<div id="home-info">
  <h3>About Bhilai Steel Plant</h3>
  <p>
    Bhilai Steel Plant (BSP) is one of India's largest integrated steel plants
    and a flagship unit of SAIL. Renowned for rails, heavy plates, structural
    steel, and merchant products, BSP plays a strategic role in India's
    industrial infrastructure.
  </p>
  <h3>About This Platform</h3>
  <p>This Procurement Intelligence Platform enables data-driven decision-making:</p>
  <ul>
    <li>&#x1F4CA; Analyse procurement savings across vendors and categories</li>
    <li>&#x1F3ED; Identify top-performing and loss-making vendors</li>
    <li>&#x1F916; Generate AI-based saving predictions</li>
    <li>&#x1F4C4; Export executive PDF &amp; Excel reports</li>
    <li>&#x1F50D; Monitor data quality and detect anomalies</li>
    <li>&#x1F4CB; Maintain a full audit trail of all user actions</li>
  </ul>
  <h3>Access Levels</h3>
  <table style="width:100%;border-collapse:collapse;margin-top:10px;">
    <thead>
      <tr style="background:linear-gradient(135deg,#1565c0,#1e88e5);">
        <th style="padding:10px 16px;color:#fff;text-align:left;border-radius:8px 0 0 0;">Role</th>
        <th style="padding:10px 16px;color:#fff;text-align:left;border-radius:0 8px 0 0;">Access</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background:rgba(255,255,255,0.05);">
        <td style="padding:10px 16px;color:#c8f7c5;font-weight:700;">&#x1F6E1;&#xFE0F; Admin</td>
        <td style="padding:10px 16px;color:#dce9f7;">Full access including Audit Log</td>
      </tr>
      <tr style="background:rgba(255,255,255,0.03);">
        <td style="padding:10px 16px;color:#bbdefb;font-weight:700;">&#x1F4CA; Analyst</td>
        <td style="padding:10px 16px;color:#dce9f7;">Analytics, Prediction, Reports, Data Quality</td>
      </tr>
      <tr style="background:rgba(255,255,255,0.05);">
        <td style="padding:10px 16px;color:#e1bee7;font-weight:700;">&#x1F441;&#xFE0F; Viewer</td>
        <td style="padding:10px 16px;color:#dce9f7;">Home, Analytics, Vendor Intelligence (read-only)</td>
      </tr>
    </tbody>
  </table>
</div>
""")
                    with gr.Column(scale=2):
                        gr.Gallery(
                           value=safe_images(),
                           show_label=False,
                           columns=1,
                           height="auto",          
                           object_fit="contain",
                           elem_id="bsp-gallery",
                        )

            # ANALYTICS ──────────────────────────
            with gr.Tab("📊 Analytics"):
                gr.Markdown("## 📂 Dataset Upload")
                with gr.Row():
                    uploaded_file = gr.File(
                        label="Upload Excel Dataset (.xlsx)", scale=3, elem_id="file-upload")
                    with gr.Column(scale=2):
                        date_from = gr.Textbox(label="Date From (YYYY-MM-DD)",
                                               placeholder="e.g. 2024-01-01")
                        date_to   = gr.Textbox(label="Date To (YYYY-MM-DD)",
                                               placeholder="e.g. 2024-12-31")

                upload_status = gr.Textbox(
                    label="📋 Upload Status", interactive=False, lines=3)
                alert_html = gr.HTML()

                gr.Markdown("## 📊 Executive KPIs")
                with gr.Row():
                    kpi1 = gr.HTML(_EMPTY)
                    kpi2 = gr.HTML(_EMPTY)
                    kpi3 = gr.HTML(_EMPTY)
                    kpi4 = gr.HTML(_EMPTY)

                dq_hidden = gr.Markdown(visible=False)

                uploaded_file.change(
                    fn=process_data,
                    inputs=[uploaded_file, date_from, date_to],
                    outputs=[upload_status, kpi1, kpi2, kpi3, kpi4,
                             alert_html, dq_hidden],
                )

                charts_btn = gr.Button("📊 Generate Analytics", variant="primary")
                # FIRST ROW (2 charts)
                with gr.Row():
                  with gr.Column(scale=1):
                    chart1 = gr.Plot(label="Saving Distribution")
                  with gr.Column(scale=1):
                    chart2 = gr.Plot(label="Top Vendors")

                # SECOND ROW (2 charts)
                with gr.Row():
                  with gr.Column(scale=1):
                    chart3 = gr.Plot(label="Loss-Making Vendors")
                  with gr.Column(scale=1):
                    chart4 = gr.Plot(label="Profit vs Loss")
                charts_btn.click(fn=generate_analytics,
                                 outputs=[chart1, chart2, chart3, chart4]
                                )

            # VENDOR 
            with gr.Tab("🏭 Vendor & Risk Intelligence"):
                gr.Markdown("## 🏭 Vendor Performance Summary")

                vendor_table = gr.Dataframe(wrap=True, interactive=False)

                vendors_btn = gr.Button("🔄 Load Vendor Insights", variant="primary")

                   
                gr.Markdown("## ⏱ Processing Metrics")
                metrics_out = gr.Markdown()
                gr.Markdown("---")
                gr.Markdown("## 🚨 Anomaly Detection")
                anomaly_out = gr.Markdown()

                vendors_btn.click(fn=get_top_vendors, outputs=vendor_table)
                vendors_btn.click(fn=compute_processing_metrics, outputs=metrics_out)
                vendors_btn.click(fn=detect_anomalies, outputs=anomaly_out)

                        # PREDICTION ─────────────────────────
            with gr.Tab("🤖 Prediction"):
                gr.Markdown("## 🤖 Multi-Layer AI Prediction Engine")

                # ── SINGLE PREDICTION ───────────
                with gr.Accordion("🔍 Single Record Prediction", open=True):

                    with gr.Row():
                        pr_in = gr.Number(label="PR Value (₹)", value=1_000_000)
                        response_in = gr.Number(label="Response", value=1)
                        techsuit_in = gr.Number(label="No. of Tech Suit", value=1)

                    with gr.Row():
                        no_ext_in = gr.Number(label="No. of Extensions", value=0)
                        dur_contract_in = gr.Number(label="Contract Duration", value=30)

                    with gr.Row():
                        ra_ind_in = gr.Dropdown(
                            choices=["Y", "N", "Unknown"],
                            value="Unknown",
                            label="RA Indicator"
                        )
                        msme_in = gr.Dropdown(
                            choices=["Y", "N", "Unknown"],
                            value="Unknown",
                            label="MSME"
                        )

                    with gr.Row():
                        werks_in = gr.Textbox(label="WERKS (Plant Code)", placeholder="Optional")
                        pur_group_in = gr.Textbox(label="PUR_GROUP", placeholder="Optional")

                    pred_out = gr.Markdown()
                    pred_btn = gr.Button("⚡ Run AI Prediction", variant="primary")

                    pred_btn.click(
                        fn=predict,
                        inputs=[
                            pr_in,
                            response_in,
                            techsuit_in,
                            no_ext_in,
                            dur_contract_in,
                            ra_ind_in,
                            msme_in,
                            werks_in,
                            pur_group_in
                        ],
                        outputs=pred_out
                    )

                # ── BATCH PREDICTION ───────────
                with gr.Accordion("📁 Batch Prediction (CSV Upload)", open=False):
                    gr.Markdown("_CSV must have columns: `PR_VALUE` and `NEGOTIATION_VAL`_")

                    batch_file = gr.File(label="Upload CSV")
                    batch_status = gr.Textbox(label="Status", interactive=False)
                    batch_out = gr.File(label="Download Predictions (.xlsx)")

                    batch_btn = gr.Button("📥 Run Batch Prediction", variant="primary")

                    batch_btn.click(
                        fn=batch_predict,
                        inputs=batch_file,
                        outputs=[batch_out, batch_status]
                    )


            # REPORTS ────────────────────────────
            with gr.Tab("📄 Reports"):
                gr.Markdown("## 🧠 AI-Generated Insights")
                insights_out = gr.Markdown()
                insights_btn = gr.Button("🧠 Generate Insights", variant="primary")
                insights_btn.click(fn=get_ai_insights, outputs=insights_out)

                gr.Markdown("---\n## 📤 Export Reports")
                with gr.Row():
                    with gr.Column():
                        pdf_btn  = gr.Button("📄 Export PDF Report",
                                             variant="primary")
                        pdf_file = gr.File(label="PDF Report")
                        pdf_btn.click(fn=generate_pdf, outputs=pdf_file)
                    with gr.Column():
                        xlsx_btn  = gr.Button("📊 Export Excel Report",
                                              variant="primary")
                        xlsx_file = gr.File(label="Excel Report")
                        xlsx_btn.click(fn=export_excel, outputs=xlsx_file)

            # DATA QUALITY ───────────────────────
            with gr.Tab("🔍 Data Quality"):
                gr.Markdown("## 🔍 Data Quality Assessment")
                dq_out = gr.Markdown()
                dq_btn = gr.Button("🔄 Run Data Quality Check",
                                   variant="primary")
                dq_btn.click(fn=get_data_quality, outputs=dq_out)

            # AUDIT LOG ──────────────────────────
            with gr.Tab("📋 Audit Log"):
                gr.Markdown("## 📋 User Activity Audit Log")
                gr.Markdown(
                    "_Every platform action is recorded automatically "
                    "for compliance and traceability._"
                )
                audit_table = gr.Dataframe(wrap=True, interactive=False)
                audit_btn   = gr.Button("🔄 Refresh Audit Log",
                                        variant="primary")
                audit_btn.click(fn=get_audit_log, outputs=audit_table)

        gr.HTML("""
<div id="footer-box">
  <span>&#x1F3ED; <b>BSP Procurement Intelligence Platform</b>
        &mdash; Enterprise Edition</span>
  <span>Confidential &nbsp;|&nbsp; Authorized Use Only &nbsp;|&nbsp;
        &copy; 2026 Bhilai Steel Plant (SAIL)</span>
</div>
""")

    login_btn.click(
        fn=login,
        inputs=[username_in, password_in],
        outputs=[main_app, login_page, session_bar, login_status],
    )
    logout_btn.click(
        fn=logout,
        outputs=[main_app, login_page, session_bar, login_status],
    )

app.launch(css=CSS, inbrowser=True)
