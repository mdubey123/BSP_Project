import os
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from src.preprocessing import clean_data
from src.feature_engineering import create_features
from src.analysis import vendor_analysis
from src.model import train_model

# -----------------------------
# GLOBAL STATE
# -----------------------------
df = None
model = None

ASSETS_DIR = "assets"
BSP_IMAGES = [
    os.path.join(ASSETS_DIR, "bsp1.jpg"),
    os.path.join(ASSETS_DIR, "bsp2.jpg"),
    os.path.join(ASSETS_DIR, "bsp3.jpg"),
]
BSP_LOGO = os.path.join(ASSETS_DIR, "bsp_logo.png")

# -----------------------------
# CUSTOM CSS
# -----------------------------
CUSTOM_CSS = """
body, .gradio-container {
    background: linear-gradient(135deg, #08111f 0%, #10243f 45%, #1d4667 100%) !important;
    color: #f5f7fb !important;
    font-family: 'Segoe UI', sans-serif;
}
.gradio-container {
    max-width: 1450px !important;
}
#top-nav {
    background: rgba(8, 20, 36, 0.92);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 14px 22px;
    margin-bottom: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.20);
    animation: fadeSlide 0.8s ease-in-out;
}
#nav-title {
    font-size: 20px;
    font-weight: 700;
    color: white;
}
#nav-subtitle {
    font-size: 13px;
    color: #c9d9ea;
    margin-top: 2px;
}
#hero-box {
    background: linear-gradient(135deg, rgba(8,20,36,0.94), rgba(20,55,90,0.90));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 24px;
    padding: 28px;
    margin-bottom: 18px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.25);
    animation: fadeSlide 1s ease-in-out;
}
#login-banner {
    background: linear-gradient(135deg, rgba(13,27,42,0.95), rgba(27,57,92,0.92));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 24px;
    padding: 36px;
    min-height: 420px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.25);
    animation: fadeSlide 1s ease-in-out;
}
#login-banner-title {
    font-size: 40px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 12px;
}
#login-banner-subtitle {
    font-size: 18px;
    color: #dce9f7;
    margin-bottom: 16px;
}
#login-banner-text {
    font-size: 15px;
    line-height: 1.8;
    color: #eef4fb;
}
#login-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 24px;
    padding: 28px;
    min-height: 420px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.25);
    animation: popIn 1s ease;
}
#login-card-title {
    font-size: 30px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
}
#login-card-subtitle {
    font-size: 15px;
    color: #d9e7f5;
    margin-bottom: 18px;
}
.section-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
    margin-top: 10px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.18);
}
#footer-box {
    margin-top: 22px;
    background: rgba(8, 20, 36, 0.92);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 16px 22px;
    color: #dce7f3;
    font-size: 14px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.18);
}
button {
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}
button:hover {
    transform: scale(1.02);
}
h1, h2, h3 {
    color: #ffffff !important;
}
@keyframes fadeSlide {
    from {
        opacity: 0;
        transform: translateY(-18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
@keyframes popIn {
    from {
        opacity: 0;
        transform: scale(0.96);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}
"""

# -----------------------------
# HELPERS
# -----------------------------
def safe_load_image_list():
    return [img for img in BSP_IMAGES if os.path.exists(img)]

def get_logo():
    return BSP_LOGO if os.path.exists(BSP_LOGO) else None

def process_data(file):
    global df, model

    if file is None:
        return (
            "Please upload an Excel file first.",
            "—", "—", "—", "—"
        )

    all_sheets = pd.read_excel(file.name, sheet_name=None)
    combined_df = pd.concat(all_sheets.values(), ignore_index=True)

    processed_df = clean_data(combined_df)
    processed_df = create_features(processed_df)

    df = processed_df
    model = train_model(df)

    total_saving, avg_saving, total_records, loss_cases = get_kpis_values()

    return (
        f"✅ Dataset loaded successfully | Rows: {len(df)} | Columns: {len(df.columns)}",
        f"₹{total_saving:,.0f}",
        f"₹{avg_saving:,.2f}",
        f"{total_records:,}",
        f"{loss_cases:,}",
    )

def login(username, password):
    if username == "bsp_admin" and password == "BSP_2026_AI":
        return gr.update(visible=True), gr.update(visible=False), "✅ Login successful"
    return gr.update(visible=False), gr.update(visible=True), "❌ Invalid credentials"

def get_kpis_values():
    if df is None:
        return 0, 0, 0, 0

    total_saving = df["saving"].sum()
    avg_saving = df["saving"].mean()
    total_records = len(df)
    loss_cases = int((df["saving"] < 0).sum())

    return total_saving, avg_saving, total_records, loss_cases

def get_top_vendors():
    if df is None:
        return pd.DataFrame({"Message": ["Upload dataset first"]})
    return vendor_analysis(df).reset_index()

def plot_saving_distribution():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    if df is None:
        ax.text(0.5, 0.5, "Upload dataset first", ha="center", va="center")
        ax.axis("off")
        return fig

    lower = df["saving"].quantile(0.01)
    upper = df["saving"].quantile(0.99)
    df_filtered = df[(df["saving"] >= lower) & (df["saving"] <= upper)]

    sns.histplot(df_filtered["saving"], bins=50, kde=True, ax=ax)
    ax.set_title("Saving Distribution")
    ax.set_xlabel("Saving")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig

def plot_vendor_chart():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    if df is None:
        ax.text(0.5, 0.5, "Upload dataset first", ha="center", va="center")
        ax.axis("off")
        return fig

    v = vendor_analysis(df).head(10)
    v.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top Vendors by Savings")
    ax.set_xlabel("Saving")
    ax.set_ylabel("Vendor")
    fig.tight_layout()
    return fig

def get_ai_insights():
    if df is None:
        return "Upload dataset first."

    total = len(df)
    negative = int((df["saving"] < 0).sum())
    positive = int((df["saving"] > 0).sum())
    negative_pct = (negative / total) * 100 if total else 0
    positive_pct = (positive / total) * 100 if total else 0
    median_saving = df["saving"].median()
    avg_saving = df["saving"].mean()
    max_saving = df["saving"].max()
    min_saving = df["saving"].min()

    vendor_df = df.dropna(subset=["L1_PARTY_NAME"]).copy()
    vendor_df = vendor_df[vendor_df["L1_PARTY_NAME"].astype(str).str.strip() != ""]

    if len(vendor_df) > 0:
        top_vendor = vendor_df.groupby("L1_PARTY_NAME")["saving"].sum().idxmax()
    else:
        top_vendor = "Not available"

    return f"""
### AI Insights

- **Savings Cases:** {positive_pct:.2f}%
- **Loss Cases:** {negative_pct:.2f}%
- **Median Saving:** ₹{median_saving:,.2f}
- **Average Saving:** ₹{avg_saving:,.2f}
- **Top Vendor:** {top_vendor}
- **Highest Saving:** ₹{max_saving:,.2f}
- **Largest Loss:** ₹{min_saving:,.2f}
"""

def predict(pr, nego):
    if model is None:
        return "Upload dataset first."

    input_df = pd.DataFrame({
        "PR_VALUE": [pr],
        "NEGOTIATION_VAL": [nego]
    })

    pred = model.predict(input_df)[0]
    return f"💡 Predicted Saving: ₹{pred:,.2f}"

def get_homepage_info():
    return """
### About Bhilai Steel Plant

Bhilai Steel Plant is one of India’s major integrated steel plants and a flagship unit of SAIL. It is widely known for rail production, heavy plates, structural steel, and its strategic importance to Indian industry.

### About This Project

This Procurement Intelligence Platform is designed to support data-driven decision-making by:
- analyzing procurement savings
- identifying top-performing vendors
- detecting loss-making cases
- generating AI-based predictions
- exporting summary reports for review
"""

def generate_pdf():
    if df is None:
        return None

    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = []

    total_saving, avg_saving, total_records, loss_cases = get_kpis_values()

    content.append(Paragraph("BSP Procurement Intelligence Report", styles["Title"]))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"Total Records: {total_records:,}", styles["Normal"]))
    content.append(Paragraph(f"Total Saving: ₹{total_saving:,.0f}", styles["Normal"]))
    content.append(Paragraph(f"Average Saving: ₹{avg_saving:,.2f}", styles["Normal"]))
    content.append(Paragraph(f"Loss Cases: {loss_cases:,}", styles["Normal"]))
    content.append(Spacer(1, 12))
    content.append(Paragraph("Generated from the Procurement Intelligence Platform.", styles["Normal"]))

    doc.build(content)
    return "report.pdf"

# -----------------------------
# UI
# -----------------------------
with gr.Blocks() as app:
    login_status = gr.Markdown()

    with gr.Column(visible=True) as login_page:
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("""
<div id="login-banner">
  <div id="login-banner-title">🏭 Bhilai Steel Plant</div>
  <div id="login-banner-subtitle">Procurement Intelligence Platform</div>
  <div id="login-banner-text">
    Welcome to the AI-powered analytics system designed for procurement monitoring,
    vendor intelligence, savings analysis, predictive insights, and decision support.
    <br><br>
    This platform helps stakeholders review procurement outcomes, identify
    high-performing vendors, detect inefficient transactions, and generate concise reports
    for presentation and evaluation.
  </div>
</div>
""")
            with gr.Column(scale=2):
                logo = get_logo()
                if logo:
                    gr.Image(value=logo, label="BSP Logo", height=120)

                gr.Markdown("""
<div id="login-card">
  <div id="login-card-title">🔐 Secure Login</div>
  <div id="login-card-subtitle">
    Sign in to access analytics, prediction, vendor insights, and reporting tools.
  </div>
</div>
""")
                username = gr.Textbox(label="Login ID", placeholder="Enter your login ID")
                password = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
                login_btn = gr.Button("Login", variant="primary")

    with gr.Column(visible=False) as main_app:
        gr.Markdown("""
<div id="top-nav">
  <div id="nav-title">🏭 BSP Procurement Intelligence Platform</div>
  <div id="nav-subtitle">AI-Powered Analytics | Vendor Intelligence | Prediction | Reporting</div>
</div>
""")

        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("""
<div id="hero-box">
  <div id="hero-title">Enterprise Procurement Dashboard</div>
  <div id="hero-subtitle">Corporate decision-support system for Bhilai Steel Plant</div>
  <div id="hero-text">
  A structured platform for procurement analysis, savings monitoring, vendor evaluation,
  predictive modeling, and report generation.
  </div>
</div>
""")
            with gr.Column(scale=1):
                logo = get_logo()
                if logo:
                    gr.Image(value=logo, label="BSP Logo", height=120)

        with gr.Tabs():
            with gr.Tab("Home"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown('<div class="section-box">')
                        gr.Markdown(get_homepage_info())
                        gr.Markdown('</div>')
                    with gr.Column(scale=2):
                        gr.Gallery(
                            value=safe_load_image_list(),
                            label="Bhilai Steel Plant",
                            columns=1,
                            height=340,
                            object_fit="cover",
                            preview=True,
                        )

            with gr.Tab("Analytics"):
                gr.Markdown("## 📂 Dataset Upload")
                uploaded_file = gr.File(label="Upload Excel Dataset (.xlsx)")
                upload_status = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("## 📊 Executive Summary")
                with gr.Row():
                    kpi1 = gr.Textbox(label="Total Saving", interactive=False)
                    kpi2 = gr.Textbox(label="Average Saving", interactive=False)
                    kpi3 = gr.Textbox(label="Total Records", interactive=False)
                    kpi4 = gr.Textbox(label="Loss Cases", interactive=False)

                uploaded_file.change(
                    fn=process_data,
                    inputs=uploaded_file,
                    outputs=[upload_status, kpi1, kpi2, kpi3, kpi4]
                )

                with gr.Row():
                    chart1 = gr.Plot(label="Saving Distribution")
                    chart2 = gr.Plot(label="Top Vendors by Savings")

                charts_btn = gr.Button("Generate Charts")
                charts_btn.click(
                    fn=lambda: (plot_saving_distribution(), plot_vendor_chart()),
                    outputs=[chart1, chart2]
                )

            with gr.Tab("Vendor Intelligence"):
                gr.Markdown("## 🏭 Top Vendor Performance")
                vendor_table = gr.Dataframe()
                vendors_btn = gr.Button("Show Top Vendors")
                vendors_btn.click(fn=get_top_vendors, outputs=vendor_table)

            with gr.Tab("Prediction"):
                gr.Markdown("## 🤖 AI Prediction Engine")
                with gr.Row():
                    pr = gr.Number(label="PR Value", value=1000000)
                    nego = gr.Number(label="Negotiation Value", value=500000)

                pred_out = gr.Textbox(label="Prediction Output")
                pred_btn = gr.Button("Predict Saving", variant="primary")
                pred_btn.click(fn=predict, inputs=[pr, nego], outputs=pred_out)

            with gr.Tab("Reports"):
                gr.Markdown("## 🧠 AI Insights")
                insights_out = gr.Markdown()
                insights_btn = gr.Button("Generate Insights")
                insights_btn.click(fn=get_ai_insights, outputs=insights_out)

                gr.Markdown("## 📄 Export PDF Report")
                pdf_btn = gr.Button("Generate PDF Report")
                pdf_file = gr.File(label="Download Report")
                pdf_btn.click(fn=generate_pdf, outputs=pdf_file)

        gr.Markdown("""
<div id="footer-box">
  <b>BSP Procurement Intelligence Platform</b><br>
  Developed as an AI-powered procurement analytics and decision-support solution for academic demonstration and industrial presentation.
</div>
""")

    login_btn.click(
        fn=login,
        inputs=[username, password],
        outputs=[main_app, login_page, login_status]
    )

app.launch(css=CUSTOM_CSS)