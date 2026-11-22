"""
Phishing Email Detection System - Streamlit Web Application.

A professional cybersecurity dashboard featuring:
- Home Dashboard (System overview, architecture, key metrics)
- Email Scanner (Live inference, confidence score, explainable indicators)
- URL Security Analysis (Static heuristic inspection without network access)
- Model Performance (Comparison of Logistic Regression, Naive Bayes, Random Forest)
- Dataset Explorer (Interactive data preview, class & category distributions)
- Cybersecurity Awareness (Threat intelligence, red flags, safe practices)
"""

import html
import json
import os
import sys
import tempfile
from typing import Any, Dict, List

# Ensure a writable matplotlib cache directory
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Setup import path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.feature_extraction import (
    analyze_single_url,
    extract_urls,
    get_email_feature_breakdown,
)
from src.prediction import PhishingPredictor
from src.preprocessing import clean_html, combine_subject_and_body, validate_email_input

# -------------------------------------------------------------
# Page Configuration & Custom CSS Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="PhishGuard AI | Phishing Email Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for high-end professional cybersecurity theme
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Heading Visibility and High-Contrast Typography */
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6,
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown h5,
    .stMarkdown h6 {
        color: #0B2545 !important;
        -webkit-text-fill-color: #0B2545 !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        visibility: visible !important;
        letter-spacing: -0.3px;
    }
    
    h1, [data-testid="stMarkdownContainer"] h1 {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #0B2545 !important;
        -webkit-text-fill-color: #0B2545 !important;
    }
    
    h2, [data-testid="stMarkdownContainer"] h2 {
        font-size: 22px !important;
        font-weight: 800 !important;
        color: #0B2545 !important;
        -webkit-text-fill-color: #0B2545 !important;
    }
    
    h3, [data-testid="stMarkdownContainer"] h3 {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #0B2545 !important;
        -webkit-text-fill-color: #0B2545 !important;
        margin-top: 16px !important;
        margin-bottom: 8px !important;
    }
    
    h4, [data-testid="stMarkdownContainer"] h4 {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #134074 !important;
        -webkit-text-fill-color: #134074 !important;
        margin-top: 14px !important;
        margin-bottom: 6px !important;
    }
    
    h5, [data-testid="stMarkdownContainer"] h5 {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #0B2545 !important;
        -webkit-text-fill-color: #0B2545 !important;
    }
    
    /* General Text Body Contrast */
    p, span, li, [data-testid="stMarkdownContainer"] p {
        color: #1E293B;
        opacity: 1 !important;
    }
    
    /* Form Labels, Radio, Inputs */
    label, [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p, .stWidgetLabel p {
        color: #0B2545 !important;
        -webkit-text-fill-color: #0B2545 !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    
    /* Sidebar Styling & Contrast */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #0B2545 !important;
        -webkit-text-fill-color: #0B2545 !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #1E293B !important;
        opacity: 1 !important;
    }
    
    /* Top Header */
    .cyber-header {
        background: linear-gradient(135deg, #0B2545 0%, #134074 60%, #0066CC 100%);
        padding: 28px 32px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(11, 37, 69, 0.15);
    }
    .cyber-header h1, .cyber-header h1 span {
        color: white !important;
        -webkit-text-fill-color: white !important;
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .cyber-header p {
        color: #E2E8F0 !important;
        -webkit-text-fill-color: #E2E8F0 !important;
        font-size: 15px;
        margin: 6px 0 0 0;
    }
    
    /* Status Pills */
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    .status-pill-active {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .status-pill-info {
        background-color: #E0F2FE;
        color: #0369A1;
        border: 1px solid #BAE6FD;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(11, 37, 69, 0.08);
    }
    .metric-title {
        color: #64748B;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #0B2545;
        font-size: 26px;
        font-weight: 800;
        margin: 4px 0;
    }
    .metric-desc {
        color: #059669;
        font-size: 12px;
        font-weight: 600;
    }
    
    /* Verdict Banners */
    .verdict-phishing {
        background: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 24px;
        margin: 16px 0;
        color: #991B1B;
    }
    .verdict-phishing h2, .verdict-phishing h2 span {
        color: #991B1B !important;
        -webkit-text-fill-color: #991B1B !important;
        opacity: 1 !important;
    }
    .verdict-safe {
        background: #F0FDF4;
        border: 2px solid #22C55E;
        border-radius: 10px;
        padding: 24px;
        margin: 16px 0;
        color: #166534;
    }
    .verdict-safe h2, .verdict-safe h2 span {
        color: #166534 !important;
        -webkit-text-fill-color: #166534 !important;
        opacity: 1 !important;
    }
    
    /* Pill Tags */
    .tag-pill {
        display: inline-block;
        background: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: 600;
        margin: 3px;
    }
    .tag-pill-danger {
        background: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: 600;
        margin: 3px;
    }
    .tag-pill-warning {
        background: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: 600;
        margin: 3px;
    }
    
    /* White Card Wrapper */
    .white-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Cached Resources
# -------------------------------------------------------------
@st.cache_resource
def get_predictor():
    """Initializes and caches the prediction engine."""
    return PhishingPredictor()


@st.cache_data
def load_dataset():
    """Loads and caches the email dataset."""
    csv_path = os.path.join(CURRENT_DIR, "data", "emails.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_data
def load_metrics_and_metadata():
    """Loads evaluation metrics and training metadata."""
    meta_path = os.path.join(CURRENT_DIR, "models", "model_metadata.json")
    metrics_path = os.path.join(CURRENT_DIR, "outputs", "model_metrics.json")
    metadata = {}
    metrics = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    return metadata, metrics


predictor = get_predictor()
df_dataset = load_dataset()
metadata, model_metrics = load_metrics_and_metadata()


# -------------------------------------------------------------
# Sidebar Navigation & System Stats
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ PhishGuard AI")
    st.markdown("**Enterprise Email Security Intelligence**")
    st.markdown("---")

    nav_choice = st.radio(
        "Navigation Menu",
        [
            "🏠 Home Dashboard",
            "🔍 Email Scanner",
            "🌐 URL Security Analysis",
            "📊 Model Performance",
            "📁 Dataset Explorer",
            "🛡️ Cybersecurity Awareness",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### ⚡ System Operational Status")
    
    best_model_name = metadata.get("best_model", "Multinomial Naive Bayes")
    best_f1 = metadata.get("best_metrics", {}).get("f1_score", 0.9846)
    best_acc = metadata.get("best_metrics", {}).get("accuracy", 0.9848)

    st.markdown(f"**Champion Model:** `{best_model_name}`")
    st.markdown(f"**Test Accuracy:** `{best_acc * 100:.2f}%`")
    st.markdown(f"**Phishing F1-Score:** `{best_f1 * 100:.2f}%`")
    st.markdown(f"**Corpus Size:** `{len(df_dataset):,} emails`")

    st.markdown("---")
    st.info(
        "🔒 **Privacy & Safety Notice**\n\n"
        "Emails and URLs are analyzed strictly in-memory using static heuristics and machine learning. "
        "No network requests are ever made to submitted URLs."
    )


# -------------------------------------------------------------
# Top Banner
# -------------------------------------------------------------
st.markdown("""
<div class="cyber-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1>🛡️ PhishGuard AI - Email Threat Detection System</h1>
            <p>Intelligent cybersecurity platform combining Natural Language Processing, URL Heuristics, and Machine Learning.</p>
        </div>
        <div style="margin-top: 10px;">
            <span class="status-pill status-pill-active">● ENGINE OPERATIONAL</span>
            <span class="status-pill status-pill-info">HYBRID ML + HEURISTICS</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================
# SECTION A: HOME DASHBOARD
# =============================================================
if nav_choice == "🏠 Home Dashboard":
    st.markdown("### 📊 Security Operations Overview")
    st.markdown(
        "PhishGuard AI analyzes incoming email communications to defend organizations against social engineering, "
        "credential harvesting, business email compromise (BEC), and malicious links."
    )

    # 4 Key Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Model Accuracy</div>
            <div class="metric-value">{best_acc * 100:.2f}%</div>
            <div class="metric-desc">✓ Verified Test Split</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Phishing F1-Score</div>
            <div class="metric-value">{best_f1 * 100:.2f}%</div>
            <div class="metric-desc">✓ Optimal Precision/Recall</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Active Algorithm</div>
            <div class="metric-value" style="font-size: 20px;">{best_model_name}</div>
            <div class="metric-desc">✓ Top Benchmark Winner</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Curated Corpus</div>
            <div class="metric-value">{len(df_dataset)}</div>
            <div class="metric-desc">✓ Balanced Ground Truth</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Pipeline Architecture Card
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("""
        <div class="white-card">
            <h4 style="color: #0B2545; margin-top: 0;">🛡️ End-to-End Threat Detection Pipeline</h4>
            <p style="color: #475569; font-size: 14px;">
                Unlike basic keyword scanners, PhishGuard AI leverages a multi-stage machine learning architecture
                that simultaneously evaluates textual syntax, psychological manipulation patterns, and link indicators:
            </p>
            <div style="display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 180px; background: #F1F5F9; padding: 14px; border-radius: 8px;">
                    <strong style="color: #0B2545;">1. Preprocessing</strong><br>
                    <span style="font-size: 12px; color: #64748B;">HTML cleaning, entity decoding, anchor extraction, whitespace normalization.</span>
                </div>
                <div style="flex: 1; min-width: 180px; background: #F1F5F9; padding: 14px; border-radius: 8px;">
                    <strong style="color: #0B2545;">2. Feature Union</strong><br>
                    <span style="font-size: 12px; color: #64748B;">TF-IDF (unigrams & bigrams) coupled with 15 engineered URL & linguistic features.</span>
                </div>
                <div style="flex: 1; min-width: 180px; background: #F1F5F9; padding: 14px; border-radius: 8px;">
                    <strong style="color: #0B2545;">3. Classification</strong><br>
                    <span style="font-size: 12px; color: #64748B;">Calibrated probability estimation and explainable threat scoring.</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="white-card">
            <h4 style="color: #0B2545; margin-top: 0;">⚡ Quick Threat Scanning</h4>
            <p style="color: #475569; font-size: 14px;">
                Ready to analyze an email? Jump directly to the scanner to inspect live communications or test curated attack samples.
            </p>
            <ul style="color: #475569; font-size: 13px; line-height: 1.8;">
                <li>Credential harvesting detection</li>
                <li>Financial scam & invoice spoofing detection</li>
                <li>Static URL security & deceptive TLD analysis</li>
                <li>Linguistic urgency & call-to-action scoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Feature Highlights Grid
    st.markdown("#### 🎯 Key Security Capabilities")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="white-card">
            <h5 style="color: #0B2545;">🌐 Static URL Defense</h5>
            <p style="font-size: 13px; color: #475569;">
                Extracts and evaluates embedded links for IP-based hosts, dangerous top-level domains (.xyz, .top, .ru),
                excessive subdomains, and credential trickery without contacting untrusted servers.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="white-card">
            <h5 style="color: #0B2545;">🧠 Psychological Cues</h5>
            <p style="font-size: 13px; color: #475569;">
                Detects social engineering triggers including artificial deadlines ("within 24 hours"),
                extortion language, urgent financial wire demands, and high-pressure casing.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="white-card">
            <h5 style="color: #0B2545;">📈 Transparent Explainability</h5>
            <p style="font-size: 13px; color: #475569;">
                Provides clear reasoning for every verdict, highlighting specific triggers,
                red flag counts, and actionable guidance for end users and security analysts.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================
# SECTION B: EMAIL SCANNER
# =============================================================
elif nav_choice == "🔍 Email Scanner":
    st.markdown("### 🔍 Live Email Security Scanner")
    st.markdown("Paste an email subject and body below, or select a pre-loaded sample for instant testing.")

    # Preset Sample Emails
    sample_options = {
        "-- Select a test sample --": {"subject": "", "body": ""},
        "🎣 Phishing: PayPal Unauthorized Transaction & Urgent Dispute": {
            "subject": "Important Notice: Unusual transaction detected on your PayPal account",
            "body": (
                "Dear Customer,\n"
                "We observed an unauthorized payment of $849.99 USD to Coinbase Global Inc on your PayPal account.\n"
                "If you did not authorize this charge, dispute the transaction and cancel the payment immediately:\n"
                "http://paypal-resolution-center.user-dispute.top/dispute.php?id=948293\n"
                "Please enter your login details, credit card number, and billing address to reverse this charge.\n"
                "PayPal Fraud Prevention Department"
            ),
        },
        "🎣 Phishing: Microsoft 365 Password Expiration Alert": {
            "subject": "URGENT: Microsoft 365 Password Expiration Notice",
            "body": (
                "Dear User,\n"
                "Your Microsoft 365 email account password will expire in 24 hours.\n"
                "Failure to renew your credentials will result in immediate suspension of your mailbox and cloud storage.\n"
                "Please update your password immediately by clicking the link below:\n"
                "http://microsoft365-verify-portal.account-security.xyz/login.php?user=verify\n"
                "Keep your credentials safe. Do not ignore this alert.\n"
                "IT Helpdesk Support Center"
            ),
        },
        "🎣 Phishing: Chase Online Banking Blocked (IP Host)": {
            "subject": "URGENT ACTION REQUIRED: Chase Online Banking Access Restricted",
            "body": (
                "Notice of Account Suspension\n"
                "We have detected irregular debit card transactions and potential identity theft on your Chase Checking account.\n"
                "For your safety, online banking access is temporarily blocked.\n"
                "To reinstate full banking privileges, please verify your Social Security Number, PIN, and account number:\n"
                "http://192.168.1.105/auth/chase-login.html\n"
                "Do not reply to this email. Complete verification online immediately.\n"
                "JPMorgan Chase & Co. Security Division"
            ),
        },
        "🛡️ Safe: Sprint 42 Planning & Backlog Grooming": {
            "subject": "Sprint 42 Planning & Backlog Grooming - Monday 10:00 AM",
            "body": (
                "Hi Team,\n"
                "Let's meet on Monday at 10:00 AM EST for our regular Sprint 42 planning session.\n"
                "We will review remaining Jira tickets, estimate upcoming user stories for the authentication service, and discuss velocity.\n"
                "Please review the sprint board prior to the call:\n"
                "https://jira.company.com/secure/RapidBoard.jspa?rapidView=142\n"
                "Google Meet Link: https://meet.google.com/abc-defg-hij\n"
                "Looking forward to speaking then!\n"
                "Best regards,\n"
                "Sarah Chen\n"
                "Scrum Master"
            ),
        },
        "🛡️ Safe: Amazon Order Shipment Confirmation": {
            "subject": "Your Amazon.com order #114-9821038-19238 has shipped!",
            "body": (
                "Hi David,\n"
                "Great news! Your package is on its way.\n"
                "Order Details:\n"
                "- Anker USB-C Hub 7-in-1 Adapter ($39.99)\n"
                "- Logitech MX Master 3S Wireless Mouse ($99.99)\n"
                "Total: $139.98\n"
                "Carrier: UPS\n"
                "Tracking Number: 1Z9999999999999999\n"
                "Estimated Delivery: Wednesday, November 25 by 8:00 PM.\n"
                "You can view your order history or make returns anytime at https://www.amazon.com/your-orders\n"
                "Thank you for shopping with Amazon!"
            ),
        },
        "🛡️ Safe: Annual Health Benefits Open Enrollment": {
            "subject": "Annual Health Benefits Open Enrollment Period (Nov 15 - Dec 1)",
            "body": (
                "Dear Employees,\n"
                "Our annual benefits open enrollment window is now open and runs through December 1.\n"
                "During this period, you may make changes to your medical, dental, vision, and flexible spending accounts.\n"
                "Please review the 2027 Benefits Guide attached on our intranet:\n"
                "https://intranet.company.internal/hr/benefits-guide-2027\n"
                "If you do not wish to make changes, your current elections will automatically roll over.\n"
                "Best regards,\n"
                "People & Culture Team"
            ),
        },
    }

    # Session state for inputs and scan results
    if "scanner_subject" not in st.session_state:
        st.session_state["scanner_subject"] = ""
    if "scanner_body" not in st.session_state:
        st.session_state["scanner_body"] = ""
    if "sample_dropdown" not in st.session_state:
        st.session_state["sample_dropdown"] = "-- Select a test sample --"
    if "scan_result" not in st.session_state:
        st.session_state["scan_result"] = None
    if "scanned_subject" not in st.session_state:
        st.session_state["scanned_subject"] = ""
    if "scanned_body" not in st.session_state:
        st.session_state["scanned_body"] = ""

    def on_sample_change():
        chosen = st.session_state.get("sample_dropdown")
        # Selecting or changing sample immediately clears any previous scan results
        st.session_state["scan_result"] = None
        st.session_state["scanned_subject"] = ""
        st.session_state["scanned_body"] = ""
        if chosen and chosen != "-- Select a test sample --":
            sample_data = sample_options[chosen]
            st.session_state["scanner_subject"] = sample_data["subject"]
            st.session_state["scanner_body"] = sample_data["body"]
        else:
            st.session_state["scanner_subject"] = ""
            st.session_state["scanner_body"] = ""

    def on_clear_reset():
        # Streamlit-safe reset callback executed before widgets are instantiated
        st.session_state["sample_dropdown"] = "-- Select a test sample --"
        st.session_state["scanner_subject"] = ""
        st.session_state["scanner_body"] = ""
        st.session_state["scan_result"] = None
        st.session_state["scanned_subject"] = ""
        st.session_state["scanned_body"] = ""

    # Sample selector
    col_sample, col_clear = st.columns([4, 1])
    with col_sample:
        selected_sample_key = st.selectbox(
            "Load Sample Email for Evaluation:",
            list(sample_options.keys()),
            key="sample_dropdown",
            on_change=on_sample_change,
        )

    with col_clear:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.button(
            "🗑️ Clear / Reset",
            use_container_width=True,
            on_click=on_clear_reset,
        )

    # Input form
    with st.form("email_scanner_form"):
        subject_input = st.text_input(
            "Email Subject Line:",
            value=st.session_state["scanner_subject"],
            placeholder="e.g. URGENT: Action required for your account",
        )
        body_input = st.text_area(
            "Email Body Content:",
            value=st.session_state["scanner_body"],
            height=200,
            placeholder="Paste raw email text or HTML content here...",
        )
        analyze_btn = st.form_submit_button("🔍 Analyze Email", use_container_width=True)

    if analyze_btn:
        st.session_state["scanner_subject"] = subject_input
        st.session_state["scanner_body"] = body_input

        # Validate and predict
        is_val, err_msg = validate_email_input(subject_input, body_input)
        if not is_val:
            st.error(f"⚠️ Validation Error: {err_msg}")
            st.session_state["scan_result"] = None
            st.session_state["scanned_subject"] = ""
            st.session_state["scanned_body"] = ""
        else:
            with st.spinner("Analyzing email semantics, linguistic patterns, and link heuristics..."):
                result = predictor.predict(subject_input, body_input)
            st.session_state["scan_result"] = result
            st.session_state["scanned_subject"] = subject_input
            st.session_state["scanned_body"] = body_input

    # Display prediction results ONLY when a scan has been run and matches current input
    if (
        st.session_state.get("scan_result") is not None
        and st.session_state.get("scanned_subject") == subject_input
        and st.session_state.get("scanned_body") == body_input
    ):
        result = st.session_state["scan_result"]

        if not result["success"]:
            st.error(f"Analysis Error: {result.get('error', 'Unknown error occurred.')}")
        else:
            is_phish = result["is_phishing"]
            confidence = result["confidence_score"] * 100
            risk_lvl = result["risk_level"]
            breakdown = result["breakdown"]
            ling = breakdown["linguistic"]
            urls = breakdown["urls"]

            # Result Banner
            if is_phish:
                st.markdown(f"""
                <div class="verdict-phishing">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <h2 style="margin: 0; color: #991B1B;">🚨 PHISHING DETECTED</h2>
                            <p style="margin: 6px 0 0 0; font-size: 16px; font-weight: 600; color: #B91C1C;">
                                Risk Assessment: <span style="background: #991B1B; color: white; padding: 2px 8px; border-radius: 4px;">{risk_lvl}</span>
                            </p>
                        </div>
                        <div style="text-align: right; margin-top: 8px;">
                            <div style="font-size: 28px; font-weight: 800; color: #991B1B;">{confidence:.1f}%</div>
                            <div style="font-size: 12px; color: #7F1D1D; font-weight: 600;">CONFIDENCE (ESTIMATED)</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-safe">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <h2 style="margin: 0; color: #166534;">🛡️ SAFE EMAIL DETECTED</h2>
                            <p style="margin: 6px 0 0 0; font-size: 16px; font-weight: 600; color: #15803D;">
                                Risk Assessment: <span style="background: #166534; color: white; padding: 2px 8px; border-radius: 4px;">{risk_lvl}</span>
                            </p>
                        </div>
                        <div style="text-align: right; margin-top: 8px;">
                            <div style="font-size: 28px; font-weight: 800; color: #166534;">{confidence:.1f}%</div>
                            <div style="font-size: 12px; color: #14532D; font-weight: 600;">CONFIDENCE (ESTIMATED)</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Diagnostic Indicator Cards
            st.markdown("#### 🔍 Threat Indicator Breakdown")
            k1, k2, k3, k4 = st.columns(4)

            with k1:
                urg_cnt = ling["urgency_count"]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Urgency Signals</div>
                    <div class="metric-value" style="color: {'#DC2626' if urg_cnt > 0 else '#0B2545'};">{urg_cnt}</div>
                    <div class="metric-desc" style="color: #64748B;">Psychological pressure cues</div>
                </div>
                """, unsafe_allow_html=True)

            with k2:
                cred_cnt = ling["credential_count"]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Credential Triggers</div>
                    <div class="metric-value" style="color: {'#DC2626' if cred_cnt > 0 else '#0B2545'};">{cred_cnt}</div>
                    <div class="metric-desc" style="color: #64748B;">Password / Login keywords</div>
                </div>
                """, unsafe_allow_html=True)

            with k3:
                fin_cnt = ling["financial_count"]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Financial Keywords</div>
                    <div class="metric-value" style="color: {'#D97706' if fin_cnt > 0 else '#0B2545'};">{fin_cnt}</div>
                    <div class="metric-desc" style="color: #64748B;">Wire / Invoice / Bank terms</div>
                </div>
                """, unsafe_allow_html=True)

            with k4:
                url_cnt = breakdown["url_count"]
                hr_cnt = breakdown["high_risk_urls_count"]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Embedded Links</div>
                    <div class="metric-value" style="color: {'#DC2626' if hr_cnt > 0 else '#0B2545'};">{url_cnt}</div>
                    <div class="metric-desc" style="color: {'#DC2626' if hr_cnt > 0 else '#64748B'};">{hr_cnt} High-Risk Flagged</div>
                </div>
                """, unsafe_allow_html=True)

            # Reasons & Detected Factors
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            col_reasons, col_actions = st.columns([3, 2])

            with col_reasons:
                st.markdown("""<div class="white-card">""", unsafe_allow_html=True)
                st.markdown("##### 📌 Key Factors Influencing Prediction")
                for r in result["reasons"]:
                    st.markdown(f"- {r}")

                # Highlighted keyword badges
                all_found_kws = (
                    ling["urgency_found"] +
                    ling["credential_found"] +
                    ling["financial_found"] +
                    ling["cta_found"]
                )
                if all_found_kws:
                    st.markdown("<div style='margin-top: 12px;'><strong>Highlighted Keywords Detected:</strong></div>", unsafe_allow_html=True)
                    pills_html = " ".join([f"<span class='tag-pill-danger'>{kw}</span>" for kw in set(all_found_kws)])
                    st.markdown(pills_html, unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: #64748B; font-size: 13px; margin-top: 8px;'>No high-risk keywords detected.</p>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            with col_actions:
                st.markdown("""<div class="white-card">""", unsafe_allow_html=True)
                st.markdown("##### 🛡️ Recommended Security Actions")
                st.markdown(f"**Action:** {result['recommendation']}")

                if is_phish:
                    st.markdown("""
                    - 🚫 Do **not** click embedded links or download attachments.
                    - 🔐 Do **not** reply with credentials, passwords, or personal details.
                    - 📢 Forward this email as an attachment to your security operations team.
                    - 🗑️ Permanently delete after reporting.
                    """)
                else:
                    st.markdown("""
                    - ✓ Normal business communication detected.
                    - 💡 Always verify unexpected payment requests or sender changes out-of-band.
                    """)
                st.markdown("</div>", unsafe_allow_html=True)

            # Static URL Quick Preview (if any)
            if urls:
                st.markdown("#### 🔗 Embedded Link Heuristic Overview")
                for idx, u_info in enumerate(urls, 1):
                    with st.expander(f"Link #{idx}: {u_info['url']} - ({u_info['risk_level']})", expanded=is_phish):
                        st.markdown(f"**Hostname:** `{u_info['hostname']}` | **Protocol:** `{'HTTPS' if u_info['is_https'] else 'HTTP (Unencrypted)'}` | **Length:** `{u_info['url_length']} chars`")
                        for f in u_info["findings"]:
                            st.markdown(f"- {f}")

            # ML Disclaimer
            st.caption(f"ℹ️ **Security Disclaimer:** {result['disclaimer']}")


# =============================================================
# SECTION C: URL SECURITY ANALYSIS
# =============================================================
elif nav_choice == "🌐 URL Security Analysis":
    st.markdown("### 🌐 Static URL Security Heuristic Engine")
    st.markdown(
        "Analyze URLs statically for malicious indicators including numeric IP hosts, suspicious top-level domains, "
        "excessive subdomains, obfuscation tokens, and deceptive path keywords."
    )

    st.warning(
        "🛡️ **Zero-Contact Guarantee**: This analysis runs strictly offline using regular expressions and parsing rules. "
        "It **never** makes network requests or contacts remote servers."
    )

    url_input_text = st.text_area(
        "Enter URL(s) to inspect (one per line or paste email text containing links):",
        height=140,
        placeholder="http://paypal-verification.security-portal.xyz/verify\nhttp://192.168.1.105/auth/login.html\nhttps://github.com",
    )

    if st.button("🔎 Inspect URLs", use_container_width=True) or url_input_text:
        extracted = extract_urls(url_input_text)
        if not extracted:
            st.info("No valid URLs detected in the input. Please enter valid URL strings.")
        else:
            st.markdown(f"#### Identified {len(extracted)} Unique Link(s)")

            for idx, raw_url in enumerate(extracted, 1):
                analysis = analyze_single_url(raw_url)
                risk_lvl = analysis["risk_level"]

                # Card color based on risk
                if risk_lvl == "CRITICAL RISK":
                    border_color = "#EF4444"
                    badge_class = "tag-pill-danger"
                elif risk_lvl == "HIGH RISK":
                    border_color = "#F97316"
                    badge_class = "tag-pill-danger"
                elif risk_lvl == "MEDIUM RISK":
                    border_color = "#F59E0B"
                    badge_class = "tag-pill-warning"
                else:
                    border_color = "#22C55E"
                    badge_class = "status-pill-active"

                st.markdown(f"""
                <div style="background: white; border-left: 6px solid {border_color}; border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <span style="font-family: monospace; font-size: 14px; font-weight: 700; color: #0B2545;">{html.escape(raw_url)}</span>
                        <span class="{badge_class}">{risk_lvl} (Score: {analysis['risk_score']}/10)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Hostname", analysis["hostname"] if analysis["hostname"] else "None")
                with col2:
                    st.metric("Raw IP Address Host", "YES (Flagged)" if analysis["is_ip"] else "No")
                with col3:
                    st.metric("Suspicious TLD", "YES (Flagged)" if analysis["has_suspicious_tld"] else "Standard")
                with col4:
                    st.metric("Protocol", "HTTPS (Encrypted)" if analysis["is_https"] else "HTTP (Unencrypted)")

                st.markdown("**Detailed Heuristic Findings:**")
                for f in analysis["findings"]:
                    st.markdown(f"- {f}")

                st.markdown("---")


# =============================================================
# SECTION D: MODEL PERFORMANCE
# =============================================================
elif nav_choice == "📊 Model Performance":
    st.markdown("### 📊 Machine Learning Model Benchmark & Evaluation")
    st.markdown(
        "All candidate algorithms were evaluated using **duplicate-aware, group-based stratified splitting (StratifiedGroupKFold)**. "
        "Template fingerprints were normalized so that near-duplicate synthetic variations could not leak between training and testing sets."
    )

    if model_metrics:
        st.markdown(f"#### 🎯 Holdout Test Split ({metadata.get('test_records', 66)} emails - Zero Template Overlap)")
        # Comparison Table
        table_rows = []
        for name, m in model_metrics.items():
            is_best = " 🏆 (Champion)" if name == metadata.get("best_model") else ""
            table_rows.append({
                "Algorithm": name + is_best,
                "Accuracy": f"{m['accuracy'] * 100:.2f}%",
                "Precision (Phishing)": f"{m['precision'] * 100:.2f}%",
                "Recall (Phishing)": f"{m['recall'] * 100:.2f}%",
                "F1-Score (Phishing)": f"{m['f1_score'] * 100:.2f}%",
                "Macro F1": f"{m['macro_f1'] * 100:.2f}%",
                "ROC-AUC": f"{m['roc_auc'] * 100:.2f}%" if m["roc_auc"] is not None else "N/A",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        # 5-Fold Stratified Group CV if available
        cv_summary = metadata.get("cross_validation_5fold", {})
        if cv_summary:
            st.markdown("#### 🔄 5-Fold Stratified Group Cross-Validation Across All Template Clusters")
            cv_rows = []
            for name, cv in cv_summary.items():
                cv_rows.append({
                    "Algorithm": name,
                    "Mean Accuracy": f"{cv['mean_accuracy'] * 100:.2f}% (±{cv['std_accuracy'] * 100:.2f}%)",
                    "Mean Precision": f"{cv['mean_precision'] * 100:.2f}% (±{cv['std_precision'] * 100:.2f}%)",
                    "Mean Recall": f"{cv['mean_recall'] * 100:.2f}% (±{cv['std_recall'] * 100:.2f}%)",
                    "Mean F1-Score": f"{cv['mean_f1'] * 100:.2f}% (±{cv['std_f1'] * 100:.2f}%)",
                })
            st.dataframe(pd.DataFrame(cv_rows), use_container_width=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Visualizations (side-by-side)
        v1, v2 = st.columns(2)

        with v1:
            st.markdown("#### 📈 Algorithm Comparison Chart")
            comparison_chart_path = os.path.join(CURRENT_DIR, "outputs", "model_comparison.png")
            if os.path.exists(comparison_chart_path):
                st.image(comparison_chart_path, use_container_width=True)
            else:
                st.info("Comparison chart not found.")

        with v2:
            st.markdown(f"#### 🎯 Confusion Matrix: {best_model_name}")
            cm_chart_path = os.path.join(CURRENT_DIR, "outputs", "confusion_matrix.png")
            if os.path.exists(cm_chart_path):
                st.image(cm_chart_path, use_container_width=True)
            else:
                st.info("Confusion matrix chart not found.")

        # Classification Report Table for Champion Model
        st.markdown(f"#### 📋 Detailed Classification Report ({best_model_name})")
        best_report = model_metrics.get(best_model_name, {}).get("classification_report", {})
        if best_report:
            report_df = pd.DataFrame(best_report).transpose()
            st.dataframe(report_df.style.format(precision=4), use_container_width=True)

        # Selection rationale
        st.markdown("""
        <div class="white-card">
            <h5 style="color: #0B2545;">💡 Model Selection Rationale</h5>
            <p style="font-size: 14px; color: #475569;">
                In cybersecurity email filtering, the cost of a <strong>False Negative</strong> (a phishing email slipping into an employee inbox)
                is catastrophic, often leading to ransomware or credential theft. Conversely, a <strong>False Positive</strong>
                causes minor administrative inconvenience.
                The champion model achieved an optimal balance with 96.97% recall and 100.0% precision on the leak-free holdout split,
                yielding an F1-score of 98.46% (and 99.38% mean 5-fold CV F1) across unseen template groups.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No saved metrics found. Please execute `python train_model.py` to generate evaluation data.")


# =============================================================
# SECTION E: DATASET EXPLORER
# =============================================================
elif nav_choice == "📁 Dataset Explorer":
    st.markdown("### 📁 Dataset Explorer & Class Distribution")
    st.markdown(
        "Explore the internally curated and synthetically generated benchmark dataset (80 manual seed archetypes + 261 template variations). "
        "All samples are designed for controlled testing and zero-leakage evaluation."
    )

    if not df_dataset.empty:
        # High level stats
        d1, d2, d3 = st.columns(3)
        phish_total = int(sum(df_dataset["label"] == 1))
        safe_total = int(sum(df_dataset["label"] == 0))
        total_emails = len(df_dataset)

        with d1:
            st.metric("Total Emails in Benchmark", f"{total_emails:,}")
        with d2:
            st.metric("Phishing Emails", f"{phish_total} ({phish_total / total_emails * 100:.1f}%)")
        with d3:
            st.metric("Safe Emails", f"{safe_total} ({safe_total / total_emails * 100:.1f}%)")

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Distribution Visualizations
        chart1, chart2 = st.columns(2)

        with chart1:
            st.markdown("#### Class Balance Distribution")
            fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
            ax.pie(
                [safe_total, phish_total],
                labels=["SAFE (Legitimate)", "PHISHING (Malicious)"],
                autopct="%1.1f%%",
                startangle=140,
                colors=["#0284C7", "#DC2626"],
                wedgeprops={"edgecolor": "white", "linewidth": 2},
                textprops={"weight": "bold", "color": "#0B2545"},
            )
            ax.axis("equal")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with chart2:
            st.markdown("#### Email Category Breakdown")
            if "category" in df_dataset.columns:
                cat_counts = df_dataset["category"].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
                sns.barplot(
                    x=cat_counts.values,
                    y=cat_counts.index,
                    palette="Blues_r",
                    ax=ax,
                    edgecolor="#0B2545",
                    linewidth=0.8,
                )
                ax.set_xlabel("Count", fontsize=10, weight="bold", color="#0B2545")
                ax.set_title("Emails per Sub-Category", fontsize=12, weight="bold", color="#0B2545")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # Interactive Table Preview
        st.markdown("#### 🔍 Interactive Dataset Filter & Search")
        filter_col1, filter_col2 = st.columns([1, 2])
        with filter_col1:
            class_filter = st.selectbox("Filter by Label:", ["All Classes", "PHISHING", "SAFE"])
        with filter_col2:
            search_query = st.text_input("Search email subject or body keyword:", "")

        filtered_df = df_dataset.copy()
        if class_filter != "All Classes":
            filtered_df = filtered_df[filtered_df["label_text"] == class_filter]
        if search_query:
            query = search_query.lower()
            filtered_df = filtered_df[
                filtered_df["subject"].str.lower().str.contains(query, na=False) |
                filtered_df["body"].str.lower().str.contains(query, na=False)
            ]

        st.markdown(f"**Showing {len(filtered_df)} of {len(df_dataset)} records:**")
        display_cols = [c for c in ["subject", "category", "label_text", "source", "body"] if c in filtered_df.columns]
        st.dataframe(filtered_df[display_cols], height=350, use_container_width=True)
    else:
        st.warning("Dataset file `data/emails.csv` not found.")


# =============================================================
# SECTION F: CYBERSECURITY AWARENESS
# =============================================================
elif nav_choice == "🛡️ Cybersecurity Awareness":
    st.markdown("### 🛡️ Cybersecurity Threat Awareness & Defense Guide")
    st.markdown(
        "Social engineering and phishing represent over 80% of initial enterprise breaches. "
        "Understanding attack tactics empowers teams to recognize and neutralize threats."
    )

    # Phishing Types Accordion
    st.markdown("#### 1. Major Attack Vectors & Social Engineering Modalities")
    t1, t2 = st.columns(2)

    with t1:
        st.markdown("""
        <div class="white-card">
            <h5 style="color: #0B2545;">🎯 Spear Phishing</h5>
            <p style="font-size: 13px; color: #475569;">
                Highly tailored attacks targeting specific individuals or departments. Attackers conduct reconnaissance
                on LinkedIn or corporate portals to reference real projects, colleagues, or vendors.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="white-card">
            <h5 style="color: #0B2545;">👑 Whaling / Executive Impersonation</h5>
            <p style="font-size: 13px; color: #475569;">
                Impersonating high-ranking executives (CEO, CFO) requesting urgent wire transfers, gift cards, or confidential
                employee tax files (W-2s) under the guise of an emergency board meeting.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="white-card">
            <h5 style="color: #0B2545;">💼 Business Email Compromise (BEC)</h5>
            <p style="font-size: 13px; color: #475569;">
                Attackers compromise or spoof legitimate vendor email accounts to send fraudulent invoices with altered
                bank routing numbers, intercepting legitimate commercial transactions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("""
        <div class="white-card">
            <h5 style="color: #0B2545;">🔐 Credential Harvesting & Fake Login Portals</h5>
            <p style="font-size: 13px; color: #475569;">
                Cloned login pages mimicking Microsoft 365, Google Workspace, or banking portals designed to capture
                usernames, passwords, and 2FA session cookies.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="white-card">
            <h5 style="color: #0B2545;">📦 Delivery & Service Lures</h5>
            <p style="font-size: 13px; color: #475569;">
                Fake delivery failure notices (FedEx, UPS, DHL) requesting a nominal "customs clearance fee" to collect
                credit card numbers and personal identity records.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="white-card">
            <h5 style="color: #0B2545;">📱 Quishing (QR Code Phishing)</h5>
            <p style="font-size: 13px; color: #475569;">
                Embedding malicious QR codes in PDF attachments or emails to bypass corporate email gateways and trick users
                into opening phishing portals on unmonitored personal smartphones.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Red Flags Checklist
    st.markdown("#### 2. The 7 Critical Red Flags Checklist")
    st.markdown("""
    | # | Red Flag Indicator | What Attackers Do | How to Verify |
    | :--- | :--- | :--- | :--- |
    | **1** | **Artificial Urgency** | Creates panic ("24 hours before account deletion", "Immediate legal action") | Genuine corporate emergencies are handled via telephone or manager channels. |
    | **2** | **Mismatched Sender** | Display name says "PayPal" but email address is `@random-domain.top` | Always expand the full email header to check the actual sending domain. |
    | **3** | **Deceptive Links** | Anchor text says `chase.com` but link destination points to `192.168.x.x` or `.xyz` | Hover over links (without clicking) to inspect the destination domain. |
    | **4** | **Generic Greetings** | "Dear Customer" or "Dear Employee" instead of your actual name | Legitimate service providers address you by your account name. |
    | **5** | **Requests for Sensitive Data** | Asking for passwords, SSNs, PINs, or 2FA codes | Legitimate IT and banks **never** ask for your password via email. |
    | **6** | **Unsolicited Attachments** | Files named `Invoice_9812.iso`, `.exe`, `.vbs`, or password-protected ZIPs | Never open executable attachments from unknown senders. |
    | **7** | **Irregular Banking Details** | "Our bank account has changed due to an annual audit; wire funds here" | Always confirm bank detail updates with vendors over verified phone numbers. |
    """)

    # Defensive Habits
    st.markdown("#### 3. Recommended Defensive Email Hygiene")
    st.markdown("""
    <div class="white-card">
        <ul style="color: #334155; line-height: 1.9; font-size: 14px;">
            <li><strong>Enable Multi-Factor Authentication (MFA):</strong> Use hardware security keys (FIDO2/WebAuthn) or authenticator apps rather than SMS wherever possible.</li>
            <li><strong>Use Dedicated Password Managers:</strong> Password managers automatically detect domain mismatches and will refuse to auto-fill credentials on phishing clones.</li>
            <li><strong>Practice Out-of-Band Verification:</strong> If a colleague, executive, or vendor requests an unusual wire transfer, verify verbally over phone or in-person.</li>
            <li><strong>Report Suspicious Messages:</strong> Use your organization's "Report Phishing" button. Every report strengthens corporate email gateway filters.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# Footer
# -------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 13px;'>"
    "🛡️ <strong>PhishGuard AI</strong> — Phishing Email Detection & Threat Intelligence System | Built with Python, Scikit-learn & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
