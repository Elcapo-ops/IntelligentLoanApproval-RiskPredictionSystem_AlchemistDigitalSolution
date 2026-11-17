"""Shared visual theme and presentation helpers."""

import streamlit as st

from prediction.scope import HISTORICAL_LOAN_STATUS_DISCLOSURE


def apply_theme() -> None:
    """Apply the application palette and shared layout styles."""
    st.markdown(
        """
        <style>
        :root {
            --navy: #10243a;
            --blue: #2874b2;
            --blue-soft: #eaf3fb;
            --green: #23845b;
            --red: #b84d55;
            --ink: #1d2c3b;
            --muted: #647487;
            --surface: #ffffff;
            --canvas: #f3f6f9;
            --border: #dce4ec;
        }
        .stApp { background: var(--canvas); color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(243, 246, 249, 0.9); }
        [data-testid="stSidebar"] {
            background: var(--navy);
            border-right: 1px solid #263d54;
        }
        [data-testid="stSidebar"] * { color: #f5f8fb; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #bdc9d6;
        }
        h1, h2, h3 { color: var(--navy); letter-spacing: -0.02em; }
        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px 14px;
            box-shadow: 0 3px 14px rgba(16, 36, 58, 0.04);
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: 0.82rem;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        [data-testid="stMetricValue"] { font-size: 1.65rem; }
        .app-eyebrow {
            color: var(--blue);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }
        .app-subtitle { color: var(--muted); margin-top: -0.6rem; }
        .app-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 20px 22px;
            margin: 0.3rem 0 1rem 0;
        }
        .app-pill {
                    display: inline-block;
                    border-radius: 999px;
                    padding: 0.25rem 0.65rem;
                    background: var(--blue-soft);
                    color: var(--navy);
                    font-size: 0.78rem;
                    font-weight: 650;
                }

                /* Collapsible sidebar on hover */
                [data-testid="stSidebar"] .css-1d39kcl {
                    transition: width 0.2s ease;
                }

                [data-testid="stSidebar"]:hover .css-1d39kcl .sidebar-text {
                    visibility: hidden;
                    width: 0;
                }

                [data-testid="stSidebar"]:hover .css-1d39kcl .sidebar-icon {
                    margin-right: 0.5rem;
                }

                [data-testid="stSidebar"]:not(:hover) .css-1d39kcl .sidebar-text {
                    visibility: visible;
                    width: auto;
                }

                [data-testid="stSidebar"]:not(:hover) .css-1d39kcl .sidebar-icon {
                    margin-right: 1rem;
                }
        .sidebar-brand {
            padding: 1.2rem 0.4rem 1rem;
            margin-bottom: 0.6rem;
            border-bottom: 1px solid #31475e;
        }
        .sidebar-brand strong { color: white; font-size: 1.08rem; }
        .sidebar-brand span { color: #bdc9d6; font-size: 0.78rem; }
        div.stButton > button[kind="primary"] {
            background: var(--blue);
            border-color: var(--blue);
            border-radius: 9px;
            font-weight: 650;
        }
        div.stButton > button { border-radius: 9px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    """Keep project branding fixed at the bottom of the sidebar."""

    st.sidebar.markdown(
    """
    <style>
    .sidebar-brand {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 255px;
        z-index: 999;
        box-sizing: border-box;

        background-color: #10263d;
        padding: 18px 15px 20px 15px;

        border: none !important;
        text-align: center;
    }

    .sidebar-brand strong {
        display: block;
        color: #ffffff;
        font-size: 14px;
        font-weight: 600;
        text-align: center;
        margin: 0;
    }

    .sidebar-brand span {
        display: block;
        color: #b9d0e5;
        font-size: 11px;
        text-align: center;
        margin-top: 6px;
    }
    </style>

    <div class="sidebar-brand">
        <strong>Loan Decision Support</strong>
        <span>Academic ML prototype</span>
    </div>
    """,
    unsafe_allow_html=True,
)




def render_page_header(eyebrow: str, title: str, subtitle: str) -> None:
    """Render a consistent page title and supporting summary."""
    st.markdown(f'<div class="app-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="app-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def render_disclaimer(text: str, *, kind: str = "info") -> None:
    """Render a standard Streamlit callout for an important limitation."""
    render = getattr(st, kind, st.info)
    render(text)


def render_historical_approval_disclosure() -> None:
    """Render the required historical Loan_Status scope statement."""
    st.warning(HISTORICAL_LOAN_STATUS_DISCLOSURE)
