"""
notification.py
---------------
Notification subsystem for the MLB K-Predictor dashboard layer.

Supports:
  - Webhook (Slack / Discord / generic HTTP POST)
  - Email via SMTP
  - Anomaly detection alerts (high K predictions, model degradation)

Environment variables
---------------------
WEBHOOK_URL   – Default webhook endpoint (can be overridden per-call)
SMTP_HOST     – SMTP server hostname (e.g. smtp.gmail.com)
SMTP_PORT     – SMTP port (default 587)
SMTP_USER     – SMTP login username
SMTP_PASS     – SMTP login password
ALERT_EMAIL   – Default recipient for alert emails
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import smtplib
import textwrap
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import pandas as pd
import requests
from loguru import logger


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------

def send_webhook(url: str, payload: dict) -> bool:
    """POST a JSON payload to a webhook URL (Slack, Discord, generic HTTP).

    Parameters
    ----------
    url:
        The webhook endpoint URL.
    payload:
        Serialisable dict that will be sent as ``application/json``.

    Returns
    -------
    bool
        ``True`` if the server responded with HTTP 2xx, ``False`` otherwise.
    """
    if not url:
        logger.warning("send_webhook called with empty URL – skipping.")
        return False
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.ok:
            logger.info(f"Webhook delivered → {url} (HTTP {response.status_code})")
            return True
        logger.warning(
            f"Webhook request failed → {url} "
            f"(HTTP {response.status_code}): {response.text[:200]}"
        )
        return False
    except requests.exceptions.Timeout:
        logger.error(f"Webhook timed out after 10 s → {url}")
        return False
    except requests.exceptions.RequestException as exc:
        logger.error(f"Webhook request error → {url}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def send_email_alert(
    subject: str,
    body: str,
    to_email: Optional[str] = None,
) -> None:
    """Send a plain-text alert email via SMTP.

    Reads connection details from environment variables::

        SMTP_HOST   – required
        SMTP_PORT   – optional (default 587)
        SMTP_USER   – required
        SMTP_PASS   – required
        ALERT_EMAIL – default recipient if *to_email* is not supplied

    If any required variable is absent the function logs a warning and
    returns without raising.

    Parameters
    ----------
    subject:
        Email subject line.
    body:
        Plain-text email body.
    to_email:
        Recipient address.  Falls back to the ``ALERT_EMAIL`` env var.
    """
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    recipient = to_email or os.environ.get("ALERT_EMAIL")

    missing = [v for v, val in [
        ("SMTP_HOST", smtp_host),
        ("SMTP_USER", smtp_user),
        ("SMTP_PASS", smtp_pass),
        ("ALERT_EMAIL / to_email", recipient),
    ] if not val]

    if missing:
        logger.warning(
            f"Email alert skipped – missing env vars / parameters: {missing}.  "
            "Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_EMAIL."
        )
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipient, msg.as_string())
        logger.info(f"Email alert sent → {recipient}: '{subject}'")
    except smtplib.SMTPException as exc:
        logger.error(f"SMTP error while sending alert to {recipient}: {exc}")
    except OSError as exc:
        logger.error(f"Network error while sending alert to {recipient}: {exc}")


# ---------------------------------------------------------------------------
# Prediction anomaly alert
# ---------------------------------------------------------------------------

def alert_prediction_anomaly(
    predictions: pd.DataFrame,
    threshold_k: float = 12.0,
    webhook_url: Optional[str] = None,
    to_email: Optional[str] = None,
) -> None:
    """Fire webhook and email alerts for any predicted K total above *threshold_k*.

    Parameters
    ----------
    predictions:
        DataFrame with at least ``pitcher_id`` and ``predicted_strikeouts``.
    threshold_k:
        Strikeout threshold above which a prediction is considered anomalous.
    webhook_url:
        Explicit webhook URL; falls back to the ``WEBHOOK_URL`` env var.
    to_email:
        Alert recipient; falls back to the ``ALERT_EMAIL`` env var.
    """
    if "predicted_strikeouts" not in predictions.columns:
        logger.warning("alert_prediction_anomaly: 'predicted_strikeouts' column missing.")
        return

    flagged = predictions[predictions["predicted_strikeouts"] > threshold_k].copy()
    if flagged.empty:
        logger.debug(
            f"No prediction anomalies found (threshold={threshold_k} K)."
        )
        return

    logger.warning(
        f"Anomaly alert: {len(flagged)} pitcher(s) predicted > {threshold_k} K."
    )

    # Build notification content
    rows = []
    for _, row in flagged.iterrows():
        parts = [
            f"  Pitcher {row['pitcher_id']}: "
            f"{row['predicted_strikeouts']:.1f} K"
        ]
        if "pred_ci_lower" in row and "pred_ci_upper" in row:
            parts.append(f"(CI: {row['pred_ci_lower']:.1f}–{row['pred_ci_upper']:.1f})")
        if "game_date" in row:
            parts.append(f"on {row['game_date']}")
        rows.append(" ".join(parts))

    summary_text = (
        f"[MLB K-Predictor] Prediction Anomaly Detected\n"
        f"Threshold: {threshold_k} K  |  Flagged: {len(flagged)} pitcher(s)\n"
        f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n\n"
        + "\n".join(rows)
    )

    # Slack-compatible payload
    slack_payload = {
        "text": summary_text,
        "username": "MLB K-Predictor",
        "icon_emoji": ":baseball:",
    }

    url = webhook_url or os.environ.get("WEBHOOK_URL", "")
    if url:
        send_webhook(url, slack_payload)
    else:
        logger.debug("No WEBHOOK_URL configured – skipping webhook for anomaly alert.")

    send_email_alert(
        subject=f"[MLB K-Predictor] Anomaly: {len(flagged)} pitcher(s) > {threshold_k} K",
        body=summary_text,
        to_email=to_email,
    )


# ---------------------------------------------------------------------------
# Model degradation alert
# ---------------------------------------------------------------------------

def alert_model_degradation(
    recent_mae: float,
    baseline_mae: float = 1.8,
    threshold_pct: float = 0.05,
    webhook_url: Optional[str] = None,
    to_email: Optional[str] = None,
) -> None:
    """Fire alerts when recent MAE degrades beyond a threshold vs. the baseline.

    Parameters
    ----------
    recent_mae:
        MAE computed over the most recent evaluation window.
    baseline_mae:
        Expected / historical baseline MAE (default 1.8).
    threshold_pct:
        Alert if ``(recent_mae - baseline_mae) / baseline_mae > threshold_pct``.
        Default 0.05 (5 %).
    webhook_url:
        Explicit webhook URL; falls back to the ``WEBHOOK_URL`` env var.
    to_email:
        Alert recipient; falls back to the ``ALERT_EMAIL`` env var.
    """
    if baseline_mae <= 0:
        logger.warning("alert_model_degradation: baseline_mae must be > 0.")
        return

    degradation_pct = (recent_mae - baseline_mae) / baseline_mae

    if degradation_pct <= threshold_pct:
        logger.debug(
            f"Model performance OK: recent MAE={recent_mae:.4f}, "
            f"baseline={baseline_mae:.4f}, "
            f"degradation={degradation_pct:+.2%} (threshold={threshold_pct:.0%})."
        )
        return

    logger.warning(
        f"Model degradation detected! recent_mae={recent_mae:.4f} vs "
        f"baseline={baseline_mae:.4f} (+{degradation_pct:.1%})"
    )

    body = textwrap.dedent(f"""
        [MLB K-Predictor] Model Degradation Alert
        ==========================================
        Recent MAE     : {recent_mae:.4f}
        Baseline MAE   : {baseline_mae:.4f}
        Degradation    : {degradation_pct:+.2%} (threshold {threshold_pct:.0%})
        Timestamp      : {datetime.now(timezone.utc).isoformat()}

        Action required: Review recent predictions, check data pipeline, and
        consider retraining or rolling back the production model.
    """).strip()

    slack_payload = {
        "text": body,
        "username": "MLB K-Predictor",
        "icon_emoji": ":warning:",
        "attachments": [
            {
                "color": "danger",
                "fields": [
                    {"title": "Recent MAE", "value": f"{recent_mae:.4f}", "short": True},
                    {"title": "Baseline MAE", "value": f"{baseline_mae:.4f}", "short": True},
                    {"title": "Degradation", "value": f"{degradation_pct:+.2%}", "short": True},
                ],
            }
        ],
    }

    url = webhook_url or os.environ.get("WEBHOOK_URL", "")
    if url:
        send_webhook(url, slack_payload)
    else:
        logger.debug("No WEBHOOK_URL configured – skipping webhook for degradation alert.")

    send_email_alert(
        subject=f"[MLB K-Predictor] Model Degradation: MAE={recent_mae:.4f} "
                f"(+{degradation_pct:.1%} vs baseline)",
        body=body,
        to_email=to_email,
    )


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------

def format_daily_summary(predictions: pd.DataFrame) -> str:
    """Format a predictions DataFrame into a clean text summary for notifications.

    Parameters
    ----------
    predictions:
        DataFrame with at least ``pitcher_id`` and ``predicted_strikeouts``.
        Optional columns: ``game_date``, ``game_pk``, ``pred_ci_lower``,
        ``pred_ci_upper``.

    Returns
    -------
    str
        A multi-line text summary suitable for Slack / Discord / email body.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required = {"pitcher_id", "predicted_strikeouts"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions DataFrame missing columns: {missing}")

    df = predictions.sort_values("predicted_strikeouts", ascending=False).reset_index(drop=True)

    date_label = ""
    if "game_date" in df.columns:
        dates = df["game_date"].dropna().unique()
        if len(dates) == 1:
            date_label = f" — {dates[0]}"
        elif len(dates) > 1:
            date_label = f" — {min(dates)} to {max(dates)}"

    lines = [
        f"MLB K-Predictor Daily Summary{date_label}",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Pitchers: {len(df)}  |  "
        f"Avg K: {df['predicted_strikeouts'].mean():.2f}  |  "
        f"Max K: {df['predicted_strikeouts'].max():.1f}",
        "",
        f"{'Rank':<5} {'Pitcher ID':<14} {'Pred K':>7}",
    ]

    has_ci = "pred_ci_lower" in df.columns and "pred_ci_upper" in df.columns
    if has_ci:
        lines[-1] += f"  {'CI Lower':>9}  {'CI Upper':>9}"

    lines.append("-" * (55 if has_ci else 30))

    for i, row in df.iterrows():
        rank = i + 1
        line = (
            f"{rank:<5} {str(row['pitcher_id']):<14} "
            f"{row['predicted_strikeouts']:>7.1f}"
        )
        if has_ci:
            line += (
                f"  {row['pred_ci_lower']:>9.1f}"
                f"  {row['pred_ci_upper']:>9.1f}"
            )
        lines.append(line)

    # Highlight outliers
    high_k = df[df["predicted_strikeouts"] >= 10]
    if not high_k.empty:
        lines.append("")
        lines.append(
            f"⚡ High-K alerts (≥10): "
            + ", ".join(
                f"Pitcher {r['pitcher_id']} ({r['predicted_strikeouts']:.1f})"
                for _, r in high_k.iterrows()
            )
        )

    return "\n".join(lines)
