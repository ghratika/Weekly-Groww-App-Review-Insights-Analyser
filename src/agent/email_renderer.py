"""
Email renderer (Phase 5).

Builds HTML and plain-text payloads for the Gmail MCP.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any

from src.agent.summarizer import Cluster

logger = logging.getLogger("pulse.email_renderer")

@dataclass
class EmailPayload:
    subject: str
    html_body: str
    text_body: str
    to: list[str]

def render_email(clusters: list[Cluster], metadata: dict, config: dict) -> EmailPayload:
    """
    Build HTML + plain-text email with deep link.
    
    Args:
        clusters: List of validated Cluster objects.
        metadata: Dict containing 'iso_week', 'doc_id', 'doc_heading_id'.
        config: Full config dict.
        
    Returns:
        EmailPayload containing subject, html_body, text_body, and recipients.
    """
    delivery_cfg = config.get("delivery", {})
    product_cfg = config.get("product", {})
    
    product_name = product_cfg.get("name", "App")
    iso_week = metadata.get("iso_week", "YYYY-WXX")
    
    subject_template = delivery_cfg.get("email_subject_template", "{product_name} Review Pulse — {iso_week}")
    subject = subject_template.format(product_name=product_name, iso_week=iso_week)
    
    to_recipients = delivery_cfg.get("recipients", [])
    
    doc_id = delivery_cfg.get("google_doc_id", "DOC_ID")
    # For now, generate a placeholder heading ID if not passed
    # The actual deep link logic in Phase 6 will use the real heading ID if it exists
    doc_heading_id = metadata.get("doc_heading_id", "heading=h.placeholder")
    deep_link = f"https://docs.google.com/document/d/{doc_id}/edit#{doc_heading_id}"
    
    total_reviews = sum(c.review_count for c in clusters)
    
    # Plain text
    text_body = f"This week's top themes from {total_reviews} {product_name} reviews:\n"
    for i, c in enumerate(clusters[:3], 1):
        text_body += f"• {c.theme_name} ({c.review_count} mentions)\n"
    
    text_body += f"\n📄 Read the full report → {deep_link}\n"
    text_body += "\n---\nThis is an automated report from the Weekly Review Pulse system.\n"
    
    # HTML text
    html_body = f"<p>This week's top themes from <b>{total_reviews}</b> {product_name} reviews:</p>\n<ul>\n"
    for c in clusters[:3]:
        html_body += f"  <li><b>{c.theme_name}</b> ({c.review_count} mentions)</li>\n"
    html_body += "</ul>\n"
    html_body += f"<p>📄 <a href=\"{deep_link}\">Read the full report →</a></p>\n"
    html_body += "<hr/>\n<p><small>This is an automated report from the Weekly Review Pulse system.</small></p>\n"
    
    return EmailPayload(
        subject=subject,
        html_body=html_body,
        text_body=text_body,
        to=to_recipients
    )
