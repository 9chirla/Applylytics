"""HTML email bodies and attachment filenames for optimised CV delivery."""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone

from applylytics.email.job_metadata import JobMetadata, is_junk_metadata_text


def slugify_filename_part(text: str, *, max_len: int = 40) -> str:
    """Safe fragment for attachment names: letters, numbers, underscores."""
    if is_junk_metadata_text(text):
        return ""
    t = (text or "").strip()
    t = re.sub(r"[^\w\s\-]", "", t, flags=re.UNICODE)
    t = re.sub(r"[\s\-]+", "_", t).strip("_")
    if not t:
        return ""
    return t[:max_len]


def build_cv_attachment_filename(
    *,
    candidate_name: str,
    job: JobMetadata,
) -> str:
    """
    e.g. Naveen_Reddy_Chirla_CV_Acme_Ltd_Junior_Data_Analyst.docx
    """
    name_part = slugify_filename_part(candidate_name, max_len=50) or "Candidate"
    company_part = slugify_filename_part(job.company_name, max_len=35) if job.has_company else ""
    role_part = (
        slugify_filename_part(job.display_title, max_len=45)
        if job.display_title != "your target role"
        else ""
    )

    parts = [name_part, "CV"]
    if company_part:
        parts.append(company_part)
    if role_part:
        parts.append(role_part)
    base = "_".join(p for p in parts if p)
    if len(base) > 120:
        base = base[:120].rstrip("_")
    return f"{base}.docx"


def build_email_subject(job: JobMetadata) -> str:
    title = job.display_title
    if job.has_company:
        return f"Your optimised CV — {title} at {job.company_name}"
    return f"Your optimised CV — {title}"


def build_optimised_cv_html(
    *,
    recipient_name: str,
    job: JobMetadata,
    portfolio_url: str = "",
) -> str:
    """Inline-styled HTML for Brevo transactional email."""
    name = html.escape(recipient_name.strip() or "there")
    role = html.escape(job.display_title)
    company = html.escape(job.company_name) if job.has_company else ""
    location = html.escape(job.location) if job.location else ""
    year = datetime.now(timezone.utc).strftime("%Y")

    if job.has_company:
        role_heading = f"{role} <span style=\"color:#64748b;font-weight:400;\">at</span> {company}"
        role_sub = "Built for your application to this company"
    else:
        role_heading = role
        role_sub = "Built for the role you are targeting"

    location_row = ""
    if location:
        location_row = (
            f'<p style="margin:10px 0 0;font-size:13px;color:#64748b;">'
            f'<span style="color:#94a3b8;">Location</span> &nbsp; {location}</p>'
        )

    portfolio_block = ""
    if (portfolio_url or "").strip():
        url = html.escape(portfolio_url.strip())
        portfolio_block = f"""
        <p style="margin:20px 0 0;font-size:13px;color:#64748b;">
          Sent via <a href="{url}" style="color:#4f46e5;text-decoration:none;">Applylytics</a>
        </p>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Your optimised CV</title>
</head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f1f5f9;padding:32px 16px;">
    <tr>
      <td align="center">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:560px;background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(15,23,42,0.08);">
          <tr>
            <td style="background:linear-gradient(135deg,#1e1b4b 0%,#4f46e5 100%);padding:28px 32px;">
              <p style="margin:0 0 6px;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;color:rgba(255,255,255,0.75);">Applylytics</p>
              <h1 style="margin:0;font-size:22px;font-weight:600;color:#ffffff;line-height:1.3;">Your optimised CV is ready</h1>
            </td>
          </tr>
          <tr>
            <td style="padding:32px;">
              <p style="margin:0 0 20px;font-size:16px;line-height:1.6;color:#334155;">Hi {name},</p>
              <p style="margin:0 0 24px;font-size:15px;line-height:1.65;color:#475569;">
                Your CV for the role below is ready. The updated document is attached as a
                <strong style="color:#1e293b;">Word (.docx)</strong> file — open it, review every line, then save as PDF if the employer asks for PDF.
              </p>
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;margin:0 0 28px;">
                <tr>
                  <td style="padding:20px 22px;">
                    <p style="margin:0 0 4px;font-size:11px;letter-spacing:0.06em;text-transform:uppercase;color:#94a3b8;font-weight:600;">Application target</p>
                    <p style="margin:0 0 8px;font-size:18px;font-weight:600;color:#0f172a;line-height:1.35;">{role_heading}</p>
                    <p style="margin:0;font-size:13px;color:#64748b;">{html.escape(role_sub)}</p>
                    {location_row}
                  </td>
                </tr>
              </table>
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#fffbeb;border:1px solid #fde68a;border-radius:10px;margin:0 0 28px;">
                <tr>
                  <td style="padding:18px 22px;">
                    <p style="margin:0 0 10px;font-size:13px;font-weight:600;color:#92400e;">Before you apply</p>
                    <ul style="margin:0;padding:0 0 0 18px;font-size:14px;line-height:1.55;color:#78350f;">
                      <li style="margin-bottom:6px;">Confirm every skill and metric is accurate — only send what you can defend in an interview.</li>
                      <li style="margin-bottom:6px;">Tweak the personal statement if anything reads generic for this employer.</li>
                      <li>Export to PDF when the job portal requires it (File → Save As → PDF in Word).</li>
                    </ul>
                  </td>
                </tr>
              </table>
              <p style="margin:0;font-size:15px;line-height:1.6;color:#475569;">Good luck with your application — you've got this.</p>
              <p style="margin:16px 0 0;font-size:14px;color:#64748b;">— Applylytics</p>
              {portfolio_block}
            </td>
          </tr>
          <tr>
            <td style="padding:16px 32px 24px;border-top:1px solid #f1f5f9;">
              <p style="margin:0;font-size:11px;color:#94a3b8;line-height:1.5;text-align:center;">
                © {year} Applylytics · CV optimisation for your job search · This inbox is not monitored for replies.
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


def build_plain_text_fallback(
    *,
    recipient_name: str,
    job: JobMetadata,
    attachment_name: str,
) -> str:
    name = recipient_name.strip() or "there"
    lines = [
        f"Hi {name},",
        "",
        "Your optimised CV is ready to download.",
        "",
        f"Role: {job.role_line()}",
    ]
    if job.location:
        lines.append(f"Location: {job.location}")
    lines.extend(
        [
            "",
            f"Attached file: {attachment_name}",
            "",
            "Before you apply:",
            "- Verify all skills and numbers are truthful.",
            "- Adjust the personal statement if needed.",
            "- Save as PDF if the employer requires it.",
            "",
            "Good luck!",
            "— Applylytics",
        ]
    )
    return "\n".join(lines)
