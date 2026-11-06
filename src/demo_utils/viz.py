ORANGE, NAVY = "#FF3621", "#1B3139"
RISK_COLORS = {"High": ORANGE, "Medium": "#F5A623", "Low": "#2BB673"}


def banner_html(title, subtitle, steps):
    arrow = f"<span style='color:{ORANGE};font-weight:800'>➜</span>"
    chips = arrow.join(
        f"<span style='background:#fff;color:{NAVY};border-radius:999px;padding:6px 12px;font-weight:600;white-space:nowrap'>{s}</span>"
        for s in steps
    )
    return f"""
<div style='font-family:sans-serif;background:linear-gradient(120deg,{NAVY} 0%,#24434D 55%,{ORANGE} 140%);color:#fff;border-radius:16px;padding:28px 32px'>
  <div style='font-size:12px;letter-spacing:.25em;opacity:.8'>KAGGLE × DATABRICKS FREE EDITION</div>
  <div style='font-size:36px;font-weight:800;margin:6px 0 4px'>{title}</div>
  <div style='font-size:16px;opacity:.9;margin-bottom:18px'>{subtitle}</div>
  <div style='display:flex;flex-wrap:wrap;gap:8px;align-items:center'>{chips}</div>
</div>"""


def kpi_html(items):
    tiles = "".join(
        f"<div style='flex:1;min-width:150px;background:{NAVY};color:#fff;border-radius:12px;padding:14px 18px;border-left:6px solid {ORANGE}'>"
        f"<div style='font-size:12px;opacity:.75;text-transform:uppercase;letter-spacing:.08em'>{k}</div>"
        f"<div style='font-size:28px;font-weight:700;margin-top:4px'>{v}</div></div>"
        for k, v in items.items()
    )
    return f"<div style='display:flex;gap:12px;flex-wrap:wrap;font-family:sans-serif'>{tiles}</div>"


def links_html(title, links):
    rows = "".join(
        f"<li style='margin:8px 0'><a href='{url}' target='_blank' style='color:{ORANGE};font-weight:600;text-decoration:none'>{label} ↗</a></li>"
        for label, url in links.items()
    )
    return (f"<div style='font-family:sans-serif;border:2px solid {ORANGE};border-radius:12px;padding:14px 22px'>"
            f"<div style='font-size:18px;font-weight:700'>{title}</div><ul style='padding-left:18px'>{rows}</ul></div>")
