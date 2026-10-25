# -*- coding: utf-8 -*-
"""
研究型报告排版（Stage 2）：md -> 符合 NUAA Research Report 模板的 HTML。
模板要求：正文 Times New Roman 10 pt、1.5 倍行距、图题 Fig.N、表题 Tab.N、三线表。
"""
import os
import re
import html as _html
import markdown
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(ROOT, "report_source.md")
OUT = os.path.join(ROOT, "formatted_report.html")
IMGDIR = os.path.join(ROOT, "images")

# 报告里用的是重命名后的副本；仓库内直接复用 results/figures 里的原图
FIG_ALT = {
    "fig01_task1_overview.png": "../task1_stitching/results/figures/task1_overview.png",
    "fig02_task1_matches.png":  "../task1_stitching/results/figures/task1_pair1_matches.png",
    "fig03_t2_calibration.png": "../task2_traffic/results/figures/fig_t2_calibration.png",
    "fig04_t2_bev.png":         "../task2_traffic/results/figures/fig_t2_bev_tracks.png",
    "fig05_t2_speed.png":       "../task2_traffic/results/figures/fig_t2_speed.png",
    "fig06_t2_timeline.png":    "../task2_traffic/results/figures/fig_t2_timeline.png",
}

BODY_PT = "10pt"        # 模板：正文 10 pt
LINE_H = "1.5"          # 模板：1.5 倍行距


def eq_width_cm(png):
    return round(Image.open(png).size[0] / 400.0 * 2.54, 2)


CSS = """
  @page { @bottom-center { content: counter(page); } }
  @page cover { @bottom-center { content: none; } }
  section[role="cover"] { page: cover; }

  :root {
    --fs-body: __BODY__;
    --fs-caption: 9pt;
    --fs-h1: 12pt;
    --fs-h2: 10.5pt;
    --fs-h3: 10pt;
    --lh: __LH__;
  }

  * { box-sizing: border-box; }
  body {
    font-family: 宋体, SimSun, serif;
    font-size: __BODY__;
    line-height: __LH__;
    color: #000000;
    background: #ffffff;
    max-width: 16.0cm;
    margin-left: auto;
    margin-right: auto;
    padding: 0;
  }

  h1, h2, h3, h4 {
    font-family: 黑体, SimHei, sans-serif;
    font-weight: bold;
    line-height: __LH__;
    color: #000000;
  }
  h1 { font-size: var(--fs-h1); margin: 1.1em 0 0.5em 0; }
  h2 { font-size: var(--fs-h2); margin: 0.9em 0 0.4em 0; }
  h3 { font-size: var(--fs-h3); margin: 0.8em 0 0.35em 0; }
  h4 { font-size: var(--fs-h3); margin: 0.7em 0 0.3em 0; }

  p { margin: 0 0 0.30em 0; text-indent: 2em; text-align: justify; line-height: __LH__; }

  ul, ol { margin: 0.2em 0 0.6em 0; padding-left: 2.2em; }
  li { margin-bottom: 0.18em; text-align: justify; }
  ol.refs li { text-align: left; }

  table.three-line-table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.4em 0 0.9em 0;
    font-size: var(--fs-caption);
  }
  table.three-line-table thead tr:first-child { border-top: 1.5px solid #000000; }
  table.three-line-table thead tr:last-child { border-bottom: 0.75px solid #000000; }
  table.three-line-table tbody tr:last-child { border-bottom: 1.5px solid #000000; }
  table.three-line-table th, table.three-line-table td {
    border-left: none; border-right: none;
    padding: 0.28em 0.45em;
    text-align: left;
    text-indent: 0;
  }
  table.three-line-table th { font-weight: bold; }

  pre {
    font-family: Consolas, "Courier New", monospace;
    font-size: 8.5pt; line-height: 1.2;
    background: #f4f4f4; padding: 0.4em 0.6em; margin: 0.4em 0 0.6em 0; text-indent: 0;
  }
  code { font-family: Consolas, "Courier New", monospace; font-size: 9pt; }

  /* 封面 */
  .cover-no { text-align: right; font-size: 10pt; margin-bottom: 2.6em; }
  .cover-line { text-align: center; font-family: 黑体, SimHei, sans-serif; font-size: 16pt; font-weight: bold; margin-bottom: 0.5em; }
  .cover-sub { text-align: center; font-family: 黑体, SimHei, sans-serif; font-size: 12pt; margin-bottom: 0.4em; }
  .cover-date { text-align: center; font-size: 11pt; margin-bottom: 3.2em; }
  .cover-title { text-align: center; font-family: 黑体, SimHei, sans-serif; font-size: 17pt; font-weight: bold; line-height: 1.35; margin-bottom: 1.6em; }
  .cover-author { text-align: center; font-size: 12pt; margin-bottom: 0.35em; }
  .cover-affil { text-align: center; font-size: 10pt; margin-bottom: 2.4em; }
  table.cover-info-list { border-collapse: collapse; margin: 0 auto; width: 10.6cm; }
  table.cover-info-list td { border: none; padding: 0.32em 0.6em; font-size: 10pt; vertical-align: top; }
  table.cover-info-list td:first-child { text-align: right; white-space: nowrap; }
  table.cover-info-list td:last-child { text-align: left; border-bottom: 1px solid #000000; }

  /* 摘要 */
  .abs-label { font-family: 黑体, SimHei, sans-serif; font-weight: bold; text-indent: 0; margin-bottom: 0.3em; }
  .abs-text { text-indent: 2em; }
"""

CSS = CSS.replace("__BODY__", BODY_PT).replace("__LH__", LINE_H)


GREEK = {"omega": "ω", "theta": "θ", "rho": "ρ", "sigma": "σ", "mu": "μ",
         "alpha": "α", "beta": "β", "lambda": "λ", "Delta": "Δ", "pi": "π"}


def inline_tex(t):
    """把行内 LaTeX 片段转成可进 Word 的 HTML（加粗向量 + 上下标 + 希腊字母）。"""
    t = re.sub(r"\\mathbf\{([^}]*)\}", r"<b>\1</b>", t)
    t = re.sub(r"\\(?:mathrm|mathsf|text|operatorname)\{([^}]*)\}", r"\1", t)
    t = t.replace("\\mathbb{R}", "ℝ")
    t = re.sub(r"\\sqrt\{([^}]*)\}", r"√(\1)", t)
    for k, v in GREEK.items():
        t = t.replace("\\" + k, v)
    for cmd, rep in (("\\tan", "tan"), ("\\log", "log"), ("\\max", "max"),
                     ("\\min", "min"), ("\\exp", "exp")):
        t = t.replace(cmd, rep)
    for lx, rep in (("\\le", "≤"), ("\\leq", "≤"), ("\\ge", "≥"), ("\\approx", "≈"),
                    ("\\times", "×"), ("\\cdot", "·"), ("\\pm", "±"), ("\\in", "∈")):
        t = t.replace(lx, rep)
    t = re.sub(r"\^\{([^}]*)\}", r"<sup>\1</sup>", t)
    t = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", t)
    t = re.sub(r"\^(\w)", r"<sup>\1</sup>", t)
    t = re.sub(r"_(\w)", r"<sub>\1</sub>", t)
    t = t.replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    return t.strip()


def build_body(md_text):
    cover = {}
    m = re.search(r"## 封面信息\s*\n(.*?)\n(?=## )", md_text, re.S)
    if m:
        for row in re.findall(r"^\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$", m.group(1), re.M):
            if row[0] in ("字段",) or row[0].startswith("-"):
                continue
            cover[row[0]] = row[1]
    body_md = md_text.replace(m.group(0), "") if m else md_text
    body_md = re.sub(r"^#\s+.+$\n?", "", body_md, count=1, flags=re.M)

    eqs = []

    def _eq_sub(mo):
        eqs.append(mo.group(1).strip())
        return "\n\nEQPLACEHOLDER%d\n\n" % (len(eqs) - 1)

    body_md = re.sub(r"\$\$(.+?)\$\$", _eq_sub, body_md, flags=re.S)

    # 行内公式 -> HTML
    body_md = re.sub(r"\$([^$\n]+?)\$", lambda m: inline_tex(m.group(1)), body_md)

    # 摘要单独抽出来
    abs_m = re.search(r"## Abstract\s*\n(.*?)\n(?=## )", body_md, re.S)
    abstract_md = abs_m.group(1) if abs_m else ""
    if abs_m:
        body_md = body_md.replace(abs_m.group(0), "\n")

    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"])
    body = md.convert(body_md)

    # 公式占位 -> 居中公式 + 右对齐编号
    def _eq_repl(mo):
        idx = int(mo.group(1))
        tex = eqs[idx]
        tag = re.search(r"\\tag\{(\d+)\}", tex)
        num = tag.group(1) if tag else str(idx + 1)
        png = os.path.join(IMGDIR, "eq", "eq%02d.png" % int(num))
        w = eq_width_cm(png)
        return ('<p class="eq-line" style="text-align:center; text-indent:0; margin:0.45em 0; line-height:1.0;">'
                '<img src="images/eq/eq%02d.png" width="%d" style="width:%.2fcm"></p>'
                % (int(num), round(w * 37.795), w))

    body = re.sub(r"<p>EQPLACEHOLDER(\d+)</p>", _eq_repl, body)

    # 图 + 图题
    def _fig_repl(mo):
        img, cap = mo.group(1), re.sub(r"</?strong>", "", mo.group(2)).strip()
        src = re.search(r'src="([^"]+)"', img)
        name = os.path.basename(src.group(1)) if src else ""
        if name and not os.path.exists(os.path.join(IMGDIR, name)) and name in FIG_ALT:
            img = img.replace('src="%s"' % src.group(1), 'src="%s"' % FIG_ALT[name])
        if name.startswith("fig01"):
            cm = 12.0
        elif name.startswith("fig06"):
            cm = 14.0
        else:
            cm = 15.0
        img2 = re.sub(r"<img ", '<img width="%d" style="width:%scm;" ' % (int(round(cm * 37.795)), int(cm)), img, count=1)
        return ('<p style="text-align:center; text-indent:0; margin:0.45em 0 0.1em 0;">%s</p>'
                '<p style="text-align:center; text-indent:0; font-size:9pt; margin:0 0 0.9em 0;">%s</p>') % (img2, cap)

    body = re.sub(r'<p>(<img[^>]*>)\s*</p>\s*<p><strong>((?:Fig\.|图)\s*\d+</strong>.*?)</p>',
                  _fig_repl, body, flags=re.S)

    # 表题（表前独立居中段）
    def _cap_repl(mo):
        cap = re.sub(r"</?strong>", "", mo.group(1)).strip()
        table = mo.group(2).replace("<table>", '<table class="three-line-table">', 1)
        return ('<p style="text-align:center; text-indent:0; font-size:9pt; margin:0.6em 0 0.25em 0;">%s</p>%s'
                % (cap.replace("\\", ""), table))

    body = re.sub(r"<p><strong>((?:Tab\.|表)\s*\d+</strong>.*?)</p>\s*(<table>.*?</table>)",
                  _cap_repl, body, flags=re.S)
    body = body.replace("<table>", '<table class="three-line-table">')

    # 文献表：左对齐并悬挂缩进（分散对齐会把长条目字距拉散）
    body = re.sub(r"(<h2>References</h2>\s*)<ol>", r'\1<ol class="refs">', body)

    # 摘要块
    abs_html = markdown.Markdown().convert(abstract_md) if abstract_md else ""
    abs_html = abs_html.replace("<p>", '<p class="abs-text">')
    abstract_block = '<p class="abs-label">Abstract</p>' + abs_html

    return body, abstract_block, cover


def main():
    md_text = open(MD, encoding="utf-8").read()
    title_m = re.search(r"^#\s+(.+)$", md_text, re.M)
    doc_title = title_m.group(1).strip() if title_m else "Research Report"
    body, abstract_block, cover = build_body(md_text)

    cv = lambda k, d="": _html.escape(cover.get(k, d) or d)

    cover_html = """
  <section role="cover">
    <p class="cover-no">No.&nbsp;{no}</p>
    <p class="cover-line">Research Report of PR &amp; IP</p>
    <p class="cover-sub">College of Automation Engineering, NUAA</p>
    <p class="cover-date">二〇二六 年 十 月</p>
    <p class="cover-title">{title}</p>
    <p class="cover-author">{author}</p>
    <p class="cover-affil">(College of Automation Engineering, NUAA)</p>
    <table class="cover-info-list">
      <tr><td>Student Name</td><td>{name}</td></tr>
      <tr><td>Student ID</td><td>{sid}</td></tr>
      <tr><td>College</td><td>自动化学院 College of Automation Engineering</td></tr>
      <tr><td>Major</td><td>{major}</td></tr>
      <tr><td>Mobile</td><td>{mobile}</td></tr>
    </table>
  </section>""".format(no=cv("No."), title=_html.escape(doc_title), author=cv("Author"),
                       name=cv("Student Name"), sid=cv("Student ID"),
                       major=cv("Major"), mobile=cv("Mobile"))

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="docx-page-size" content="A4">
<title>{_html.escape(doc_title)}</title>
<style>
{CSS}
</style>
</head>
<body>
{cover_html}
  <section role="body" data-page-restart="1">
{abstract_block}
<main class="paper-body">
{body}
</main>
  </section>
</body>
</html>
"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    print("HTML:", OUT, len(doc), "bytes")
    print("cover:", {k: v for k, v in cover.items() if k in ("No.", "Author", "Major", "Mobile")})


if __name__ == "__main__":
    main()
