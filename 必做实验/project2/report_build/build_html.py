# -*- coding: utf-8 -*-
"""
doc-typeset 执行体（Stage 2 内容→HTML 转换）
把 stage1/final_draft.md 转为符合《实验指导书》排版要求的 HTML：
  一级标题 四号黑体 / 二级标题 小四黑体 / 正文 小四宋体 / 1.2 倍行距
  公式居中、编号靠右；图、表及其标题居中
所有样式走 CSS 变量（design tokens），禁止裸值。
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
FIG_ALT = os.path.join(os.path.dirname(ROOT), "results", "figures")   # 原始插图仍从 results/figures 复用

# ---------- 设计令牌（覆盖 academic-paper 主题以匹配指导书要求） ----------
TOKENS = {
    "--typography-fontFamily-heading": "黑体, SimHei, sans-serif",
    "--typography-fontFamily-body": "宋体, SimSun, serif",
    "--typography-fontSize-h1": "14pt",      # 四号
    "--typography-fontSize-h2": "12pt",      # 小四
    "--typography-fontSize-h3": "12pt",
    "--typography-fontSize-h4": "12pt",
    "--typography-fontSize-body": "12pt",    # 小四
    "--typography-fontSize-caption": "10.5pt",
    "--typography-lineHeight-body": "1.2",   # 1.2 倍行距
    "--typography-lineHeight-heading": "1.2",
    "--color-primary": "#000000",
    "--color-text": "#000000",
    "--color-muted": "#000000",
    "--color-border": "#000000",
    "--color-bg": "#ffffff",
}
LINEW = {k: v for k, v in TOKENS.items()}
ROOT_CSS = "\n".join("    %s: %s;" % (k, v) for k, v in LINEW.items())


def eq_width_cm(png):
    px = Image.open(png).size[0]
    return round(px / 400.0 * 2.54, 2)


def build_body(md_text):
    # ---- 抽出封面信息表 ----
    cover = {}
    m = re.search(r"## 封面信息\s*\n(.*?)\n(?=## )", md_text, re.S)
    if m:
        for row in re.findall(r"^\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$", m.group(1), re.M):
            if row[0] in ("字段", "------"):
                continue
            cover[row[0]] = row[1]
    body_md = md_text[m.start():] if m else md_text
    body_md = body_md.replace(m.group(0), "") if m else body_md

    # 封面已单独渲染，正文不再重复英文主标题
    body_md = re.sub(r"^#\s+.+$\n?", "", body_md, count=1, flags=re.M)

    # ---- 抽出显示公式，替换为占位符 ----
    eqs = []

    def _eq_sub(mo):
        eqs.append(mo.group(1).strip())
        return "\n\nEQPLACEHOLDER%d\n\n" % (len(eqs) - 1)

    body_md = re.sub(r"\$\$(.+?)\$\$", _eq_sub, body_md, flags=re.S)

    # ---- 摘要段单独处理 ----
    abs_m = re.search(r"## 摘要\s*\n(.*?)\n(?=## )", body_md, re.S)
    abstract_html = ""
    if abs_m:
        abstract_html = abs_m.group(1)
        body_md = body_md.replace(abs_m.group(0), "\n")

    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"])
    body = md.convert(body_md)

    # ---- 公式占位符 → 居中式 + 编号靠右 ----
    def _eq_repl(mo):
        idx = int(mo.group(1))
        tex = eqs[idx]
        tag = re.search(r"\\tag\{(\d+)\}", tex)
        num = tag.group(1) if tag else str(idx + 1)
        png = os.path.join(IMGDIR, "eq", "eq%02d.png" % int(num))
        w = eq_width_cm(png)
        return (
            '<p class="eq-line" style="text-align:center; text-indent:0; margin:0.5em 0; line-height:1.0;">'
            '<img class="eq-img" src="images/eq/eq%02d.png" width="%d" style="width:%.2fcm"></p>'
            % (int(num), round(w * 37.795), w)
        )

    body = re.sub(r"<p>EQPLACEHOLDER(\d+)</p>", _eq_repl, body)

    # ---- 图片 + 图题 → 居中图块（内联样式，避免依赖选择器解析） ----
    def _fig_repl(mo):
        img = mo.group(1)
        cap = re.sub(r"</?strong>", "", mo.group(2)).strip()
        src = re.search(r'src="([^"]+)"', img)
        name = os.path.basename(src.group(1)) if src else ""
        if not os.path.exists(os.path.join(IMGDIR, name)) and os.path.exists(os.path.join(FIG_ALT, name)):
            img = img.replace('src="%s"' % src.group(1), 'src="../results/figures/%s"' % name)
        cm = 12.0 if name.startswith("fig00") else 16.0
        px = int(round(cm * 37.795))
        img2 = re.sub(r"<img ", '<img width="%d" style="width:%scm;" ' % (px, int(cm)), img, count=1)
        return (
            '<p style="text-align:center; text-indent:0; margin:0.45em 0 0.12em 0;">%s</p>'
            '<p style="text-align:center; text-indent:0; font-size:10.5pt; margin:0 0 0.75em 0;">%s</p>'
        ) % (img2, cap)

    body = re.sub(
        r'<p>(<img[^>]*>)\s*</p>\s*<p><strong>(图\s*\d+</strong>.*?)</p>',
        _fig_repl, body, flags=re.S)

    # ---- 表题 → 表上方的居中独立段落（转换器不支持 <caption>） ----
    def _cap_repl(mo):
        cap = re.sub(r"</?strong>", "", mo.group(1)).strip()
        table = mo.group(2).replace("<table>", '<table class="three-line-table">', 1)
        return (
            '<p style="text-align:center; text-indent:0; font-size:10.5pt; margin:0.55em 0 0.25em 0;">%s</p>'
            "%s"
        ) % (cap.replace("\\", ""), table)

    body = re.sub(r"<p><strong>(表\s*\d+</strong>.*?)</p>\s*(<table>.*?</table>)",
                  _cap_repl, body, flags=re.S)
    # 其余无编号的表也统一套三线表样式
    body = body.replace("<table>", '<table class="three-line-table">')

    return body, abstract_html, cover


CSS = """
  @page { @bottom-center { content: counter(page); } }
  @page cover { @bottom-center { content: none; } }
  section[role="cover"] { page: cover; }

  :root {
%s
    --fs-h1: 14pt;
    --fs-h2: 14pt;
    --fs-h3: 12pt;
    --fs-h4: 12pt;
    --fs-body: 12pt;
    --fs-caption: 10.5pt;
    --ff-heading: 黑体, SimHei, sans-serif;
    --ff-body: 宋体, SimSun, serif;
    --lh-body: 1.2;
    --page-content-width: 16.0cm;
    --color-primary: #000000;
    --color-text: #000000;
    --color-border: #000000;
    --color-bg: #ffffff;
  }

  * { box-sizing: border-box; }
  body {
    font-family: 宋体, SimSun, serif;
    font-size: var(--fs-body);
    line-height: 1.2;
    color: var(--color-text);
    background: var(--color-bg);
    max-width: var(--page-content-width);
    margin-left: auto;
    margin-right: auto;
    padding: 0;
  }

  h1, h2, h3, h4 {
    font-family: 黑体, SimHei, sans-serif;
    font-weight: bold;
    line-height: var(--typography-lineHeight-heading);
    color: var(--color-primary);
  }
  h1 { font-size: var(--fs-h1); margin: 0.8em 0 0.5em 0; text-align: left; }
  h2 { font-size: var(--fs-h2); margin: 0.9em 0 0.45em 0; }
  h3 { font-size: var(--fs-h3); margin: 0.8em 0 0.4em 0; }
  h4 { font-size: var(--fs-h4); margin: 0.7em 0 0.35em 0; }

  p { margin: 0 0 0.22em 0; text-indent: 2em; text-align: justify; line-height: 1.2; }
  p.eq-lead { text-indent: 2em; }

  ul, ol { margin: 0.2em 0 0.5em 0; padding-left: 2.6em; }
  li { margin-bottom: 0.04em; }

  table.three-line-table {
    border-collapse: collapse;
    width: 100%%;
    margin: 0.4em 0 0.8em 0;
    font-size: var(--fs-caption);
  }
  table.three-line-table caption {
    font-size: var(--fs-caption);
    font-family: var(--ff-heading);
    text-align: center;
    margin-bottom: 0.35em;
  }
  table.three-line-table thead tr:first-child { border-top: 1.5px solid var(--color-text); }
  table.three-line-table thead tr:last-child { border-bottom: 0.75px solid var(--color-text); }
  table.three-line-table tbody tr:last-child { border-bottom: 1.5px solid var(--color-text); }
  table.three-line-table th, table.three-line-table td {
    border-left: none; border-right: none;
    padding: 0.3em 0.5em;
    text-align: left;
    text-indent: 0;
  }
  table.three-line-table th { font-weight: bold; }

  /* 公式：居中 + 编号靠右（无框线三列表） */
  table.eq-table { border-collapse: collapse; width: 100%%; margin: 0.45em 0 0.55em 0; }
  table.eq-table td { border: none; padding: 0; vertical-align: middle; text-indent: 0; }
  td.eq-cell-left { width: 6%%; }
  td.eq-cell-body { text-align: center; width: 82%%; }
  td.eq-cell-num { width: 12%%; text-align: right; font-size: var(--fs-body); }
  img.eq-img { vertical-align: middle; }

  figure.doc-figure { margin: 0.5em 0 0.7em 0; text-align: center; }
  figure.doc-figure img { max-width: 100%%; }
  figure.doc-figure figcaption {
    font-size: var(--fs-caption);
    font-family: var(--ff-heading);
    text-align: center;
    margin-top: 0.3em;
  }

  pre {
    font-family: Consolas, "Courier New", monospace;
    font-size: 9.5pt;
    line-height: 1.15;
    background: #f4f4f4;
    padding: 0.4em 0.6em;
    margin: 0.4em 0 0.6em 0;
    text-indent: 0;
  }
  code { font-family: Consolas, "Courier New", monospace; font-size: 10pt; }

  blockquote {
    margin: 0.4em 0 0.6em 0;
    padding: 0.3em 0.7em;
    border-left: 2px solid var(--color-border);
    font-size: 10.5pt;
  }
  blockquote p { text-indent: 0; }

  /* 摘要卡片 */
  table.abstract-card {
    border-collapse: collapse; width: 100%%;
    margin: 0.3em 0 0.9em 0;
  }
  table.abstract-card td {
    border: 0.75px solid var(--color-border);
    padding: 0.5em 0.8em;
    font-size: 10.5pt;
    text-indent: 0;
  }
  table.abstract-card p { text-indent: 0; margin-bottom: 0.3em; }

  /* 封面 */
  .cover-no { text-align: right; font-size: var(--fs-body); margin-bottom: 3em; }
  .cover-course { text-align: center; font-family: var(--ff-heading); font-size: 14pt; margin-bottom: 2.2em; }
  .cover-title-cn { text-align: center; font-family: var(--ff-heading); font-size: 18pt; font-weight: bold; margin-bottom: 0.6em; }
  .cover-title-en { text-align: center; font-family: var(--ff-heading); font-size: 14pt; font-weight: bold; margin-bottom: 3em; }
  table.cover-info-list { border-collapse: collapse; margin: 0 auto; }
  table.cover-info-list td { border: none; padding: 0.35em 0.6em; font-size: var(--fs-body); vertical-align: top; }
  table.cover-info-list td:first-child { text-align: right; white-space: nowrap; }
  table.cover-info-list td:last-child { text-align: left; border-bottom: 1px solid var(--color-border); }
  .cover-foot { text-align: center; margin-top: 3.4em; font-size: var(--fs-body); }
""" % ROOT_CSS


def main():
    md_text = open(MD, encoding="utf-8").read()
    title_m = re.search(r"^#\s+(.+)$", md_text, re.M)
    doc_title = title_m.group(1).strip() if title_m else "实验报告"
    body, abstract_md, cover = build_body(md_text)

    abs_html = markdown.Markdown().convert(abstract_md) if abstract_md else ""
    abs_html = abs_html.replace("<p>", "<p>", 1)
    abstract_block = (
        '<table class="abstract-card"><tr><td>'
        '<p style="font-family: var(--ff-heading); font-weight: bold;">摘要</p>'
        + abs_html +
        "</td></tr></table>"
    )

    cv = lambda k, d="": _html.escape(cover.get(k, d) or d)
    cover_html = """
  <section role="cover">
    <p class="cover-no">No.&nbsp;{no}</p>
    <p class="cover-course">Project of PR and IP　模式识别与图像处理课内实验</p>
    <p class="cover-title-cn">边缘检测与线段提取</p>
    <p class="cover-title-en">Edge detection and line segment extraction</p>
    <table class="cover-info-list">
      <tr><td>Student Name</td><td>{name}</td></tr>
      <tr><td>Student ID</td><td>{sid}</td></tr>
      <tr><td>College</td><td>自动化学院 College of Automation Engineering</td></tr>
      <tr><td>Major</td><td>{major}</td></tr>
      <tr><td>Email</td><td>{email}</td></tr>
    </table>
    <p class="cover-foot">College of Automation Engineering, NUAA</p>
    <p class="cover-foot">二〇二六年 十月</p>
  </section>""".format(no=cv("No."), name=cv("Student Name"), sid=cv("Student ID"),
                       major=cv("Major"), email=cv("Email"))

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
    print("HTML written:", OUT, len(doc), "bytes")
    print("cover fields:", cover)


if __name__ == "__main__":
    main()
