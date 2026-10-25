# -*- coding: utf-8 -*-
"""选做实验报告的公式渲染：XeLaTeX + Cambria Math，编号用 Times New Roman，PIL 合成定宽画布。"""
import os
import subprocess
import shutil
import pymupdf
from PIL import Image

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "eq")
TMP = os.path.join(BASE, "_tex")
os.makedirs(TMP, exist_ok=True)

DPI = 400
TEXT_W_CM = 13.9
CANVAS_W = int(round(TEXT_W_CM / 2.54 * DPI))

EQUATIONS = [
    (1, r"\mathbf{x}' \sim \mathbf{H}\mathbf{x},\qquad \mathbf{H}\in\mathbb{R}^{3\times 3},\ \|\mathbf{H}\|_F=1"),
    (2, r"\mathbf{A}\mathbf{h}=\mathbf{0},\qquad \mathbf{h}=\operatorname*{arg\,min}_{\|\mathbf{h}\|=1}\|\mathbf{A}\mathbf{h}\|_2"),
    (3, r"E(\mathbf{x},\mathbf{x}')=\sqrt{\tfrac{1}{2}\left(d(\mathbf{x},\mathbf{H}^{-1}\mathbf{x}')^{2}+d(\mathbf{x}',\mathbf{H}\mathbf{x})^{2}\right)}"),
    (4, r"w_{\mathrm{px}}(v)=\frac{W_{\mathrm{lane}}\cos\theta\,(v-v_{\mathrm{vp}})}{h}"),
    (5, r"Z=\frac{h\,(f^{2}-pP)}{f\,w},\qquad X=\frac{(u-u_{c})\left(hP/f+Z\right)}{\sqrt{f^{2}+P^{2}}}"),
    (6, r"\mathbf{x}_{k}=\mathbf{F}\mathbf{x}_{k-1}+\mathbf{w}_{k},\qquad \mathbf{z}_{k}=\mathbf{H}\mathbf{x}_{k}+\mathbf{v}_{k}"),
    (7, r"v=3.6\sqrt{v_{X}^{2}+v_{Z}^{2}},\qquad v_{(\cdot)}=\operatorname*{arg\,min}_{a}\sum_{k}\rho\!\left(y_{k}-a\,t_{k}-b\right)"),
]

DOC = r"""\documentclass[12pt,border=1pt]{standalone}
\usepackage{amsmath,amssymb}
\usepackage{unicode-math}
\setmathfont{Cambria Math}
\setmainfont{Times New Roman}
\begin{document}
%s%s
\end{document}
"""


def render_pdf(name, body, text):
    src = os.path.join(TMP, name + ".tex")
    with open(src, "w", encoding="utf-8") as f:
        f.write(DOC % (body, text))
    subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"],
                   cwd=TMP, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pdf = os.path.join(TMP, name + ".pdf")
    if not os.path.exists(pdf):
        raise RuntimeError("xelatex failed: " + name)
    return pdf


def pdf_to_rgba(pdf):
    doc = pymupdf.open(pdf)
    pix = doc[0].get_pixmap(dpi=DPI)
    rgb = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    alpha = Image.eval(rgb.convert("L"), lambda v: 255 - v)
    out = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    out.putalpha(alpha)
    return out


def compose(num, latex):
    f = pdf_to_rgba(render_pdf("f%02d" % num, "$\\displaystyle %s$" % latex, ""))
    g = pdf_to_rgba(render_pdf("n%02d" % num, "", "(%d)" % num))
    pad = 6
    H = max(f.size[1], g.size[1]) + pad * 2
    if f.size[0] + g.size[0] > CANVAS_W - 4 * pad:
        print("WARN %02d too wide: %d+%d > %d" % (num, f.size[0], g.size[0], CANVAS_W))
    canvas = Image.new("RGBA", (CANVAS_W, H), (0, 0, 0, 0))
    canvas.alpha_composite(f, ((CANVAS_W - f.size[0]) // 2, (H - f.size[1]) // 2))
    canvas.alpha_composite(g, (CANVAS_W - g.size[0] - pad, (H - g.size[1]) // 2))
    canvas.save(os.path.join(BASE, "eq%02d.png" % num))
    print("eq%02d  %dx%d  formula=%dpx" % (num, CANVAS_W, H, f.size[0]))


if __name__ == "__main__":
    for e in EQUATIONS:
        compose(*e)
    shutil.rmtree(TMP, ignore_errors=True)
    print("done")
