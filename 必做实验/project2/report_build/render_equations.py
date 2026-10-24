# -*- coding: utf-8 -*-
"""
公式图渲染：公式本体用 **Cambria Math**（Word 的数学字体），编号用 **Times New Roman**。
路径：XeLaTeX (unicode-math) -> PDF -> PyMuPDF 400dpi -> 白底转透明 -> PIL 定宽合成
      （公式水平居中、编号贴右、垂直居中）。
"""
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
    (1,  r"Y = 0.299R + 0.587G + 0.114B"),
    (2,  r"B(i,j)=\operatorname{median}\{\,A(i+u,\,j+v):(u,v)\in\Omega_k\,\}"),
    (3,  r"G_x=\begin{bmatrix}1&0\\0&-1\end{bmatrix},\qquad G_y=\begin{bmatrix}0&1\\-1&0\end{bmatrix}"),
    (4,  r"G_{\mathrm{Robert}}=|G_x|+|G_y|"),
    (5,  r"G_x=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix},\qquad G_y=\begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}"),
    (6,  r"G_{\mathrm{Sobel}}=\sqrt{G_x^{2}+G_y^{2}}"),
    (7,  r"\nabla^{2}I=\frac{\partial^{2}I}{\partial x^{2}}+\frac{\partial^{2}I}{\partial y^{2}},\qquad K_4=\begin{bmatrix}0&1&0\\1&-4&1\\0&1&0\end{bmatrix}"),
    (8,  r"G_{\mathrm{Lap}}=\left|\nabla^{2}I\right|"),
    (9,  r"G_\sigma(x,y)=\frac{1}{2\pi\sigma^{2}}\exp\!\left(-\frac{x^{2}+y^{2}}{2\sigma^{2}}\right)"),
    (10, r"M=\sqrt{G_x^{2}+G_y^{2}},\qquad \theta=\operatorname{atan2}(G_y,\,G_x)"),
    (11, r"\sigma_b^{2}(t)=\omega_0(t)\,\omega_1(t)\,[\mu_0(t)-\mu_1(t)]^{2}"),
    (12, r"P=\frac{\left|\{\,p\in\mathrm{Pred}:d(p,\mathrm{GT})\le tol\,\}\right|}{\left|\mathrm{Pred}\right|}"),
    (13, r"R=\frac{\left|\{\,q\in\mathrm{GT}:d(q,\mathrm{Pred})\le tol\,\}\right|}{\left|\mathrm{GT}\right|}"),
    (14, r"F_1=\frac{2PR}{P+R}"),
    (15, r"\rho=x\cos\theta+y\sin\theta"),
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
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"],
                       cwd=TMP, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pdf = os.path.join(TMP, name + ".pdf")
    if not os.path.exists(pdf):
        raise RuntimeError("xelatex failed: " + name)
    return pdf


def pdf_to_rgba(pdf):
    doc = pymupdf.open(pdf)
    pix = doc[0].get_pixmap(dpi=DPI)
    rgb = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    g = rgb.convert("L")
    alpha = Image.eval(g, lambda v: 255 - v)
    out = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    out.putalpha(alpha)
    return out


def compose(num, latex):
    f = pdf_to_rgba(render_pdf("f%02d" % num, "$\\displaystyle %s$" % latex, ""))
    gimg = pdf_to_rgba(render_pdf("n%02d" % num, "", "(%d)" % num))

    pad = 6
    H = max(f.size[1], gimg.size[1]) + pad * 2
    if f.size[0] + gimg.size[0] > CANVAS_W - 4 * pad:
        print("WARN %02d 过宽: %d+%d > %d" % (num, f.size[0], gimg.size[0], CANVAS_W))

    canvas = Image.new("RGBA", (CANVAS_W, H), (0, 0, 0, 0))
    canvas.alpha_composite(f, ((CANVAS_W - f.size[0]) // 2, (H - f.size[1]) // 2))
    canvas.alpha_composite(gimg, (CANVAS_W - gimg.size[0] - pad, (H - gimg.size[1]) // 2))
    canvas.save(os.path.join(BASE, "eq%02d.png" % num))
    print("eq%02d  %dx%d  formula=%dpx  canvas=%.2fcm"
          % (num, CANVAS_W, H, f.size[0], CANVAS_W / DPI * 2.54))


if __name__ == "__main__":
    for e in EQUATIONS:
        compose(*e)
    shutil.rmtree(TMP, ignore_errors=True)
    print("done")
