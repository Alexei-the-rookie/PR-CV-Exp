# report_build —— 选做实验研究报告的排版构建链

`选做实验/选做实验报告_PRCV.docx` / `.pdf` 由本目录的脚本生成。
正文内容取自 `../task1_stitching/` 与 `../task2_traffic/` 的真实运行结果（指标、图、日志），
本目录只负责**把内容排成符合 `Research Report_Template.docx` 的 Word 文档**。

## 目录

```
report_build/
├── report_source.md        报告源文件（Markdown，公式用 $$...\tag{n}$$ 标记）
├── images/eq/eq01–eq07.png Cambria Math 公式图（脚本生成，已随仓库提供）
├── build_html.py           md → 排版 HTML（封面/摘要/三线表/图表题/公式居中）
├── render_equations.py     公式 → 透明 PNG（XeLaTeX + unicode-math + Cambria Math）
├── postfix_docx.py         转换后校准（字体/间距/防跨页）
└── formatted_report.html   中间产物（已 gitignore，重跑即得）
```

插图不在此重复存放：`build_html.py` 里的 `FIG_ALT` 把报告中的图直接指向
`../task1_stitching/results/figures/` 与 `../task2_traffic/results/figures/`，省去约 9 MB 冗余。

## 依赖

- Python 3.12：`python-docx`、`pymupdf`、`Pillow`、`markdown`
- **TeX Live**（含 `xelatex`）：公式渲染；需系统装有 `Cambria Math` 字体
- `html-to-docx`（腾讯文档 DOCX 插件的转换引擎）

## 复现

```bash
cd 选做实验/report_build

python render_equations.py                 # 1. 生成 images/eq/*.png
python build_html.py                       # 2. report_source.md -> formatted_report.html
python -m html_to_docx convert formatted_report.html -o ../选做实验报告_PRCV.docx   # 3. HTML -> DOCX
python postfix_docx.py                     # 4. 转换后校准（必须执行，见下）
```

第 4 步不是可选项——`html-to-docx` 有几处已知行为，不校准会出问题。

## 格式对照（模板要求）

| 要求 | 实现方式 |
|---|---|
| 正文 Times New Roman 10 pt | `Normal` = 10 pt，西文 TNR、中文宋体 |
| 1.5 倍行距 | `line_spacing = 1.5` |
| 图题 `Fig.N` / 表题 `Tab.N` | 独立居中段落（`<caption>` 不被转换器支持） |
| 三线表 | 1.5 / 0.75 / 1.5 pt，无竖线 |
| 公式居中、编号靠右 | 渲染时把编号画进图片右端，整幅图居中 |
| References 15+ | 共 16 条，正文 `[n]` 与文献表一一对应 |

## 已知坑（html-to-docx）

1. **只认 `<img width="px">` 属性**，不认内联 CSS 的 `width`；且会把宽度截断到约 13.97 cm。`px ÷ 37.795 = cm`。
2. **不支持 `<caption>`**，表题必须写成表格前的居中 `<p>`。
3. **run 级字体会被统一写成 body 字体** —— `postfix_docx.py` 逐 run 覆盖。
4. `Heading 4` 自带斜体，需显式关掉。
5. **行内公式 `$...$` 不认**，会原样漏出 LaTeX 源码 —— `build_html.py` 里的 `inline_tex()` 自行转 HTML。
6. 图注**不要**设 `keep_with_next`：图片段已粘住图注，再粘下一段会把整块顶到下一页、留半页空白。

## 为什么公式是图片

`html-to-docx` 不产出 OMML，HTML→DOCX 也没有可用的公式通道。
公式用 **XeLaTeX + unicode-math + `\setmathfont{Cambria Math}`** 渲染成透明 PNG
（编号用 Times New Roman），再按定宽画布合成：公式水平居中、编号贴右。
