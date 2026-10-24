# report_build —— 实验报告的排版构建链

`必做实验/project2/` 下的 `实验报告_Project2.docx` / `.pdf` 由本目录的脚本生成。
正文内容来自 `src/` 的真实实验结果，本目录只负责**把内容排成符合《实验指导书》格式的 Word 文档**。

## 目录

```
report_build/
├── report_source.md        报告源文件（标准 Markdown，公式用 $$...\tag{n}$$ 标记）
├── design_tokens.json      排版令牌（字号/字体/行距的唯一定义处）
├── images/
│   ├── fig00_pipeline.png  总体流程图（本目录自产的插图）
│   └── eq/eq01–eq15.png    Cambria Math 公式图（脚本生成）
├── build_html.py           md → 排版 HTML（封面/摘要/三线表/图表题/公式居中）
├── render_equations.py     公式 → 透明 PNG（XeLaTeX + Cambria Math）
├── postfix_docx.py         转换后校准（字体/间距/防跨页）
└── formatted_report.html   中间产物（可删，重跑即得）
```

原始实验插图仍复用 `../results/figures/`，不在此重复存放。

## 依赖

- Python 3.12：`python-docx`、`pymupdf`、`Pillow`、`markdown`
- **TeX Live**（含 `xelatex`）：公式渲染
- `html-to-docx`（腾讯文档 DOCX 插件的转换引擎）

## 复现

```bash
cd 必做实验/project2/report_build

python render_equations.py                 # 1. 生成 images/eq/*.png
python build_html.py                       # 2. report_source.md -> formatted_report.html
python -m html_to_docx convert formatted_report.html -o ../实验报告_Project2.docx   # 3. HTML -> DOCX
python postfix_docx.py                     # 4. 转换后校准（必须执行，见下）
```

第 4 步不是可选项——`html-to-docx` 有几处已知行为，不校准会出问题。

## 格式对照

| 要求（指导书） | 实现方式 |
|---|---|
| 一级标题 四号黑体 | `Heading 1/2` = 黑体 14 pt（编号章节用 H2） |
| 二级标题 小四黑体 | `Heading 3/4` = 黑体 12 pt |
| 正文 小四宋体 | `Normal` = 宋体 12 pt，西文 Times New Roman |
| 1.2 倍行距 | `line_spacing = 1.2` |
| 公式居中、编号靠右 | 公式渲染时把编号画进图片右端，整幅图水平居中 |
| 图、表标题居中 | 图题/表题为独立居中段落（`<caption>` 不被支持） |
| 三线表 | 1.5 / 0.75 / 1.5 pt，无竖线 |

## 已知坑（html-to-docx）

1. **只认 `<img width="px">` 属性**，不认内联 CSS 的 `width`；且会把宽度截断到约 13.97 cm。
   `px ÷ 37.795 = cm`。
2. **不支持 `<caption>`**，表题必须写成表格前的居中 `<p>`。
3. **run 级字体会被统一写成 body 字体**，标题会丢黑体 —— `postfix_docx.py` 逐 run 覆盖。
4. `Heading 4` 自带斜体，需显式关掉。

## 为什么公式是图片

`html-to-docx` 不产出 OMML，而 `HTML→DOCX` 也没有可用的公式通道。
因此公式用 **XeLaTeX + unicode-math + `\setmathfont{Cambria Math}`** 渲染成透明 PNG
（编号用 Times New Roman），再以定宽画布合成：公式水平居中、编号贴右。
这样既保证了 Cambria Math 的字形，也保证了"居中 + 编号右对齐"的版式。
