
基于 Dify 工作流的医学基准评测自动化流水线，支持从论文中提取方法、自动生成评测代码，对 Medbench 2025 进行评测。

---

## 项目结构

```
.
├── dify_exactor_method.py          # 上传 PDF → 提取论文方法 → 保存至 Excel
├── dify_auto_code.py               # 读取提取的方法 → 通过 Dify 自动生成评测代码
├── medbench_eval.py                # Medbench 直接匹配评测（baseline）
├── medbench_eval_LLM_as_judge.py   # LLM-as-a-Judge 评测（单模型 vs 标准答案，含 swap）
├── requirements.txt
└── README.md
```

---

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖包：`openai >= 1.0.0`、`requests >= 2.28.0`、`openpyxl >= 3.1.0`

---

## 配置凭据

运行前，在每个脚本顶部的 **Configuration** 区域填写你自己的凭据：

**Dify 脚本**（`dify_exactor_method.py`、`dify_auto_code.py`）

```python
BASE_URL = "http://YOUR_DIFY_HOST/v1"   # 替换为你的 Dify 服务地址
API_KEY  = "your-dify-api-key"          # 替换为你的 Dify 应用 API Key
```

**评测脚本**（`medbench_eval.py`、`medbench_eval_LLM_as_judge.py`、）

```python
API_KEY  = "your-openai-compatible-api-key"                       # 替换为你的 LLM API Key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"   # 或替换为你的 API 端点
```

---

## 完整使用流程

### Step 1 — 上传论文 PDF，运行 Dify 工作流提取方法

```bash
python dify_exactor_method.py --file paper.pdf --output dify_results.xlsx
```

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--file` | `-f` | `example.pdf` | 要上传的本地文件路径 |
| `--type` | `-t` | `document` | Dify 文档类型：`document` / `image` / `audio` / `video` |
| `--var` | `-v` | `file` | Dify 工作流中接收文件的变量名 |
| `--output` | `-o` | `dify_results.xlsx` | 保存结果的 Excel 文件路径 |
| `--debug` | `-d` | `False` | 打印请求 payload 用于调试 |

输出 Excel 包含列：`timestamp`、`file_path`、`run_id`，以及工作流输出的所有字段（如 `method_llm`）。

---

### Step 2 — 读取提取结果，通过 Dify 自动生成评测代码

```bash
python dify_auto_code.py \
    --input dify_results.xlsx \
    --ref_code medbench_eval.py \
    --output dify_auto_code_results.xlsx \
    --code_dir generated_code/
```

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | `dify_results.xlsx` | Step 1 输出的 Excel，需含 `method_llm` 和 `file_path` 列 |
| `--ref_code` | `-r` | `medbench_eval.py` | 作为参考的基准代码文件，随 `paper_method` 一起送入工作流 |
| `--output` | `-o` | `dify_auto_code_results.xlsx` | 保存代码生成结果的 Excel |
| `--code_dir` | `-c` | `./` | 生成的 `.py` 文件保存目录 |
| `--debug` | `-d` | `False` | 打印请求详情用于调试 |

每行 Excel 记录对应一个论文 PDF，脚本会：
1. 取 `method_llm` 字段作为 `paper_method` 输入
2. 读取 `--ref_code` 文件内容作为 `reference_code` 输入
3. 调用 Dify 工作流生成评测代码
4. 将代码保存为 `{pdf_name}_medbench_eval.py`

---

### Step 3 — Medbench 直接匹配评测（baseline）

```bash
python medbench_eval.py \
    --input_dir Medbench_2025_direct_match \
    --output_dir eval_results_qwen-plus \
    --model qwen-plus \
    --workers 8
```

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_dir` | `-i` | `Medbench_2025_direct_match` | 包含 `.jsonl` 评测文件的目录 |
| `--output_dir` | `-o` | `eval_results__qwen-plus` | 保存每条预测结果和汇总的目录 |
| `--model` | `-m` | `qwen-plus` | 被评测的模型名称 |
| `--workers` | `-w` | `8` | 并发线程数 |
| `--limit` | `-n` | `0`（全量） | 每个文件只处理前 N 条，0 表示全部 |
| `--files` | — | 全部 `.jsonl` | 指定只运行哪些文件，如 `--files exam_a.jsonl exam_b.jsonl` |

每个 `.jsonl` 对应生成一个 `{dataset}_results.jsonl`，并在 `--output_dir` 下保存 `summary.json`：

```json
{
  "model": "qwen-plus",
  "results": [...],
  "overall": { "total": 500, "correct": 410, "errors": 0, "accuracy": 0.82 }
}
```

**断点续跑**：脚本自动读取已有输出文件，跳过已处理的题目 ID，可安全中断后重新运行。

---

### Step 4 — LLM-as-a-Judge 评测（单模型 vs 标准答案）

```bash
python medbench_eval_LLM_as_judge.py \
    --input_dir Medbench_2025_direct_match \
    --output_dir eval_results_judge \
    --model qwen-plus \
    --judge_model qwen-plus \
    --workers 8
```

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_dir` | `-i` | `Medbench_2025_direct_match` | 包含 `.jsonl` 评测文件的目录 |
| `--output_dir` | `-o` | `eval_results_llm_judge_pairwise_swap_qwen-plus` | 结果保存目录 |
| `--model` | `-m` | `qwen-plus` | 被评测模型（生成答案） |
| `--judge_model` | `-j` | `qwen-plus` | 裁判模型（比较答案质量） |
| `--workers` | `-w` | `8` | 并发线程数 |
| `--limit` | `-n` | `0` | 每文件限制前 N 条，0 为全部 |
| `--files` | — | 全部 `.jsonl` | 指定部分文件运行 |

每条结果包含：`model_response`、`comparison_result`（A/B/TIE）、`judgment_1_raw`、`judgment_2_raw`、`llm_judge_correct`、`direct_match_correct`。

---

## LLM-as-a-Judge 方法说明

评测方法参考：

> **"From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge"**
> EMNLP 2025 Survey

核心机制：

- **Pairwise 比较**：将两个回答同时提交给裁判模型，要求输出 `A` / `B` / `TIE`
- **Swap 操作（位置去偏）**：对同一对回答做两次判断（正序 + 换序），只有两次结论一致时才采纳，否则记为 `TIE`
- **Rule-Augmented Prompting**：提示词中嵌入明确评分标准（准确性、完整性、清晰度、相关性、安全性）
- **Chain-of-Thought**：要求裁判模型逐步分析后再给出结论

---

## 输入数据格式

`Medbench_2025_direct_match/` 目录下的每个 `.jsonl` 文件，每行一条题目：

```json
{
  "question": "下列哪项属于 ...",
  "answer": "<A>",
  "other": {
    "id": "12345",
    "source": "dataset_name"
  }
}
```

答案格式约定：

| 题型 | 格式示例 | 说明 |
|------|----------|------|
| 单选题 | `<A>` | 单个选项，用尖括号包围 |
| 多选题 | `<A,C,E>` | 多个选项按字母排序，逗号分隔 |
| 填空题 | `词语1\|词语2` | 多个填空项排序后用 `\|` 分隔 |

---

## TODO

### TODO 1 — 自动收集论文

从 arXiv、PubMed、Semantic Scholar 等学术网站批量抓取最新论文 PDF，补充为 Dify 工作流的输入数据源。

```
# arXiv API 示例
https://export.arxiv.org/api/query?search_query=medical+LLM+evaluation&max_results=50

# Semantic Scholar API 示例
https://api.semanticscholar.org/graph/v1/paper/search?query=clinical+QA+benchmark&fields=title,year,externalIds
```

---

### TODO 2 — 从论文中整理可落地的改进 Idea

在 Step 1 提取完论文方法后，增加一个 Idea 整理环节：对所有论文的 `method_llm` 字段进行汇总分析，提炼出可用于提升当前评测流程的具体想法。

**计划步骤：**

- [ ] 汇总 `dify_results.xlsx` 中所有论文的 `method_llm` 字段内容
- [ ] 设计专用 Dify 工作流（或直接调用 LLM），输入多篇论文方法摘要，输出结构化 Idea 列表
