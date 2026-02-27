# Med-Code: Automated Medical Benchmark Evaluation with LLM-as-a-Judge

An automated pipeline built on Dify workflows for extracting methods from research papers, auto-generating evaluation code, and evaluating LLMs on Medbench 2025 .

> 📖 [中文文档 README.zh.md](./README.zh.md)

---

## Project Structure

```
.
├── dify_exactor_method.py           # Upload PDF → extract paper method → save to Excel
├── dify_auto_code.py                # Read extracted methods → auto-generate eval code via Dify
├── medbench_eval.py                 # Medbench direct-match evaluation (baseline)
├── medbench_eval_LLM_as_judge.py    # LLM-as-a-Judge eval (single model vs ground truth, with swap)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `openai >= 1.0.0`, `requests >= 2.28.0`, `openpyxl >= 3.1.0`

---

## Configuration

Before running any script, fill in your credentials in the **Configuration** section at the top of each file:

**Dify scripts** (`dify_exactor_method.py`, `dify_auto_code.py`)

```python
BASE_URL = "http://YOUR_DIFY_HOST/v1"   # Your Dify server address
API_KEY  = "your-dify-api-key"          # Your Dify application API key
```

**Evaluation scripts** (`medbench_eval.py`, `medbench_eval_LLM_as_judge.py`, `LLM-as-a-judge_medbench_eval.py`)

```python
API_KEY  = "your-openai-compatible-api-key"                       # Your LLM provider API key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"   # Or your custom endpoint
```

---

## Full Pipeline

### Step 1 — Upload a paper PDF and extract the method via Dify

```bash
python dify_exactor_method.py --file paper.pdf --output dify_results.xlsx
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--file` | `-f` | `example.pdf` | Path to the local file to upload |
| `--type` | `-t` | `document` | Dify document type: `document` / `image` / `audio` / `video` |
| `--var` | `-v` | `file` | Workflow input variable name for the file |
| `--output` | `-o` | `dify_results.xlsx` | Excel file path to save results |
| `--debug` | `-d` | `False` | Print request payload for debugging |

The output Excel contains: `timestamp`, `file_path`, `run_id`, and all workflow output fields (e.g. `method_llm`).

---

### Step 2 — Auto-generate evaluation code via Dify

```bash
python dify_auto_code.py \
    --input dify_results.xlsx \
    --ref_code medbench_eval.py \
    --output dify_auto_code_results.xlsx \
    --code_dir generated_code/
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | `dify_results.xlsx` | Excel from Step 1, must contain `method_llm` and `file_path` columns |
| `--ref_code` | `-r` | `medbench_eval.py` | Reference code file sent alongside `paper_method` to the workflow |
| `--output` | `-o` | `dify_auto_code_results.xlsx` | Excel to save code generation results |
| `--code_dir` | `-c` | `./` | Directory to save generated `.py` files |
| `--debug` | `-d` | `False` | Print request details for debugging |

For each Excel row the script will:
1. Use the `method_llm` field as `paper_method` input
2. Read the `--ref_code` file content as `reference_code` input
3. Call the Dify workflow to generate evaluation code
4. Save the code as `{pdf_name}_medbench_eval.py`

---

### Step 3 — Medbench direct-match evaluation (baseline)

```bash
python medbench_eval.py \
    --input_dir Medbench_2025_direct_match \
    --output_dir eval_results_qwen-plus \
    --model qwen-plus \
    --workers 8
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input_dir` | `-i` | `Medbench_2025_direct_match` | Directory containing `.jsonl` evaluation files |
| `--output_dir` | `-o` | `eval_results__qwen-plus` | Directory to save per-item results and summary |
| `--model` | `-m` | `qwen-plus` | Model name to evaluate |
| `--workers` | `-w` | `8` | Number of concurrent threads |
| `--limit` | `-n` | `0` (all) | Process only the first N items per file; 0 means all |
| `--files` | — | all `.jsonl` | Run only specific files, e.g. `--files exam_a.jsonl exam_b.jsonl` |

Each `.jsonl` produces a `{dataset}_results.jsonl`, and a `summary.json` is saved under `--output_dir`:

```json
{
  "model": "qwen-plus",
  "results": [...],
  "overall": { "total": 500, "correct": 410, "errors": 0, "accuracy": 0.82 }
}
```

**Resume support**: The script automatically reads existing output files and skips already-processed item IDs. It is safe to interrupt and re-run at any time.

---

### Step 4 — LLM-as-a-Judge evaluation (single model vs ground truth)

```bash
python medbench_eval_LLM_as_judge.py \
    --input_dir Medbench_2025_direct_match \
    --output_dir eval_results_judge \
    --model qwen-plus \
    --judge_model qwen-plus \
    --workers 8
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input_dir` | `-i` | `Medbench_2025_direct_match` | Directory containing `.jsonl` evaluation files |
| `--output_dir` | `-o` | `eval_results_llm_judge_pairwise_swap_qwen-plus` | Directory to save results |
| `--model` | `-m` | `qwen-plus` | Model being evaluated (generates answers) |
| `--judge_model` | `-j` | `qwen-plus` | Judge model (compares answer quality) |
| `--workers` | `-w` | `8` | Number of concurrent threads |
| `--limit` | `-n` | `0` | Process only first N items per file; 0 means all |
| `--files` | — | all `.jsonl` | Run only specific files |

Each result record contains: `model_response`, `comparison_result` (A/B/TIE), `judgment_1_raw`, `judgment_2_raw`, `llm_judge_correct`, `direct_match_correct`.

---


## LLM-as-a-Judge Methodology

The judge evaluation follows the methodology from:

> **"From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge"**
> EMNLP 2025 Survey

Key mechanisms implemented:

- **Pairwise comparison**: both responses are submitted to the judge model simultaneously; the model outputs `A`, `B`, or `TIE`
- **Swapping operation (position de-bias)**: each pair is judged twice (original order + swapped order); a verdict is accepted only when both runs agree, otherwise the result is `TIE`
- **Rule-augmented prompting**: evaluation criteria are embedded directly in the prompt (accuracy, completeness, clarity, relevance, safety)
- **Chain-of-Thought**: the judge model is required to reason step-by-step before giving a final verdict

---

## Input Data Format

Each `.jsonl` file under `Medbench_2025_direct_match/` contains one question per line:

```json
{
  "question": "Which of the following ...",
  "answer": "<A>",
  "other": {
    "id": "12345",
    "source": "dataset_name"
  }
}
```

Answer format conventions:

| Question type | Format | Description |
|---------------|--------|-------------|
| Single-choice | `<A>` | Single option wrapped in angle brackets |
| Multi-choice | `<A,C,E>` | Multiple options sorted alphabetically, comma-separated |
| Fill-in-the-blank | `term1\|term2` | Multiple answers sorted and separated by `\|` |

---

## TODO

### TODO 1 — Automated paper collection

Batch-crawl the latest papers on arXiv, PubMed, Semantic Scholar, and feed them as PDF inputs to the Dify extraction workflow.

**Reference APIs:**

```
# arXiv API
https://export.arxiv.org/api/query?search_query=medical+LLM+evaluation&max_results=50

# Semantic Scholar API
https://api.semanticscholar.org/graph/v1/paper/search?query=clinical+QA+benchmark&fields=title,year,externalIds
```

---

### TODO 2 — Distill improvement ideas from collected papers

After Step 1 extracts the method descriptions, add an idea-mining stage: aggregate all `method_llm` entries and use an LLM to produce a structured list of actionable improvement ideas for the current pipeline.

**Planned steps:**

- [ ] Aggregate all `method_llm` fields from `dify_results.xlsx`
- [ ] Design a dedicated Dify workflow (or call an LLM directly) that takes multiple paper-method summaries as input and outputs a structured idea list
