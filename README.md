# LLM Cultural Alignment Pipeline: Response Generation and Justification Analysis
![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Status](https://img.shields.io/badge/Status-Research_Prototype-green) 
![Focus](https://img.shields.io/badge/Focus-Bilingual_Inference_%2B_Justification_Analysis-orange)

## Abstract

This repository hosts a research pipeline for computational social science studies on cross-cultural alignment in Large Language Models (LLMs). It retains the original **Batch Inference Orchestrator** for bilingual response generation and also includes a formal **justification analysis module** for the same political questions and model outputs.

The project therefore has two connected layers:

1. **Response Generation Pipeline** for bilingual political prompting and model response collection
2. **Justification Analysis Module** for downstream comparison of justification patterns across:
   - Chinese-origin vs Western-origin models
   - English vs Chinese input language
   - pressure vs no-pressure condition

The main justification-analysis result layers are:

- `G1`: semantic drift under pressure
- `G3`: framing sensitivity
- `G4`: cross-lingual reframing / consistency
- `G5`: rhetorical compression

`G2` remains exploratory only and is not used for the main claims.

## ✨ Key Features

### Response Generation

- **High-Concurrency Architecture**: Supports asynchronous batch inference across heterogeneous LLMs.
- **Bilingual Alignment Support**: Native handling of paired prompt fields for English and Chinese comparative studies.
- **Robustness**: Retry logic, intermediate saving, and data sanitization for long-running inference jobs.
- **Model-Agnostic Routing**: Compatible with multiple proprietary and open-weight models through a unified API-style interface.

### Justification Analysis

- **Same-Question Comparative Design**: Uses the same political questions to compare justification differences rather than only raw responses.
- **Conditioned Comparison**: Tracks model origin group, language condition, and pressure condition in one analysis workflow.
- **Structured Result Layers**: Produces grouped summaries and figures for `G1`, `G3`, `G4`, and `G5`.
- **Research-Oriented Outputs**: Separates main-result tables and figures from diagnostic, QA, and legacy materials.

## System Architecture

```text
.
+-- README.md
+-- llm_aihubmix.py                          # Core response-generation engine
+-- requirements.txt                        # Dependency manifest
+-- Data Sample/                            # Sample generation inputs / outputs
+-- justification_analysis/
    +-- pipeline/
    |   +-- 01_run_core_from_rdata.py
    |   +-- 02_postprocess_final_tables.py
    |   +-- 03_build_css_group_tables.py
    |   |-- 04_plot_css_main_results.py
    +-- config/
    |   +-- config_justification.json
    |   +-- model_metadata.csv
    |   |-- AllQuestions.xlsx (individual)
    +-- results/
    |   +-- main_tables/
    |   +-- main_figures/
    +-- justification_utils.py
    +-- justification_config.py
    |-- g4_strict.py
```

At the GitHub presentation level, the repository should be read as a single research repo with:

- an upstream **response generation** layer
- a downstream **justification analysis** layer

## Getting Started

### Prerequisites

- Python 3.8+
- A valid API key from Aihubmix or another compatible OpenAI-format provider for response generation

### Installation

Clone the repository:

```bash
git clone https://github.com/Esther016/LLM-Cultural-Alignment-Pipeline.git
cd LLM-Cultural-Alignment-Pipeline
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Security Configuration

Create a local `.env` file for API credentials and do not commit it:

```env
AIHUBMIX_API_KEY=sk-xxxxxxxxxxxxxxxxx
```

## Usage Pipeline

### 1. Response Generation Pipeline

The original generation layer remains the first stage of the project.

Core files and folders:

- `llm_aihubmix.py`
- `requirements.txt`
- `Data Sample/`
- `outputs/`

Prepare an input file with bilingual prompt fields and run:

```bash
python llm_aihubmix.py 0.7
```

This stage produces structured model-response outputs, typically with paired columns such as:

- `{Model_Name}` for English responses
- `{Model_Name}_CN` for Chinese responses

**Data Preparation**

Prepare an input Excel file (default: `AllQuestions.xlsx`) with the following schema to support comparative studies:  

| Prompt               | Prompt_CN           | Question             | Question_CN          |  
|----------------------|---------------------|----------------------|----------------------|  
| System instruction...| 系统指令...         | Input question...    | 输入问题...          |  

**Execution**

Run the orchestrator with a specified temperature parameter to control generation stochasticity:  
```bash
# Syntax: python script_name.py [temperature]
python llm_aihubmix.py 0.7
```  
- **Temperature**: Float (0.0–1.0). Lower values (e.g., 0.2) for factual extraction; higher values (e.g., 0.7) for creative/ideological simulations.  

**Output Analysis**

Results are automatically aggregated into the `outputs/` directory. The engine generates comparative columns for each model:  
- `{Model_Name}`: English Response  
- `{Model_Name}_CN`: Chinese Response

The sample result is in the folder `Data Sample`

<img src="https://github.com/Esther016/LLM-Cultural-Alignment-Pipeline/blob/main/image/sample-outcome.png?raw=true" alt="sample-outcome-screenshot" width="800">

#### ⚙️ Advanced Configuration  

Researchers can fine-tune pipeline parameters directly in the script for experimental customization:  

| Parameter       | Default | Description                                                                 |  
|-----------------|---------|-----------------------------------------------------------------------------|  
| `MODELS`        | [List]  | Array of model identifiers to test (e.g., `gpt-4o`, `claude-3-5-sonnet`).   |  
| `BATCH_SIZE`    | 20      | Number of rows processed before forcing a disk save.                        |  
| `MAX_WORKERS`   | 5       | Thread pool size. Increase for higher throughput (monitor API rate limits).  |  
| `TIMEOUT`       | 120s    | Max wait time per API call before triggering retry logic.                   |  

---

### 2. Justification Analysis Module

The second stage is the downstream justification analysis module for the same political questions.

Main analysis pipeline:

- `justification_analysis/pipeline/01_run_core_from_rdata.py`
- `justification_analysis/pipeline/02_postprocess_final_tables.py`
- `justification_analysis/pipeline/03_build_css_group_tables.py`
- `justification_analysis/pipeline/04_plot_css_main_results.py`

Supporting files:

- `justification_analysis/config/config_justification.json`
- `justification_analysis/config/model_metadata.csv`
- `justification_analysis/config/AllQuestions.xlsx`
- `justification_analysis/justification_utils.py`
- `justification_analysis/justification_config.py`
- `justification_analysis/g4_strict.py`

Recommended execution flow:

```bash
python justification_analysis/pipeline/01_run_core_from_rdata.py --config justification_analysis/config/config_justification.json
python justification_analysis/pipeline/02_postprocess_final_tables.py --config justification_analysis/config/config_justification.json
python justification_analysis/pipeline/03_build_css_group_tables.py --config justification_analysis/config/config_justification.json --model-metadata justification_analysis/config/model_metadata.csv --question-metadata justification_analysis/config/AllQuestions.xlsx
python justification_analysis/pipeline/04_plot_css_main_results.py --config justification_analysis/config/config_justification.json
```

Conceptually, the full workflow is:

`bilingual prompting -> model responses -> justification-derived input -> core analysis -> postprocessing -> grouped summary tables -> main figures`

## Main Findings

For GitHub presentation, only the main analysis layers should be surfaced in the README:

- `G1`: pressure-induced semantic drift
- `G3`: framing sensitivity
- `G4`: cross-lingual reframing / consistency
- `G5`: rhetorical compression / style drift

Recommended main figures:

- `justification_analysis/results/main_figures/g1_drift_by_lang_origin_group_groupedbar.png`
- `justification_analysis/results/main_figures/g4_translation_vs_reframing_by_origin_group.png`
- `justification_analysis/results/main_figures/g4_crosslingual_inconsistency_by_origin_group.png`
- `justification_analysis/results/main_figures/g4_translation_reframing_slope_by_origin_group.png`
- `justification_analysis/results/main_figures/g5_length_compression_lang_origin_group_faceted.png`

Optional supporting figure:

- `justification_analysis/results/main_figures/g3_abs_shift_axis_lang_origin_group_faceted.png`

Main summary tables to reference:

- `justification_analysis/results/main_tables/g1_by_lang_origin_group_summary.csv`
- `justification_analysis/results/main_tables/g3_by_axis_lang_origin_group_summary.csv`
- `justification_analysis/results/main_tables/g4_by_origin_group_summary.csv`
- `justification_analysis/results/main_tables/g5_by_metric_lang_origin_group_summary.csv`

`qa/`, `archive/`, legacy materials, and exploratory `G2` outputs should not be presented as main findings.

## ⚙️ Advanced Configuration

### Response Generation

Researchers may still tune generation-stage parameters in the orchestration layer, including model lists, batching strategy, worker count, and timeout settings.

### Justification Analysis

The analysis-stage configuration is managed through:

- `justification_analysis/config/config_justification.json`
- `justification_analysis/config/model_metadata.csv`
- `justification_analysis/config/AllQuestions.xlsx`

This layer controls analysis paths, thresholds, metadata joins, grouped summaries, and figure generation, while keeping the statistical logic of the main pipeline intact.

## Dataset & Privacy Note

This repository should be understood as a research repository that includes:

- a bilingual response-generation framework
- a justification-analysis module
- selected main-result tables and figures

It should not be interpreted as a guarantee that the full raw response archive or all intermediate study data are publicly redistributed.

The justification-analysis stage depends on processed or derived inputs built from response data. Sample materials may be included for demonstration and transparency, but not all source materials need to be public.

## Contribution

This repository was developed for research on cross-cultural LLM alignment and justification-level political text analysis. Contributions that improve orchestration robustness, reproducibility, result presentation, or analysis usability are welcome.

## Disclaimer

This project is intended for academic research purposes. Users are responsible for complying with the terms of service and usage policies of the respective model providers.

---

*Designed for reproducible computational social science, from bilingual LLM response generation to justification-level comparative analysis.*
