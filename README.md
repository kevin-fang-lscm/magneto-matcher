<h2 align="center">Magneto: Combining Small and Large Language Models for
Schema Matching</h2>

> Welcome to Magneto!

This repository contains the codebase of our paper "[Magneto: Combining Small and Large Language Models for Schema Matching](https://arxiv.org/abs/2412.08194)".

Magneto is an innovative framework designed to enhance schema matching (SM) by intelligently combining small, pre-trained language models (SLMs) with large language models (LLMs). Our approach is structured to be both cost-effective and broadly applicable.

The framework operates in two distinct phases:
- **Candidate Retrieval**: This phase involves using SLMs to quickly identify a manageable subset of potential matches from a vast pool of possibilities. Optional LLM-powered fine-tuning can be performed.
- **Match Reranking**: In this phase, LLMs take over to assess and reorder the candidates, simplifying the process for users to review and select the most suitable matches.

## Contents

This README file is divided into the following sections:

* [1. Environment Setup](#gear-1-environment-setup)
* [2. Code Structure](#gear-2-code-structure)
* [3. Example Usage](#gear-3-example-usage)

## :gear: 1. Environment Setup

### 🔥 1.1 Create a virtual environment
This step is optional but recommended. To isolate dependencies and avoid library conflicts with your local environment, you may want to use a Python virtual environment manager. To do so, you should run the following commands to create and activate the virtual environment:
```bash
python -m venv ./venv
source ./venv/bin/activate
```

### 🔥 1.2 Install dependencies

To install the required dependencies, run the following command:
```bash
pip install -r requirements.txt
```

### 🔥 1.3 Data Preparation

The data folder contains the datasets used for data integration tasks. Download the data folder from [this Google Drive link](https://drive.google.com/drive/folders/19kCWQI0CWHs1ZW9RQEUSeK6nuXoA-5B7?usp=sharing) and place it in the `data` directory. Contents include:
- **`gdc`**: GDC benchmark from the paper. Contains ten tumor analysis study datasets to be matched to Genomics Data Commons (GDC) standards.
- **`Valentine-datasets`**: Schema matching benchmark from [Valentine paper](https://delftdata.github.io/valentine/) (also available on [Zenodo](https://zenodo.org/records/5084605#.YOgWHBMzY-Q): DOI 10.5281/zenodo.5084605).
- **`synthetic`**: Synthetic data generated using `llm-aug` and `struct-aug` for LLM-based fine-tuning. You can use the provided JSON files directly or regenerate by modifying the underlying LLM model and other configurations in the [code](https://github.com/VIDA-NYU/data-integration-eval/blob/main/algorithms/magneto/finetune/data_generation/synthetic_data_gen.py). Processed data for synthetic match generation is located in the same folder under `unique_columns` directory.

### 🔥 1.4 Download the fine-tuned model for GDC benchmark

This step is optional but required for `MagnetoFT` and `MagnetoFTGPT`. Download the fine-tuned model of your choice from [this Google Drive link](https://drive.google.com/drive/folders/1vlWaTm4rpEH4hs-Kq3mhSfTyffhDEp6P?usp=sharing) and place it in the `models` directory.

### 🔥 1.5 Set the Environment Variable
This step is optional but required for `MagnetoGPT` and `MagnetoFTGPT`. Set the `OPENAI_API_KEY` environment variable using the following commands based on your operating system:
#### For Windows:
```bash
set OPENAI_API_KEY=your_openai_api_key_here
```
#### For macOS/Linux:
```bash
export OPENAI_API_KEY=your_api_key_here
```
To use `LLaMA3.3` as the LLM reranker, you can also set up `LLAMA_API_KEY` accordingly.

## :gear: 2. Code Structure
> note that batched benchmark on baseline methods are on this [repo](https://github.com/VIDA-NYU/data-harmonization-benchmark).

```bash
|-- algorithm
    |-- magneto # code for Magneto
        |-- finetune # code for Magneto FT
        |-- magneto # Magneto core
    |-- gpt_matcher # code for GPT-based matcher
        |-- gpt_matcher.py # GPT-based matcher core
    |-- topk_metrics.py # Introducing Recall @ topk
|-- experiments
    |-- ablations # code for ablation study
        |-- run_bp_gdc.py # ablation study for bipartite graph on GDC data
        |-- run_bp_valentine.py # ablation study for bipartite graph on Valentine data
        |-- run_encoding_sampling_ablation_gdc.py # ablation study for encoding sampling on GDC data
        |-- run_encoding_sampling_ablation_valentine.py # ablation study for encoding sampling on Valentine data
        |-- run_multistrategy_ablation_gdc.py # ablation study for multi-strategy on GDC data
        |-- run_multistrategy_ablation_valentine.py # ablation study for multi-strategy on Valentine data
    |-- benchmark # code for benchmark study, note that batched benchmark on baseline methods are on this [repo](https://github.com/VIDA-NYU/data-harmonization-benchmark)
        |-- gdc_benchmark.py # benchmark study on GDC data
        |-- valentine_benchmark.py # benchmark study on Valentine data
|-- results_visualization # notebooks for results visualization
```

## :gear: 3. Example Usage
To reproduce the GDC benchmark results, you can run the following command:
```bash
python experiments/benchmarks/gdc_benchmark.py --mode [MODE] --embedding_model [EMBEDDING_MODEL] --llm_model [LLM_MODEL]
```
- `[MODE]`: Specifies the operational mode. Options include: `header-value-default`, `header-value-repeat`, and `header-value-verbose`.
- `[EMBEDDING_MODEL]`: Selects the pre-trained language model to use as the retriever. Available options are `mpnet`, `roberta`, `e5`, `arctic`, or `minilm`. The default model is `mpnet`.
- `[LLM_MODEL]`: Specifies the llm-based reranker. Options are `gpt-4o-mini` or `llama3.3-70b`.

To reproduce the Valentine benchmark results, you can run the following command:
```bash
python experiments/benchmarks/valentine_benchmark.py --mode [MODE] --dataset [DATASET]
```
where `[MODE]` is similar to the GDC benchmark and `[DATASET]` can be one of the following:
- `chembl`
- `magellan`
- `opendata`
- `tpc`
- `wikidata`

You can also change other Mageto configurations in the corresponding benchmark file.