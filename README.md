
<h2 align="center">Magneto: Combining Small and Large Language Models for
Schema Matching</h2>

> Welcome to Magneto!

Magneto is an innovative framework designed to enhance schema matching (SM) by intelligently combining small, pre-trained language models (SLMs) with large language models (LLMs). Our approach is structured to be both cost-effective and broadly applicable.

The framework operates in two distinct phases:
- **Candidate Retrieval**: This phase involves using SLMs to quickly identify a manageable subset of potential matches from a vast pool of possibilities.
- **Reranking**: In this phase, LLMs take over to assess and reorder the candidates, simplifying the process for users to review and select the most suitable matches.

## Code Structure
> note that batched benchmark on baseline methods are on this [repo](https://github.com/VIDA-NYU/data-harmonization-benchmark).

```sh
|-- algorithm
    |-- schema_matching
        |-- magneto # code for Magneto
            |-- finetune # code for Magneto FT
            |-- magneto.py # Magneto core
        |-- topk_metrics.py # Introducing Recall @ topk
|-- experiments
    |-- ablation # code for ablation study
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
|-- visualizations # graphs and tables
```

## Data Folder
The data folder contains the datasets used for data integration tasks. You can download the data folder from [this Google Drive link](https://drive.google.com/drive/folders/19kCWQI0CWHs1ZW9RQEUSeK6nuXoA-5B7?usp=sharing).
