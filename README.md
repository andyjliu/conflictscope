# ConflictScope
This repository contains code and data associated with [Generative Value Conflicts Reveal LLM Priorities](https://www.arxiv.org/abs/2509.25369). It can also be used to run the ConflictScope evaluation pipeline over custom value sets and models. 

## Setup
```
conda create -n conflictscope python=3.12
conda activate conflictscope
pip install -r requirements.txt
conda env config vars set OPENAI_API_KEY=<>
conda env config vars set ANTHROPIC_API_KEY=<>
```
[vLLM](https://github.com/vllm-project/vllm) is used for open-weight model inference; adding new models or inference libraries can be done by modifying `src/model_wrappers.py`.

## Evaluation
The general evaluation flow is:
1. Generate data by calling   src/generate_scenarios.py   and   src/filter_scenarios.py (example scripts for doing so are in the scripts repository),
2. Evaluate models, either using multiple-choice or simulated-user evaluation, by calling `src/evaluate_models.py`,
3. Use `src/analyze_experiment.py` to generate dataset statistics and model rankings from the evaluation results.

To run on custom value sets, you can define your own value sets at `value_sets/[VALUE_SET_NAME].json` and pass the value set name into all scripts.

## Citation
Please cite our paper if you use ConflictScope in your own work:
```
@article{liu2025generative,
  title={Generative Value Conflicts Reveal LLM Priorities},
  author={Liu, Andy and Ghate, Kshitish and Diab, Mona and Fried, Daniel and Kasirzadeh, Atoosa and Kleiman-Weiner, Max},
  journal={arXiv preprint arXiv:2509.25369},
  year={2025}
}
```
