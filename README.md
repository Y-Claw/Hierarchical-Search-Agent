# Hierarchical Search Agent

Hierarchical Search Agent is an experimental search and reasoning agent framework for question answering tasks.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Configuration

Start from the example config:

```bash
cp configs/qwq_32b/hierarchical_tot_config_qwq.example.yaml configs/local.yaml
```

Keep local configs, datasets, results, logs, and `.env` files out of Git. Use environment variables for private endpoints, model paths, API keys, and proxies.

## Usage

```bash
python -m deep_search.evaluation.hotpotqa --config configs/local.yaml
```
