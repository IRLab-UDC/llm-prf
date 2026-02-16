# LLM-Assisted Pseudo-Relevance Feedback

This repo holds the code to reproduce the experiments of the following paper:

#### LLM-Assisted Pseudo-Relevance Feedback @ ECIR '26 ([Link](https://arxiv.org/abs/2601.11238))

This code will only run grid search on training collections.

To run these experiments, you will need maven and the JDK 21 installed on your system. Then, you can compile the project with

```
mvn clean package
```

This will create a fatjar under the ```target/``` folder. Execute this jar to launch the experiments.


### Configure Dataset

Edit `src/main/scripts/dataset_config.sh` to set your data paths:

```bash
FOLDER="/path/to/your/data/"
```

The script supports multiple datasets (AP8889, ROBUST04) with automatic path switching.

### Start LLM Services

For MonoT5-based strategies:
```bash
python src/main/python/mono_t5.py
```

For VLLM-based strategies:
```bash
python src/main/python/serve_vllm.py
```

These services must be running before the next step.

### Run Experiments (grid search)

```bash
cd src/main/scripts
./run_grid_search.sh
```

The grid search automatically:
- Runs baseline LMDirichlet retrieval
- Tests multiple PRF strategies (VLLM, VLLM-PROB, etc.)
- Sweeps over parameters: depth (k), expansion terms (e), lambda (λ)
- Skips already completed experiments (resumable)

### Grid Search Parameters

Default parameter ranges (configurable in `dataset_config.sh`):
- **Depths (k)**: [5, 10, 25, 50, 75, 100]
- **Expansion terms (e)**: [5, 10, 15, 20, 25, 30]
- **Lambda (λ)**: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
- **RF Strategies**: [VLLM, VLLM-PROB, MONOT5, MONOT5-PROB, etc.]

### Cache System

The system uses collection-specific caches to avoid redundant LLM queries:
- MonoT5 cache: `{cache_dir}/t5_cache.tsv`
- VLLM cache: `{cache_dir}/vllm_cache.tsv`

Caches are automatically loaded and saved across runs.
