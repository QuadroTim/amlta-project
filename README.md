AMLTA Project
=============

# Requirements

Python 3.11

Tested on Windows and in Colab.

# Setup

With [uv](https://docs.astral.sh/uv/):

```console
$ uv sync
$ pre-commit install # optional
```

or pip:

```console
$ python -m venv .venv
$ . ./.venv/Scripts/activate
$ python -m pip install -e .
$ pre-commit install # optional
```

# Structure

## Notebooks

Scripts and notebooks are found in the `notebooks` folder.
- [`probas_eda.ipynb`](notebooks/probas_eda.ipynb): EDA of the probas data.
- [`openai_batch_generate_questions.ipynb`](notebooks/openai_batch_generate_questions.ipynb)
    Was used to send batches to the OpenAI API.
- [`processes_transform_yaml.ipynb`](notebooks/processes_transform_yaml.ipynb): Used to transform
    the processes data to yaml files.
- [`tapas_cache_train_data.ipynb`](notebooks/tapas_cache_train_data.ipynb): Used to prepare and
    store training data for TAPAS.
- [`tapas_fine_tuning.ipynb`](notebooks/tapas_fine_tuning.ipynb): TAPAS fine-tuning.
- [`tapas_eval_qwen_query_generation.ipynb`](notebooks/tapas_eval_qwen_query_generation.ipynb):
    Used to rephrase validation questions.
- [`tapas_eval.ipynb`](notebooks/tapas_eval.ipynb): TAPAS evaluation.
- [`LaBSE.py`](notebooks/LaBSE.py): This script implements the LaBSE (Language-agnostic BERT Sentence Embedding) model. Creates embeddings. Evaluates model effectiveness with and without reranking.
- [`Models Evaluation.py`](notebooks/Models%20Evaluation.py): This script evaluates the performance of various embedding models using metrics such as Precision@K, Recall@K, F1-Score, NDCG, and MRR.
- [`Reranked Models Evaluation.py`](notebooks/Reranked%20Models%20Evaluation.py): This script evaluates the effectiveness of models after applying reranking techniques to improve the quality of retrieved results.
- [`jina-embeddings-v2-base-de Fine-tuned.py`](notebooks/jina-embeddings-v2-base-de%20Fine-tuned.py): This script implements the fine-tuned version of the jina-embeddings-v2-base-de model. Creates embeddings. Evaluates model effectiveness with and without reranking.
- [`jina-embeddings-v2-base-de.py`](notebooks/jina-embeddings-v2-base-de.py): This script implements the base version of the jina-embeddings-v2-base-de model. Generates embeddings. Evaluates model effectiveness with and without reranking.
- [`multilingual-e5-large.py`](notebooks/multilingual-e5-large.py): This script implements the multilingual-e5-large model. Generates embeddings. Evaluates model effectiveness with and without reranking.
- [`paraphrase-multilingual-mpnet-base-v2.py`](notebooks/paraphrase-multilingual-mpnet-base-v2.py): This script implements the paraphrase-multilingual-mpnet-base-v2 model. Generates embeddings. Evaluates model effectiveness with and without reranking.
- [`Fine tuning of Jina base de.ipynb`](notebooks/Fine%20tuning%20of%20Jina%20base%20de.ipynb): This notebook is fine-tuning process of the jina-embeddings-v2-base-de model. It includes data preparation, training, and evaluation steps. 


Remaining notebooks can be ignored.


## Data

The `data` folder of the repo contains only essential files that have small size. The remaining data can be found in the
linked Google Drive and be merged.

The folder structure shortly described:
- all top level files + folders `schemas` and `ILCD` are ProBas data
  - main data that was used is in `ILCD/processes_json`
- `qdrant-yaml` is the folder for the qdrant vector store
- `generated` contains data related to the synthesized question generation
  - `tapas-eval-questions.jsonl` contains 100 rephrased then rewritten questions of the validation set
  - the full set of generated questions are in `generated/questions/out`
  - corresponding files including the syntactically generated query parameters are in files `generated/questions/batch_inputs/*_input.jsonl`
- `jina-ft` contains intermediate outputs used for fine-tuning the jina model
- `tapas-ft` contains checkpoints as well as training data used for fine tuning tapas


## Source code (`src/amlta`)

### `probas`

Contains code to download, process and load the probas data.

### `formatting`

Contains code to format structured data to markdown or yaml. Used for RAG.


### `question_generation`

Contains code to generate synthetic queries and questions.

### `tapas`

Contains code for preprocessing for TAPAS as well as inference retrieval.

The used model can be found on [Hugging Face - tapas-finetuned-probas-supervised-2](https://huggingface.co/woranov/tapas-finetuned-probas-supervised-2).


### `app`

Contains code for the streamlit app. To use it with Ollama, the base url and the llm model name
should be passed as arguments.

```console
$ amlta-app -- --model "qwen2.5:32b-instruct-q3_K_M" --base-url "http://random-domain.ngrok-free.app"
```

It is designed to be ran with `qwen2.5:32b-instruct-q3_K_M`. For that, ollama can be proxied with
Colab + ngrok using the [`colab_proxy.ipynb` notebook](https://colab.research.google.com/github/woranov/amlta-project/blob/main/notebooks/colab_proxy.ipynb).
