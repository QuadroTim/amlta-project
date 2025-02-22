AMLTA Project
=============

# Requirements

Python 3.11

# Setup

With [uv](https://docs.astral.sh/uv/):

```console
$ uv sync
$ pre-commit install
```

or pip:

```console
$ python -m venv .venv
$ . ./.venv/Scripts/activate
$ python -m pip install -e .
$ pre-commit install
```

# App

1. Open the [`colab_proxy.ipynb` in Colab](https://colab.research.google.com/github/woranov/amlta-project/blob/main/notebooks/colab_proxy.ipynb)
2. Run Ollama and pull the model
3. Run ngrok and copy the URL
4. Run the app. Example:
    ```console
    $ amlta-app -- --model "qwen2.5:32b-instruct-q3_K_M" --base-url "http://random-domain.ngrok-free.app"
    ```


# Packages

## [`probas`](src/amlta/probas/README.md)

### Download

Download the data from Google Drive. Or download manually from the API using the `probas-dl` command.

#### Manual

- Download index

  ```console
  $ probas-dl download-index --format json
  ```
  and/or
  ```console
  $ probas-dl download-index --format xml
  ```

- Download ILCD data

  ```console
  $ probas-dl download-lcis
  ```

  - Or LCI results only

    ```console
    $ probas-dl download-lcis --uuids-file lci-results-uuids.txt
    ```

- Download processes as json

  ```console
  $ probas-dl download-processes-json
  ```

- Download LCIA methods

  ```console
  $ probas-dl download-lcia-methods
  ```
