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
