# Usage

## CLI

```console
$ probas-dl --help
usage: probas-dl [-h] [--base-path BASE_PATH] [--data-folder-name DATA_FOLDER_NAME]
                 {download-index,store-uuids,download-categories,download-lcis,download-lcia-methods,download-processes-json} ...

positional arguments:
  {download-index,store-uuids,download-categories,download-lcis,download-lcia-methods,download-processes-json}
                        Command
    download-index      Download process index
    store-uuids         Store UUIDs in a text file
    download-categories
                        Download categories
    download-lcis       Download processes
    download-lcia-methods
                        Download LCIA methods
    download-processes-json
                        Download processes as JSON

options:
  -h, --help            show this help message and exit
  --base-path BASE_PATH
                        Base project path (default: current directory)
  --data-folder-name DATA_FOLDER_NAME
                        Name for data folder (default: data)
```


### `download-index`

```console
$ probas-dl download-index --help
usage: probas-dl download-index [-h] [--out OUT] [--format FORMAT]

options:
  -h, --help       show this help message and exit
  --out, -o OUT    Output file name (default: processes.<ext>)
  --format FORMAT  Output format -- json or xml (default: json)
```


### `store-uuids`

```console
$ probas-dl store-uuids --help
usage: probas-dl store-uuids [-h] [--out OUT] [--lci-results]

options:
  -h, --help            show this help message and exit
  --out, -o OUT         Output file name (default: uuids.txt, or lci-results-uuids.txt)
  --lci-results, --results
                        Store only LCI results
```


### `download-categories`

```console
$ probas-dl download-categories --help
usage: probas-dl download-categories [-h] [--out OUT]

options:
  -h, --help     show this help message and exit
  --out, -o OUT  Output file name (default: categories.json)
```


### `download-lcis`

```console
$ probas-dl download-lcis --help
usage: probas-dl download-lcis [-h] [--limit LIMIT] [--uuids-file UUIDS_FILE]

options:
  -h, --help            show this help message and exit
  --limit LIMIT         Limit number of processed to download
  --uuids-file UUIDS_FILE
                        File with UUIDs to download (default: uuids.txt)
```


### `download-lcia-methods`

```console
$ probas-dl download-lcia-methods --help
usage: probas-dl download-lcia-methods [-h]

options:
  -h, --help  show this help message and exit
```


### `download-processes-json`

```console
$ probas-dl download-processes-json --help
usage: probas-dl download-processes-json [-h] [--limit LIMIT] [--uuids-file UUIDS_FILE]

options:
  -h, --help            show this help message and exit
  --limit LIMIT         Limit number of processed to download
  --uuids-file UUIDS_FILE
                        File with UUIDs to download (default: uuids.txt)
```
