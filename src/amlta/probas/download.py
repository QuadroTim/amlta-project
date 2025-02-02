import argparse
import io
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, TypeAlias, cast

import httpx
from tqdm.auto import tqdm

from amlta.probas.xml import ET, get_tree


@dataclass
class Config:
    API_BASE_PARAMS: ClassVar[dict] = {
        "format": "json",
        "sortOrder": "true",
        "lang": "en",
        "langFallback": "true",
    }

    probas_base_url: str = "https://data.probas.umweltbundesamt.de/resource"
    probas_datastock: str = "ebee4288-5f27-4d18-8e2d-c98e985cda5a"

    base_path: Path = Path(".")
    data_foldername: str = "data"
    processes_index_filename: str = "processes.json"
    categories_filename: str = "categories.json"
    uuids_download_filename: str = "uuids.txt"

    index_page_size: int = 500

    @property
    def probas_index_url(self):
        return f"{self.probas_base_url}/datastocks/{self.probas_datastock}/processes"

    def probas_categories_url(self, parent_categories=()):
        url = self.probas_index_url + "/categories"

        if parent_categories:
            return "/".join([url, *parent_categories, "subcategories"])
        else:
            return url

    def probas_ilcd_zip_url(self, uuid: str):
        return f"{self.probas_index_url}/{uuid}/zipexport"

    def probas_lcia_methods_url(self, uuid: str):
        return f"{self.probas_base_url}/lciamethods/{uuid}"

    @property
    def data_folder(self):
        folder = self.base_path / self.data_foldername
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def processes_index_file(self):
        return self.data_folder / self.processes_index_filename

    @property
    def categories_file(self):
        return self.data_folder / self.categories_filename

    @property
    def uuids_file(self):
        return self.data_folder / self.uuids_download_filename

    @property
    def ilcd_data_folder(self):
        return self.data_folder / "ILCD"

    @property
    def ilcd_processes_location(self):
        return self.ilcd_data_folder / "processes"

    @property
    def ilcd_lcia_methods_location(self):
        return self.ilcd_data_folder / "lciamethods"


DEFAULT_CONFIG = Config()


def fetch_categories(
    parent_categories=(),
    category_system: Literal["UBA", "NACE 1.1"] = "UBA",
    config: Config = DEFAULT_CONFIG,
):
    data = []

    params = config.API_BASE_PARAMS | {
        "startIndex": 0,
        "pageSize": 100,
        "sortOrder": "true",
        "sortBy": "name",
        "catSystem": category_system,
    }

    url = config.probas_categories_url(parent_categories)
    response = httpx.get(url, params=params)

    if category_system == "NACE 1.1" and response.status_code == 400:
        # this error even happens on the website. E.g., for
        #   "Einzelhandel (ohne Handel mit Kraftfahrzeugen und ohne Tankstellen); Reparatur von Gebrauchsgütern"
        return data

    response.raise_for_status()
    resp_data = response.json()
    for category in resp_data.get("category") or []:
        category_name = category["value"]
        time.sleep(0.1)
        category["sub"] = fetch_categories(
            category_system=category_system,
            parent_categories=parent_categories + (category_name,),
            config=config,
        )
        data.append(category)

    return data


def download_categories_all(config: Config = DEFAULT_CONFIG):
    out_file = config.categories_file

    if out_file.exists():
        print(f"File {out_file} already exists, skipping")
        return

    data = {
        "uba": {
            "name": "UBA",
            "categories": fetch_categories(category_system="UBA", config=config),
        },
        "nace": {
            "name": "NACE 1.1",
            "categories": fetch_categories(category_system="NACE 1.1", config=config),
        },
    }
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)

    return data


def fetch_index_page(
    page=0, format="json", config: Config = DEFAULT_CONFIG
) -> dict | tuple[ET, dict]:
    params = config.API_BASE_PARAMS | {
        "format": format,
        "startIndex": page * config.index_page_size,
        "pageSize": config.index_page_size,
    }

    response = httpx.get(config.probas_index_url, params=params)
    response.raise_for_status()
    if format == "json":
        return response.json()
    else:
        return get_tree(io.BytesIO(response.content))


def download_index_all(format="json", config: Config = DEFAULT_CONFIG):
    out_file = config.processes_index_file.with_suffix(f".{format}")

    if out_file.exists():
        print(f"File {out_file} already exists, skipping")
        return

    page = 0
    first = fetch_index_page(page, format=format, config=config)

    if format == "json":
        first = cast(dict, first)
        all_items = []
        extend = all_items.extend

        total = first["totalCount"]
        total_pages = (total + config.index_page_size - 1) // config.index_page_size

        extend(first["data"])
        for page in tqdm(range(1, total_pages), desc="Fetching pages"):
            page_resp = cast(dict, fetch_index_page(page, format=format, config=config))
            extend(page_resp["data"])

        with open(out_file, "w") as f:
            json.dump(all_items, f, indent=2)

        return all_items
    else:
        tree, ns = first
        root = tree.getroot()

        sapi_ns_uri = ns["sapi"]
        sapi_key = lambda key: f"{{{sapi_ns_uri}}}{key}"  # noqa: E731

        total = int(root.attrib[sapi_key("totalSize")])
        total_pages = (total + config.index_page_size - 1) // config.index_page_size
        del root.attrib[sapi_key("startIndex")]
        del root.attrib[sapi_key("pageSize")]

        for page in tqdm(range(1, total_pages), desc="Fetching pages"):
            page_tree, _ = fetch_index_page(page, format=format, config=config)
            page_root = page_tree.getroot()
            processes = page_root.findall(".//p:process", namespaces=page_root.nsmap)
            root.extend(processes)

        tree.write(out_file, pretty_print=True)  # type: ignore

        return root


def load_index(config: Config = DEFAULT_CONFIG):
    path = config.processes_index_file

    with path.open("r") as f:
        return json.load(f)


ResultType: TypeAlias = Literal["lci-results"]


def store_uuids(
    index=None, type_filter: ResultType | None = None, config: Config = DEFAULT_CONFIG
):
    if index is None:
        index = load_index(config)

    filter_item = lambda item: True  # noqa: E731

    if type_filter == "lci-results":
        filter_item = lambda item: item["type"] == "LCI result"  # noqa: E731

    out_file = config.uuids_file
    uuids = {item["uuid"] for item in filter(filter_item, index)}

    out_file.write_text("\n".join(uuids))


def download_lci(uuid, config: Config = DEFAULT_CONFIG):
    out_folder = config.ilcd_data_folder

    if out_folder.name == "ILCD":
        out_folder = out_folder.parent

    zip_url = config.probas_ilcd_zip_url(uuid)
    bio = io.BytesIO()

    with httpx.stream("GET", zip_url, timeout=httpx.Timeout(60)) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes():
            bio.write(chunk)

    with zipfile.ZipFile(bio, "r") as zf:
        zf.extractall(out_folder)


def download_lcis(
    uuids: list[str],
    limit: int | None = None,
    force: bool | None = False,
    config: Config = DEFAULT_CONFIG,
):
    processes_folder = config.ilcd_processes_location
    out_folder = config.ilcd_data_folder
    total = len(uuids)

    if not force:
        downloaded_uuids = {
            f.name.split("_")[0] for f in processes_folder.glob("*_*.xml")
        }
        already_downloaded = set(uuids) & downloaded_uuids
        uuids = [uuid for uuid in uuids if force or uuid not in downloaded_uuids]

    uuids = uuids[:limit]

    out_folder.mkdir(parents=True, exist_ok=True)
    for uuid in (pbar := tqdm(uuids, desc="Downloading LCIs")):
        download_lci(uuid, config=config)
        if not force:
            pbar.set_postfix_str(f"({pbar.n + len(already_downloaded)}/{total})")


def download_lcis_from_file(limit=None, config: Config = DEFAULT_CONFIG):
    uuids_file = config.uuids_file
    uuids = uuids_file.read_text().splitlines()
    download_lcis(uuids, limit=limit, config=config)


def download_processes_json(
    uuids: list[str], limit: int | None = None, config: Config = DEFAULT_CONFIG
):
    processes_folder = config.ilcd_processes_location.with_name(
        config.ilcd_processes_location.name + "_json"
    )
    processes_folder.mkdir(parents=True, exist_ok=True)

    total = len(uuids)
    uuids = uuids[:limit]

    downloaded_uuids = {f.name.split("_")[0] for f in processes_folder.glob("*_*.json")}
    already_downloaded = set(uuids) & downloaded_uuids
    uuids = [uuid for uuid in uuids if uuid not in downloaded_uuids]

    for uuid in (pbar := tqdm(uuids, desc="Downloading processes")):
        url = f"{config.probas_base_url}/processes/{uuid}"
        params = config.API_BASE_PARAMS | {"format": "json", "view": "extended"}
        response = httpx.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        filename = processes_folder / f"{uuid}_{data['version']}.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        pbar.set_postfix_str(f"({pbar.n + len(already_downloaded)}/{total})")


def download_processes_from_file(limit=None, config: Config = DEFAULT_CONFIG):
    uuids_file = config.uuids_file
    uuids = uuids_file.read_text().splitlines()
    download_processes_json(uuids, limit=limit, config=config)


def download_process_lcia_methods(
    process_path: Path, pbar=None, config: Config = DEFAULT_CONFIG
):
    tree, ns = get_tree(process_path)
    root = tree.getroot()

    for lcia_method_ref_node in root.findall(
        ".//referenceToLCIAMethodDataSet", namespaces=ns
    ):
        uuid = lcia_method_ref_node.attrib["refObjectId"]
        path = config.ilcd_lcia_methods_location / f"{uuid}.xml"
        if path.exists():
            continue

        url = config.probas_lcia_methods_url(uuid)
        params = config.API_BASE_PARAMS | {"format": "xml"}
        response = httpx.get(url, params=params)
        response.raise_for_status()
        path.write_text(response.text)

        if pbar:
            pbar.update()


def download_all_lcia_methods(config: Config = DEFAULT_CONFIG):
    processes_path = config.ilcd_processes_location
    out_folder = config.ilcd_lcia_methods_location

    lcia_methods_pbar = tqdm(desc="Downloading LCIA methods")

    out_folder.mkdir(parents=True, exist_ok=True)
    files = list(processes_path.glob("*.xml"))
    for process_path in tqdm(files, desc="Processing processes"):
        download_process_lcia_methods(
            process_path, pbar=lcia_methods_pbar, config=config
        )


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--base-path",
        help="Base project path (default: current directory)",
        type=Path,
        default=Path("."),
    )
    arg_parser.add_argument(
        "--data-folder-name",
        help="Name for data folder (default: data)",
        type=Path,
        default=None,
    )

    sub = arg_parser.add_subparsers(required=True, dest="command", help="Command")

    cmd_index = sub.add_parser("download-index", help="Download process index")
    cmd_index.add_argument(
        "--out",
        "-o",
        help="Output file name (default: processes.<ext>)",
        type=str,
        default=None,
    )
    cmd_index.add_argument(
        "--format",
        help="Output format -- json or xml (default: json)",
        type=str,
        default="json",
    )

    cmd_store_uuids = sub.add_parser("store-uuids", help="Store UUIDs in a text file")
    cmd_store_uuids.add_argument(
        "--out",
        "-o",
        help="Output file name (default: uuids.txt, or lci-results-uuids.txt)",
        type=str,
        default=None,
    )
    cmd_store_uuids.add_argument(
        "--lci-results",
        "--results",
        help="Store only LCI results",
        action="store_true",
    )

    cmd_categories = sub.add_parser("download-categories", help="Download categories")
    cmd_categories.add_argument(
        "--out",
        "-o",
        help="Output file name (default: categories.json)",
        type=str,
        default=None,
    )

    cmd_downlaod = sub.add_parser("download-lcis", help="Download processes")
    cmd_downlaod.add_argument(
        "--limit",
        help="Limit number of processed to download",
        type=int,
        default=None,
    )
    cmd_downlaod.add_argument(
        "--uuids-file",
        help="File with UUIDs to download (default: uuids.txt)",
        type=str,
        default=None,
    )

    cmd_download_lcia_methods = sub.add_parser(  # noqa: F841
        "download-lcia-methods", help="Download LCIA methods"
    )

    cmd_download_processes_json = sub.add_parser(
        "download-processes-json", help="Download processes as JSON"
    )
    cmd_download_processes_json.add_argument(
        "--limit",
        help="Limit number of processed to download",
        type=int,
        default=None,
    )
    cmd_download_processes_json.add_argument(
        "--uuids-file",
        help="File with UUIDs to download (default: uuids.txt)",
        type=str,
        default=None,
    )

    args = arg_parser.parse_args()
    config = Config(base_path=args.base_path)
    if args.data_folder_name:
        config.data_foldername = args.data_folder_name

    match args.command:
        case "download-index":
            if args.out:
                config.processes_index_filename = args.out

            download_index_all(format=args.format, config=config)

        case "store-uuids":
            if args.out:
                config.uuids_download_filename = args.out

            if args.lci_results:
                type_filter = "lci-results"
                if not args.out:
                    config.uuids_download_filename = (
                        "lci-results-" + config.uuids_download_filename
                    )
            else:
                type_filter = None

            store_uuids(type_filter=type_filter, config=config)

        case "download-categories":
            if args.out:
                config.categories_filename = args.out

            download_categories_all(config=config)

        case "download-lcis":
            if args.uuids_file:
                config.uuids_download_filename = args.uuids_file

            download_lcis_from_file(config=config, limit=args.limit)

        case "download-lcia-methods":
            download_all_lcia_methods(config=config)

        case "download-processes-json":
            if args.uuids_file:
                config.uuids_download_filename = args.uuids_file

            download_processes_from_file(config=config, limit=args.limit)

        case _:
            raise ValueError("Invalid command")


if __name__ == "__main__":
    main()
