import json
from pathlib import PureWindowsPath



def get_urls(sources, json_path="txt_to_url.json"):
    urls = []

    for source in sources:
        if isinstance(source, dict):
            source_path = source.get("source", "")
        else:
            source_path = source

        file_name = PureWindowsPath(source_path).name

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        url = data.get(file_name)

        urls.append({
            "file_name": file_name,
            "url": url
        })

    return urls