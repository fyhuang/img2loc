"""
Download webdataset from Backblaze B2 using provided application key.
"""

import argparse
import json
import subprocess

from pathlib import Path

from braceexpand import braceexpand
import requests


def get_authorization_token(b2_api_key):
    """
    Get authorization token from B2 using provided application key.

    Args:
        b2_api_key (dict): Application key json file.

    Returns:
        str: Authorization token.
    """
    auth_url = "https://api.backblazeb2.com/b2api/v3/b2_authorize_account"
    auth_response = requests.get(auth_url, auth=(b2_api_key["keyID"], b2_api_key["applicationKey"]))
    auth_response.raise_for_status()

    return auth_response.json()["authorizationToken"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, required=True, help="Path to application key json file")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to download files to")
    parser.add_argument("urls", nargs='+', type=str, help="WDS URLs to download")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Expand URLs
    urls = []
    for url_template in args.urls:
        urls.extend(braceexpand(url_template))
    print(urls)

    # Load application key
    with open(args.key, "r") as f:
        b2_api_key = json.load(f)

    # Get authentication token from B2
    token = get_authorization_token(b2_api_key)

    # Download all URLs in parallel (with aria2c)
    urls_input = "\n".join(urls).encode("utf-8")
    subprocess.run(
        [
            "aria2c",
            "--dir", args.output_dir,
            "--input-file=-",
            f"--header=Authorization: {token}",
        ],
        check=True,
        input=urls_input,
    )
    #subprocess.run(
    #    [
    #        "parallel",
    #        "-q", # quote arguments (needed for Authorization header)
    #        "-u", # output directly to terminal
    #        #"--progress",
    #        "-j8",

    #        "curl",
    #        "--output-dir",
    #        args.output_dir,
    #        "-C", # automatically resume
    #        "-",
    #        "-OL",
    #        "-H",
    #        f"Authorization: {token}",
    #    ],
    #    check=True,
    #    input=urls_input,
    #)


if __name__ == "__main__":
    main()