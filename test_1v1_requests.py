# test_1vs1_all_tables.py

import requests
from bs4 import BeautifulSoup
import json

URL = "https://one-versus-one.com/en/predictions/premier-league"

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Connection": "keep-alive",
})

def main():
    resp = session.get(URL, timeout=15)
    print("Status:", resp.status_code)
    if resp.status_code != 200:
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table")
    print(f"Found {len(tables)} tables on the page.")

    for idx, table in enumerate(tables):
        print(f"\n=== TABLE {idx} ===")

        # Headers (if any)
        header_cells = table.find_all("th")
        headers = [h.get_text(strip=True) for h in header_cells]
        print("Headers:", headers if headers else "(no explicit <th> headers)")

        # Sample first few rows
        rows = table.find_all("tr")
        preview = []
        for row in rows[:5]:
            cells = [c.get_text(strip=True) for c in row.find_all(["th", "td"])]
            if cells:
                preview.append(cells)

        print("First rows:")
        print(json.dumps(preview, indent=2))

if __name__ == "__main__":
    main()
