"""
Pull recent 10-K filings from SEC EDGAR for a small set of finance companies,
strip HTML, and save clean text into data/raw/ for the RAG pipeline to ingest.

SEC requires a real identifying User-Agent on every request (name + email).
Replace YOUR_NAME and YOUR_EMAIL below before running, or SEC will block you.

Usage:
    python scripts/fetch_sec_filings.py
"""
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# --- REPLACE THESE TWO LINES WITH YOUR OWN INFO ---
USER_AGENT = "Aviva Kahlon avivakahlon2003@gmail.com"
# ----------------------------------------------------

HEADERS = {"User-Agent": USER_AGENT}

COMPANIES = {
    "jpmorgan_chase": "0000019617",
    "visa": "0001403161",
    "mastercard": "0001141391",
}

FILINGS_PER_COMPANY = 2  # most recent N 10-Ks per company
OUTPUT_DIR = Path("data/raw")


def get_recent_10k_filings(cik: str, limit: int = 2):
    """Query SEC's submissions API and return the N most recent 10-K filings."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    filings = []
    for form, accession, primary_doc, filing_date in zip(
        recent["form"], recent["accessionNumber"], recent["primaryDocument"], recent["filingDate"]
    ):
        if form == "10-K":
            filings.append(
                {
                    "accession": accession.replace("-", ""),
                    "primary_doc": primary_doc,
                    "filing_date": filing_date,
                }
            )
        if len(filings) >= limit:
            break
    return filings


def fetch_and_clean_filing(cik: str, accession: str, primary_doc: str) -> str:
    """Download a filing document and strip it down to plain text."""
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(separator="\n")
    # collapse excessive blank lines left behind by stripped HTML
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text


def main():
    if "YOUR_EMAIL" in USER_AGENT:
        raise SystemExit(
            "Set your real name/email in USER_AGENT at the top of this script before running. "
            "SEC blocks requests without a real identifying User-Agent."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for company_name, cik in COMPANIES.items():
        print(f"\nFetching 10-Ks for {company_name}...")
        filings = get_recent_10k_filings(cik, FILINGS_PER_COMPANY)

        for f in filings:
            year = f["filing_date"][:4]
            out_path = OUTPUT_DIR / f"{company_name}_{year}_10K.txt"
            print(f"  -> {f['filing_date']}: {f['primary_doc']}")

            text = fetch_and_clean_filing(cik, f["accession"], f["primary_doc"])
            out_path.write_text(text, encoding="utf-8")
            print(f"     saved to {out_path} ({len(text):,} chars)")

            time.sleep(0.5)  # be polite to SEC's servers

    print("\nDone. Check data/raw/ for the saved filings.")


if __name__ == "__main__":
    main()