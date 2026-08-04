import json
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


SYSTEM_PROMPT_TEACHER = """You are a fintech compliance tutor grounded in Indian regulatory documents.
Read the source passage carefully. Generate THREE high-quality, exam-style questions and answers, ALL strictly grounded in the passage text — do NOT add outside knowledge.

Requirements:
- Each question must require reasoning ABOUT the passage (interpretation, application, "what does this mean for a reporting entity"), not a trivia lookup.
- Each answer must be clear and well-structured (2-5 sentences) directly referencing the passage content.
- Output ONLY valid JSON with the form: {"pairs": [{"question": "...", "answer": "..."}, ...]} with exactly 3 pairs.
- Do not include the source text in the output.
"""


class AutoDidactFetcher:
    """Fetches source documents ONLY from sites that allow crawling.

    robots.txt verified:
      - FIU-IND: allowed (no Disallow)
      - CERSAI: no robots.txt -> allowed
      - SEBI: main content allowed (only /js, /css disallowed)
      - RBI: public regulatory PDFs on official rbidocs server (public documents)
    Skipped (disallow all): IRDAI, PFRDA, NPCI.
    """

    def __init__(self, cache_dir: str = "data/autodidact_src"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })

    def _cache(self, key: str, data) -> bool:
        path = self.cache_dir / f"{key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return True

    def _load_cache(self, key: str):
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"[^\x00-\x7F]+", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def fetch_html(self, url: str, key: str) -> str:
        cached = self._load_cache(key)
        if cached:
            return cached
        r = self.session.get(url, timeout=45)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = self._clean_text(soup.get_text("\n", strip=True))
        self._cache(key, text)
        time.sleep(0.5)
        return text

    def fetch_pdf(self, url: str, key: str) -> str:
        cached = self._load_cache(key)
        if cached:
            return cached
        r = self.session.get(url, timeout=60)
        r.raise_for_status()
        reader = PdfReader(BytesIO(r.content))
        text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
        text = self._clean_text(text)
        self._cache(key, text)
        time.sleep(0.5)
        return text

    # ---------- FIU-IND ----------
    def fetch_fiu_ind(self) -> list:
        pages = {
            "pmla_2002": "https://fiuindia.gov.in/files/AML_Legislation/pmla_2002.html",
            "pml_rules_2005": "https://fiuindia.gov.in/files/AML_Legislation/notification.html",
            "scheduled_offences": "https://fiuindia.gov.in/files/AML_Legislation/scheduled_offences.html",
            "section_12a": "https://fiuindia.gov.in/files/AML_Legislation/DOR_Section12A_WMD.html",
            "faqs": "https://fiuindia.gov.in/files/FAQs/faqs.html",
        }
        docs = []
        for name, url in pages.items():
            try:
                text = self.fetch_html(url, f"fiu_{name}")
                if len(text) > 1000:
                    docs.append({
                        "title": f"FIU-IND {name}",
                        "url": url,
                        "text": text[:50000],
                        "source": "fiu_ind",
                    })
                    print(f"  FIU {name}: {len(text)} chars")
            except Exception as e:
                print(f"  FIU {name} ERR: {str(e)[:70]}")
        # FIU-IND compliance order PDFs (AML enforcement actions) + publications
        pdf_groups = [
            ("https://fiuindia.gov.in/files/Compliance_Orders/orders.html", 130, "fiu_ord_"),
            ("https://fiuindia.gov.in/files/Publication/Publication.html", 40, "fiu_pub_"),
        ]
        for list_url, max_n, prefix in pdf_groups:
            try:
                r = self.session.get(list_url, timeout=45)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "lxml")
                pdfs = []
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if ".pdf" in href.lower():
                        if href.startswith("../../"):
                            href = "https://fiuindia.gov.in/" + href.replace("../../", "", 1)
                        elif href.startswith("/"):
                            href = "https://fiuindia.gov.in" + href
                        elif href.startswith("../"):
                            href = "https://fiuindia.gov.in/" + href.replace("../", "", 1)
                        pdfs.append(href)
                pdfs = list(dict.fromkeys(pdfs))
                print(f"  {prefix} PDFs found: {len(pdfs)}")
                for i, pdf_url in enumerate(pdfs[:max_n]):
                    try:
                        text = self.fetch_pdf(pdf_url, f"{prefix}{i}")
                        if len(text) > 1500:
                            docs.append({
                                "title": f"FIU-IND {prefix}{i}",
                                "url": pdf_url,
                                "text": text[:50000],
                                "source": "fiu_ind",
                            })
                            print(f"    {prefix}{i}: {len(text)} chars")
                    except Exception as e:
                        print(f"    {prefix}{i} ERR: {str(e)[:50]}")
            except Exception as e:
                print(f"  {prefix} list ERR: {str(e)[:70]}")
        return docs

    # ---------- CERSAI ----------
    def fetch_cersai(self) -> list:
        base = "https://cersai.org.in/CERSAI/"
        urls = [
            f"{base}home.prg",
            f"{base}aboutus.prg",
            f"{base}circulars.prg",
            f"{base}faq.prg",
            f"{base}entityregn.prg",
            f"{base}feestructure.prg",
        ]
        docs = []
        for i, url in enumerate(urls):
            try:
                text = self.fetch_html(url, f"cersai_{i}")
                if len(text) > 1000:
                    docs.append({
                        "title": f"CERSAI page {i}",
                        "url": url,
                        "text": text[:50000],
                        "source": "cersai",
                    })
                    print(f"  CERSAI {i}: {len(text)} chars")
            except Exception as e:
                print(f"  CERSAI {i} ERR: {str(e)[:70]}")
        return docs

    # ---------- SEBI ----------
    def fetch_sebi_orders(self, max_pages: int = 4) -> list:
        docs = []
        base = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=4"
        for page in range(max_pages):
            url = base if page == 0 else f"{base}&n={page}"
            try:
                r = self.session.get(url, timeout=45)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "lxml")
                order_links = []
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    title = a.get_text(strip=True)
                    if (href.startswith("https://www.sebi.gov.in/")
                            and "/publiccomment" not in href
                            and title and len(title) > 25):
                        order_links.append((title, href))
                print(f"  SEBI list page {page}: {len(order_links)} doc links")
                for title, href in order_links[:6]:
                    try:
                        body = self.fetch_html(href, f"sebi_order_{hashlib_md5(href)}")
                        if len(body) > 400:
                            docs.append({
                                "title": title,
                                "url": href,
                                "text": body[:50000],
                                "source": "sebi",
                            })
                    except Exception as e:
                        print(f"    doc ERR {str(e)[:60]}")
                if not order_links:
                    break
            except Exception as e:
                print(f"  SEBI list {page} ERR: {str(e)[:70]}")
        return docs

    def fetch_all(self) -> dict:
        print("Fetching FIU-IND...")
        fiu = self.fetch_fiu_ind()
        print("Fetching SEBI...")
        sebi = self.fetch_sebi_orders()
        return {
            "fiu_ind": fiu,
            "sebi": sebi,
        }

    def save(self, data: dict, filename: str = "autodidact_source_docs.json"):
        path = Path("data") / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"Saved source docs to {path}")


def hashlib_md5(s: str) -> str:
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()[:10]


if __name__ == "__main__":
    f = AutoDidactFetcher()
    data = f.fetch_all()
    f.save(data)
