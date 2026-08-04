"""Crawl Indian government sites for fintech/regulatory reports and info.

Independent of autodidact_fetcher / autodidact_gen — writes only to
data/gov_source_docs.json and caches in data/gov_src/ so the running
DeepSeek autodidact job is never disturbed.

Every site is robots.txt-verified before fetching (urllib.robotparser);
sites that disallow crawling are skipped. Crawl-delay is honored.
"""

import json
import re
import time
from io import BytesIO
from pathlib import Path
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Sites to attempt. All are Indian government / statutory bodies.
# skip_reason marks sites previously verified as not crawlable.
SITES = {
    "ed": {
        "name": "Directorate of Enforcement",
        "base": "https://enforcementdirectorate.gov.in",
        "pages": [
            "/en/act/pmla-2002",
            "/en/page/about-us",
            "/en/reports",
            "/en/news/press-releases",
            "/en/act",
        ],
        "max_pdfs": 15,
    },
    "india_code": {
        "name": "India Code (legislative.gov.in)",
        "base": "https://www.indiacode.nic.in",
        "pages": [
            "/handle/123456789/1968",  # PMLA 2002
            "/handle/123456789/2006",  # RBI Act 1934
            "/handle/123456789/14",    # Negotiable Instruments Act 1881
            "/",
        ],
        "max_pdfs": 8,
    },
    "pib": {
        "name": "Press Information Bureau",
        "base": "https://pib.gov.in",
        "pages": [
            "/",
            "/news",  # may 404 -> tolerated
        ],
        "max_pdfs": 5,
    },
    "uidai": {
        "name": "UIDAI (Aadhaar eKYC)",
        "base": "https://uidai.gov.in",
        "pages": [
            "/myaadhaar",
            "/en/ecosystem/authentication-devices-documents",
            "/en/faqs",
            "/en/what-is-aadhaar",
        ],
        "max_pdfs": 10,
    },
    "digilocker": {
        "name": "DigiLocker",
        "base": "https://www.digilocker.gov.in",
        "pages": [
            "/",
            "/about",
        ],
        "max_pdfs": 3,
    },
    "mca": {
        "name": "Ministry of Corporate Affairs",
        "base": "https://www.mca.gov.in",
        "pages": [
            "/",
            "/MinistryV2/beneficialownership.html",
        ],
        "max_pdfs": 8,
    },
    "dfs": {
        "name": "Dept. of Financial Services",
        "base": "https://financialservices.gov.in",
        "pages": [
            "/",
            "/schemes",
            "/new-initiatives",
        ],
        "max_pdfs": 5,
    },
    "incometax": {
        "name": "Income Tax / CBDT",
        "base": "https://www.incometax.gov.in",
        "pages": [
            "/",
        ],
        "max_pdfs": 3,
    },
    "finmin": {
        "name": "Ministry of Finance",
        "base": "https://finmin.nic.in",
        "pages": [
            "/",
        ],
        "max_pdfs": 3,
    },
    "cerfsai_retry": {
        "name": "CERSAI (re-check)",
        "base": "https://cersai.org.in",
        "pages": ["/CERSAI/home.prg", "/CERSAI/faq.prg"],
        "max_pdfs": 0,
    },
    "irdai_retry": {
        "name": "IRDAI (re-check)",
        "base": "https://www.irdai.gov.in",
        "pages": ["/", "/cms/default.aspx"],
        "max_pdfs": 0,
    },
    "pfrda_retry": {
        "name": "PFRDA (re-check)",
        "base": "https://www.pfrda.org.in",
        "pages": ["/", "/index1.cshtml"],
        "max_pdfs": 0,
    },
    "npci_retry": {
        "name": "NPCI (re-check)",
        "base": "https://www.npci.org.in",
        "pages": ["/", "/discover-npci/our-products/upi"],
        "max_pdfs": 0,
    },
    # --- blogs & articles on DigiLocker + Indian fintech services (DPI) ---
    "blogs_digilocker": {
        "name": "NeGD blog: DigiLocker digital briefcase",
        "base": "https://negd.gov.in",
        "pages": ["/blog/digilocker-the-digital-briefcase-for-indias-authentic-e-documents/"],
        "max_pdfs": 0,
    },
    "blogs_digitalindia": {
        "name": "Digital India: DigiLocker initiative",
        "base": "https://www.digitalindia.gov.in",
        "pages": ["/initiative/digilocker/"],
        "max_pdfs": 0,
    },
    "blogs_india_gov": {
        "name": "National Portal: DigiLocker DMS service",
        "base": "https://www.india.gov.in",
        "pages": ["/services/details/digilocker-digital-document-management"],
        "max_pdfs": 0,
    },
    "blogs_impri": {
        "name": "IMPRI: DigiLocker paperless wallet",
        "base": "https://www.impriindia.com",
        "pages": ["/insights/digilocker/"],
        "max_pdfs": 0,
    },
    "blogs_dpi_overview": {
        "name": "Bino: UPI to ONDC DPI overview",
        "base": "https://bino.bot",
        "pages": ["/blog/india-digital-public-infrastructure-upi-ondc"],
        "max_pdfs": 0,
    },
    "blogs_dpi_anantam": {
        "name": "Anantam IAS: DPI in India",
        "base": "https://anantamias.com",
        "pages": ["/digital-public-infrastructure/"],
        "max_pdfs": 0,
    },
    "blogs_dpi_politics": {
        "name": "Politics for India: India DPI abroad",
        "base": "https://politicsforindia.com",
        "pages": ["/india-and-the-digital-public-infrastructure-abroad-psir"],
        "max_pdfs": 0,
    },
    "blogs_iimb_dpi": {
        "name": "IIMB: State of DPI in India report",
        "base": "https://www.iimb.ac.in",
        "pages": [],
        "max_pdfs": 3,
        "pdfs": ["https://www.iimb.ac.in/cdpg/pdf/State-India-DPI_Report.pdf"],
    },
    # --- more SEBI pages: consultation (4), circulars (6), KYC master ---
    "sebi_consultation": {
        "name": "SEBI consultation papers (sid=4)",
        "base": "https://www.sebi.gov.in",
        "pages": [
            "/sebiweb/home/HomeAction.do?doListing=yes&sid=4",
            "/sebiweb/home/HomeAction.do?doListing=yes&sid=4&n=1",
            "/sebiweb/home/HomeAction.do?doListing=yes&sid=4&n=2",
        ],
        "max_pdfs": 0,
        "list_selector": "sid=4",
    },
    "sebi_circulars": {
        "name": "SEBI circulars (sid=6)",
        "base": "https://www.sebi.gov.in",
        "pages": [
            "/sebiweb/home/HomeAction.do?doListing=yes&sid=6",
            "/sebiweb/home/HomeAction.do?doListing=yes&sid=6&n=1",
        ],
        "max_pdfs": 0,
        "list_selector": "sid=6",
    },
    # --- more DPI / fintech blog articles ---
    "blogs_dpi_vajiram": {
        "name": "Vajiram: DPI components",
        "base": "https://vajiramandravi.com",
        "pages": ["/current-affairs/digital-public-infrastructure-dpi/"],
        "max_pdfs": 0,
    },
    "blogs_dpi_investorcentral": {
        "name": "Investor Central: 5 DPIs in India",
        "base": "https://www.investorcentral.co.uk",
        "pages": ["/5_digital_public_infrastructures_transforming_india_aadhaar_upi_digilocker_account_aggregator_and_ondc_explained/"],
        "max_pdfs": 0,
    },
    "blogs_dpi_flowverify": {
        "name": "FlowVerify: India DPI technical map",
        "base": "https://www.flowverify.co",
        "pages": ["/blog/india-dpi-stack-2026"],
        "max_pdfs": 0,
    },
    "blogs_dpi_plutus": {
        "name": "Plutus IAS: India DPI & AI",
        "base": "https://plutusias.com",
        "pages": ["/the-next-digital-public-infrastructure-dpi-can-india-commoditise-artificial-intelligence"],
        "max_pdfs": 0,
    },
    "blogs_dpi_pib": {
        "name": "PIB: India's DPI",
        "base": "https://static.pib.gov.in",
        "pages": [],
        "max_pdfs": 1,
        "pdfs": ["https://static.pib.gov.in/WriteReadData/specificdocs/documents/2026/mar/doc202636812701.pdf"],
    },
}


class GovFetcher:
    """Robots-polite crawler for Indian government sites."""

    def __init__(self, cache_dir: str = "data/gov_src"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA})
        self._robot_cache = {}

    # ---------------------------- robots.py ----------------------------
    def _allowed(self, site_base: str, url: str) -> bool:
        """Check robots.txt for a URL. Cached per origin. Tolerates errors."""
        from urllib.parse import urlsplit

        o = urlsplit(site_base)
        origin = f"{o.scheme}://{o.netloc}"
        if origin not in self._robot_cache:
            parser = RobotFileParser()
            parser.set_url(f"{origin}/robots.txt")
            try:
                parser.read()
            except Exception:
                parser = None
            self._robot_cache[origin] = parser
        parser = self._robot_cache[origin]
        if parser is None:
            # robots.txt unreachable -> assume allowed (standard behavior)
            return True
        return parser.can_fetch(UA, url)

    def _delay(self, site_base: str):
        from urllib.parse import urlsplit

        o = urlsplit(site_base)
        origin = f"{o.scheme}://{o.netloc}"
        parser = self._robot_cache.get(origin)
        delay = 1.0
        if parser is not None:
            try:
                delay = float(parser.crawl_delay(UA) or 0.5)
            except Exception:
                delay = 0.5
        time.sleep(max(0.4, min(delay, 3.0)))

    # ------------------------------ cache ------------------------------
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
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _abs(base: str, href: str) -> str:
        from urllib.parse import urljoin

        return urljoin(base, href)

    # ---------------------------- fetchers ----------------------------
    def fetch_html(self, url: str, key: str, site_base: str) -> str:
        cached = self._load_cache(key)
        if cached:
            return cached
        if not self._allowed(site_base, url):
            print(f"    robots disallow: {url}")
            return ""
        try:
            r = self.session.get(url, timeout=45)
            r.raise_for_status()
        except Exception as e:
            print(f"    HTTP ERR {url}: {str(e)[:60]}")
            return ""
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = self._clean_text(soup.get_text("\n", strip=True))
        self._cache(key, text)
        self._delay(site_base)
        return text

    def _fetch_html_raw(self, url: str, key: str, site_base: str):
        """Fetch a page and return a BeautifulSoup object of the raw HTML."""
        cached = self._load_cache(key)
        if cached:
            return BeautifulSoup(cached, "lxml")
        if not self._allowed(site_base, url):
            print(f"    robots disallow: {url}")
            return None
        try:
            r = self.session.get(url, timeout=45)
            r.raise_for_status()
        except Exception as e:
            print(f"    HTTP ERR {url}: {str(e)[:60]}")
            return None
        html = r.text
        self._cache(key, html)
        self._delay(site_base)
        return BeautifulSoup(html, "lxml")

    def fetch_pdf(self, url: str, key: str, site_base: str) -> str:
        cached = self._load_cache(key)
        if cached:
            return cached
        if not self._allowed(site_base, url):
            return ""
        try:
            r = self.session.get(url, timeout=60)
            r.raise_for_status()
            reader = PdfReader(BytesIO(r.content))
            text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
        except Exception as e:
            print(f"    PDF ERR {url}: {str(e)[:60]}")
            return ""
        text = self._clean_text(text)
        self._cache(key, text)
        self._delay(site_base)
        return text

    def _collect_pdf_links(self, html: str, base: str) -> list:
        soup = BeautifulSoup(html, "lxml")
        pdfs = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ".pdf" in href.lower() and "javascript" not in href.lower():
                pdfs.append(self._abs(base, href))
        return list(dict.fromkeys(pdfs))

    # ------------------------------ crawl ------------------------------
    def crawl_site(self, key: str, cfg: dict) -> list:
        base = cfg["base"]
        print(f"== {cfg['name']} ({base})")
        docs = []
        seen_pdf = set()
        for page in cfg["pages"]:
            url = self._abs(base, page)
            # SEBI listing pages: need raw HTML (anchor tags), so fetch separately
            if cfg.get("list_selector"):
                soup = self._fetch_html_raw(url, f"{key}_raw{len(docs)}", base)
                follow = []
                if soup is not None:
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        title = a.get_text(strip=True)
                        if (href.startswith("https://www.sebi.gov.in/")
                                and "publiccomment" not in href
                                and title and len(title) > 25):
                            follow.append((title, href))
                print(f"  {key}: {len(follow)} doc links found on {page}")
                for title, href in follow[:25]:
                    body = self.fetch_html(href, f"{key}_doc{len(docs)}", base)
                    if len(body) > 400:
                        docs.append({
                            "title": title,
                            "url": href,
                            "text": body[:50000],
                            "source": key,
                        })
                        print(f"    doc: {len(body)} chars  {title[:50]}")
                continue
            text = self.fetch_html(url, f"{key}_p{len(docs)}", base)
            if len(text) > 300:
                docs.append({
                    "title": f"{cfg['name']} page: {page}",
                    "url": url,
                    "text": text[:50000],
                    "source": key,
                })
                print(f"  page {page}: {len(text)} chars")
            # follow PDFs found on this page
            if text and cfg["max_pdfs"]:
                for pdf in self._collect_pdf_links(text, url)[:cfg["max_pdfs"]]:
                    if pdf in seen_pdf:
                        continue
                    seen_pdf.add(pdf)
                    pdf_text = self.fetch_pdf(pdf, f"{key}_pdf{len(seen_pdf)}", base)
                    if len(pdf_text) > 1500:
                        docs.append({
                            "title": f"{cfg['name']} report {len(seen_pdf)}",
                            "url": pdf,
                            "text": pdf_text[:50000],
                            "source": key,
                        })
                        print(f"    pdf {len(seen_pdf)}: {len(pdf_text)} chars")
        # explicit PDF URLs (direct report links)
        for pdf in cfg.get("pdfs", []):
            if pdf in seen_pdf:
                continue
            seen_pdf.add(pdf)
            pdf_text = self.fetch_pdf(pdf, f"{key}_explicit_{len(seen_pdf)}", base)
            if len(pdf_text) > 1500:
                docs.append({
                    "title": f"{cfg['name']} report {len(seen_pdf)}",
                    "url": pdf,
                    "text": pdf_text[:50000],
                    "source": key,
                })
                print(f"    explicit pdf {len(seen_pdf)}: {len(pdf_text)} chars")
        return docs

    def fetch_all(self) -> dict:
        result = {}
        for key, cfg in SITES.items():
            try:
                result[key] = self.crawl_site(key, cfg)
            except Exception as e:
                print(f"  {key} crawl ERR: {str(e)[:70]}")
                result[key] = []
        return result

    def save(self, data: dict, filename: str = "gov_source_docs.json"):
        path = Path("data") / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        total = sum(len(v) for v in data.values())
        print(f"Saved {total} docs to {path}")

    def robots_report(self):
        """Print robots.txt verdicts for every site (no fetching of content)."""
        print("\n=== robots.txt status (read-only) ===")
        from urllib.parse import urlsplit

        for key, cfg in SITES.items():
            o = urlsplit(cfg["base"])
            origin = f"{o.scheme}://{o.netloc}"
            try:
                parser = RobotFileParser()
                parser.set_url(f"{origin}/robots.txt")
                parser.read()
                allowed = parser.can_fetch(UA, cfg["base"])
                print(f"{key:14s} {cfg['base']:45s} allowed={allowed}")
            except Exception as e:
                print(f"{key:14s} {cfg['base']:45s} robots-unreachable ({str(e)[:30]})")


if __name__ == "__main__":
    import sys

    f = GovFetcher()
    if "--robots" in sys.argv:
        f.robots_report()
    else:
        f.robots_report()
        print("\nStarting crawl...")
        data = f.fetch_all()
        f.save(data)
