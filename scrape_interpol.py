#!/usr/bin/env python3
"""
Interpol Red Notices Image Scraper
Downloads mugshot photos from the Interpol public Red Notices portal.

API: https://ws-public.interpol.int/notices/v1/red
     (same API used by the official Interpol website)

Key requirement: the Interpol API blocks standard Python HTTP libraries because
they have a different TLS/JA3 fingerprint from real browsers.  This script uses
`curl-cffi` to impersonate Chrome's TLS handshake and bypass that protection.

Dependencies:
    pip install curl-cffi tqdm

Usage:
    python scrape_interpol.py                              # download everything
    python scrape_interpol.py -o data/interpol -c 8       # custom output dir / threads
    python scrape_interpol.py --max-pages 2 --verbose     # test run (first 2 pages)
    python scrape_interpol.py --help
"""

import os
import re
import sys
import json
import math
import time
import glob
import random
import logging
import argparse
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed

from curl_cffi import requests as cffi_requests   # TLS-fingerprint-aware HTTP client
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_API          = "https://ws-public.interpol.int/notices/v1/red"
PAGE_ORIGIN       = "https://www.interpol.int"
PAGE_REFERER      = "https://www.interpol.int/How-we-work/Notices/Red-Notices/View-Red-Notices"
BROWSER_IMPERSONATE = "chrome131"   # curl-cffi impersonation profile

# Number of results per API page — must stay small (e.g. 20) because the
# Interpol CDN caches resultPerPage=160 responses and serves the same
# page-1 data for all page numbers. The website itself uses 20 per page.
RESULTS_PER_PAGE  = 20

CONTENT_TYPE_EXT  = {
    "image/jpeg": ".jpg",
    "image/jpg":  ".jpg",
    "image/png":  ".png",
    "image/gif":  ".gif",
    "image/webp": ".webp",
}

COMMON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin":          PAGE_ORIGIN,
    "Referer":         PAGE_REFERER,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_filename(text: str, fallback: str = "UNKNOWN") -> str:
    """Sanitise a string so it is safe to use as a filename component."""
    if not text:
        return fallback
    s = re.sub(r"[^\w\s\-]", "", str(text), flags=re.UNICODE)
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_").upper() or fallback


def cffi_get_json(url: str, params: dict = None, max_retries: int = 4) -> dict | None:
    """
    GET a JSON endpoint using curl-cffi (browser TLS impersonation).
    Builds the query string directly into the URL (curl_cffi params dict
    can be silently dropped in some builds). Retries with backoff.
    """
    if params:
        url = f"{url}?{urlencode(params)}"

    for attempt in range(1, max_retries + 1):
        try:
            resp = cffi_requests.get(
                url,
                headers=COMMON_HEADERS,
                impersonate=BROWSER_IMPERSONATE,
                timeout=25,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            wait = 3 * (2 ** attempt)  # 6s, 12s, 24s, 48s
            if attempt < max_retries:
                logger.warning(
                    "GET %s failed (attempt %d/%d): %s -- retrying in %ds",
                    url, attempt, max_retries, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error("GET %s failed after %d attempts: %s", url, max_retries, exc)
    return None


def cffi_get_image(url: str, base_path: str, max_retries: int = 3) -> str | None:
    """
    Download an image and save to *base_path*.<ext>.
    Supports resume: skips if a non-empty file already exists.
    Returns the saved filename or None on failure.
    """
    # Resume support: skip already-downloaded images
    for existing in glob.glob(f"{base_path}.*"):
        if os.path.getsize(existing) > 0:
            logger.debug("Skipping (exists): %s", existing)
            return os.path.basename(existing)

    for attempt in range(1, max_retries + 1):
        try:
            resp = cffi_requests.get(
                url,
                headers=COMMON_HEADERS,
                impersonate=BROWSER_IMPERSONATE,
                timeout=30,
                stream=True,
            )
            resp.raise_for_status()

            ct  = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
            ext = CONTENT_TYPE_EXT.get(ct, ".jpg")
            target = f"{base_path}{ext}"

            with open(target, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)

            return os.path.basename(target)

        except Exception as exc:
            wait = 2 ** attempt
            if attempt < max_retries:
                logger.debug("Image download failed (%s) attempt %d: %s", url, attempt, exc)
                time.sleep(wait)
            else:
                logger.warning("Giving up on image %s: %s", url, exc)
    return None


# ---------------------------------------------------------------------------
# Per-notice processing
# ---------------------------------------------------------------------------

def process_notice(notice: dict, output_dir: str) -> tuple[str, dict]:
    """
    Download the mugshot photo for a single Red Notice entry.
    Returns (safe_id, metadata_dict).

    Only uses the thumbnail URL already present in the listing response —
    this avoids a second API call to /images which is aggressively rate-limited.
    The thumbnail is the same portrait photo shown on the Interpol website.
    """
    entity_id = notice.get("entity_id", "")
    forename  = notice.get("forename") or ""
    name      = notice.get("name")    or ""
    links     = notice.get("_links", {})

    safe_id     = entity_id.replace("/", "-")
    safe_name   = safe_filename(f"{name}_{forename}", fallback=safe_id)
    file_prefix = f"{safe_name}_{safe_id}"

    thumbnail_url = (links.get("thumbnail") or {}).get("href")

    downloaded = []

    if thumbnail_url:
        base  = os.path.join(output_dir, f"{file_prefix}")
        fname = cffi_get_image(thumbnail_url, base)
        if fname:
            downloaded.append({"url": thumbnail_url, "file": fname, "type": "thumbnail"})

    metadata = {
        "entity_id":     entity_id,
        "name":          name,
        "forename":      forename,
        "nationalities": notice.get("nationalities"),
        "date_of_birth": notice.get("date_of_birth"),
        "images":        downloaded,
        "_links": {
            "self":      (links.get("self")      or {}).get("href"),
            "images":    (links.get("images")    or {}).get("href"),
            "thumbnail": (links.get("thumbnail") or {}).get("href"),
        },
    }
    return safe_id, metadata


# ---------------------------------------------------------------------------
# Metadata persistence
# ---------------------------------------------------------------------------

def load_metadata(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.error("Failed to load metadata (%s): %s", path, exc)
    return {}


def save_metadata(data: dict, path: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.error("Failed to save metadata (%s): %s", path, exc)


# ---------------------------------------------------------------------------
# Query Iteration
# ---------------------------------------------------------------------------

def _paginate_query_pages(base_query, first_res, delay):
    notices = (first_res.get("_embedded") or {}).get("notices", [])
    if notices:
        yield base_query, 1, notices
    
    total = first_res.get("total", 0)
    total_pages = math.ceil(total / RESULTS_PER_PAGE)
    max_pages = min(total_pages, 8)  # API limit is 160 total results
    
    for page in range(2, max_pages + 1):
        if delay > 0:
            jitter = random.uniform(-0.5, 0.5)
            time.sleep(max(1.0, delay + jitter))
            
        q = {**base_query, "page": page}
        res = cffi_get_json(BASE_API, params=q)
        if not res: break
        
        notices = (res.get("_embedded") or {}).get("notices", [])
        if not notices: break
        yield base_query, page, notices


def iter_notice_pages(delay=0):
    """
    Yields (query_dict, page_num, notices) ensuring we never hit the 160 limit.
    Recursively subdivides searches by Age, then Sex, then Name initial.
    """
    import string
    
    for age in range(18, 121):
        q_age = {"ageMin": age, "ageMax": age, "resultPerPage": RESULTS_PER_PAGE}
        res_age = cffi_get_json(BASE_API, params=q_age)
        if not res_age: continue
        t_age = res_age.get("total", 0)
        if t_age == 0: continue
        
        if t_age <= 160:
            yield from _paginate_query_pages(q_age, res_age, delay)
            continue
            
        # Total > 160, subdivide by sex
        for sex in ["M", "F", "U"]:
            q_sex = {**q_age, "sexId": sex}
            res_sex = cffi_get_json(BASE_API, params=q_sex)
            if not res_sex: continue
            t_sex = res_sex.get("total", 0)
            if t_sex == 0: continue
            
            if t_sex <= 160:
                yield from _paginate_query_pages(q_sex, res_sex, delay)
                continue
                
            # Total > 160, subdivide by name initial
            for letter in string.ascii_uppercase:
                q_name = {**q_sex, "name": f"^{letter}"}
                res_name = cffi_get_json(BASE_API, params=q_name)
                if not res_name: continue
                t_name = res_name.get("total", 0)
                if t_name == 0: continue
                
                if t_name > 160:
                    logger.warning("Query %s still has %d results! Truncating to 160.", q_name, t_name)
                    
                yield from _paginate_query_pages(q_name, res_name, delay)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interpol Red Notices Image Scraper\n"
            "Downloads all publicly available mugshot photos from the Interpol\n"
            "Red Notices portal using the official public API."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="data/interpol_criminals",
        help="Directory to save images and metadata.json (default: data/interpol_criminals)",
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int, default=6,
        help="Concurrent download threads per API page (default: 6)",
    )
    parser.add_argument(
        "-d", "--delay",
        type=float, default=5.0,
        help="Polite delay in seconds between API listing pages (default: 5.0)",
    )
    parser.add_argument(
        "-p", "--max-pages",
        type=int, default=None,
        help="Max listing pages to fetch (default: all)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "scraper.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    logger.info("=" * 60)
    logger.info("Interpol Red Notices Scraper — Starting")
    logger.info("Output directory : %s", os.path.abspath(args.output_dir))
    logger.info("Concurrency      : %d threads", args.concurrency)
    logger.info("Page delay       : %.2f s", args.delay)
    logger.info("=" * 60)

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    db = load_metadata(metadata_path)
    logger.info("Loaded %d existing records from metadata", len(db))

    new_images = 0
    skipped    = 0

    for query, page, notices in iter_notice_pages(args.delay):
        logger.info("Processing %d notices for query %s (page %d) ...", len(notices), query, page)

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(process_notice, notice, args.output_dir): notice
                for notice in notices
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Q:{query.get('ageMin','?')} {query.get('sexId','')}",
                unit="notice",
            ):
                notice = futures[future]
                try:
                    safe_id, meta = future.result()
                except Exception as exc:
                    logger.error("Error processing %s: %s", notice.get("entity_id", "?"), exc)
                    continue

                if safe_id in db:
                    old_n = len(db[safe_id].get("images", []))
                    new_n = len(meta.get("images", []))
                    if new_n > old_n:
                        new_images += new_n - old_n
                    else:
                        skipped += 1
                else:
                    new_images += len(meta.get("images", []))

                db[safe_id] = meta

        save_metadata(db, metadata_path)
        logger.info("DB size: %d records | New images this run: %d", len(db), new_images)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Scraping complete!")
    logger.info("Total records in dataset : %d", len(db))
    logger.info("Newly downloaded images  : %d", new_images)
    logger.info("Skipped (already cached) : %d", skipped)
    logger.info("Metadata saved to        : %s", metadata_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
