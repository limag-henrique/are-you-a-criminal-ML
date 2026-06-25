#!/usr/bin/env python3
"""
FBI Wanted Criminals Image Scraper
Downloads all criminal photos from the FBI Wanted API and builds a structured dataset with metadata.
"""

import os
import re
import sys
import json
import time
import glob
import logging
import argparse
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
API_URL = "https://api.fbi.gov/wanted/v1/list"
CONTENT_TYPE_MAP = {
    'image/jpeg': '.jpg',
    'image/jpg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/webp': '.webp'
}

def clean_filename(name):
    """
    Sanitizes a string to make it safe for file systems.
    """
    if not name:
        return "UNKNOWN"
    # Remove characters that aren't alphanumeric, spaces, hyphens, or underscores
    s = re.sub(r'[^a-zA-Z0-9_\-\s]', '', name)
    # Replace spaces and hyphens with underscores
    s = re.sub(r'[\s\-]+', '_', s)
    # Remove consecutive underscores
    s = re.sub(r'_+', '_', s)
    return s.strip('_').upper()

def get_file_extension_from_url(url):
    """
    Extracts file extension from URL if it exists and is clean.
    """
    parsed = urlparse(url)
    path = parsed.path
    _, ext = os.path.splitext(path)
    if ext and ext.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        return ext.lower()
    return None

def create_session(retries=5, backoff_factor=1.0, pool_connections=30, pool_maxsize=30):
    """
    Creates a requests Session with automated retries and custom pool sizes.
    """
    session = requests.Session()
    # Add a user-agent to resemble a browser request if needed
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def download_image(session, url, target_base_path):
    """
    Downloads an image, detecting its content type to ensure the correct extension.
    Returns the relative path of the downloaded image if successful, or None.
    """
    # Check if this image was already downloaded with any known extension
    existing_files = glob.glob(f"{target_base_path}.*")
    if existing_files:
        # Check if file has size
        for f in existing_files:
            if os.path.getsize(f) > 0:
                logger.debug(f"Skipping download: {f} already exists.")
                return os.path.basename(f)

    # Fetch headers and content in a streaming fashion to read content-type
    try:
        response = session.get(url, stream=True, timeout=15)
        response.raise_for_status()
        
        # Determine extension
        content_type = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
        ext = CONTENT_TYPE_MAP.get(content_type)
        
        # Fallback to URL extension parsing if Content-Type is generic or missing
        if not ext:
            ext = get_file_extension_from_url(url)
            
        # Hard fallback to .jpg
        if not ext:
            ext = '.jpg'
            
        target_path = f"{target_base_path}{ext}"
        
        # Download and write chunks
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        return os.path.basename(target_path)
    except Exception as e:
        logger.warning(f"Failed to download image {url}: {e}")
        return None

def process_criminal(item, output_dir, session):
    """
    Processes a single criminal item: parses details and downloads all associated images.
    Returns updated metadata dict for this criminal.
    """
    uid = item.get('uid')
    title = item.get('title')
    safe_title = clean_filename(title)
    images = item.get('images', [])
    
    downloaded_images = []
    
    for idx, img_info in enumerate(images):
        # Prefer original, then large, then thumb
        img_url = img_info.get('original') or img_info.get('large') or img_info.get('thumb')
        if not img_url:
            continue
            
        # Target filename prefix
        target_base = os.path.join(output_dir, f"{safe_title}_{uid}_{idx}")
        filename = download_image(session, img_url, target_base)
        
        if filename:
            downloaded_images.append({
                "original_url": img_url,
                "local_filename": filename,
                "caption": img_info.get('caption')
            })
            
    # Extract clean metadata fields for ML
    metadata = {
        "uid": uid,
        "title": title,
        "description": item.get('description'),
        "url": item.get('url'),
        "images": downloaded_images,
        "aliases": item.get('aliases'),
        "subjects": item.get('subjects'),
        "publication": item.get('publication'),
        "dates_of_birth_used": item.get('dates_of_birth_used'),
        "race": item.get('race'),
        "sex": item.get('sex'),
        "hair": item.get('hair'),
        "eyes": item.get('eyes'),
        "height_min": item.get('height_min'),
        "height_max": item.get('height_max'),
        "weight_min": item.get('weight_min'),
        "weight_max": item.get('weight_max'),
        "nationality": item.get('nationality'),
        "occupations": item.get('occupations'),
        "caution": item.get('caution'),
        "remarks": item.get('remarks'),
        "details": item.get('details'),
        "warning_message": item.get('warning_message')
    }
    
    return uid, metadata

def load_existing_metadata(metadata_path):
    """
    Loads existing metadata JSON file if it exists.
    """
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load existing metadata file {metadata_path}: {e}")
    return {}

def save_metadata(metadata_dict, metadata_path):
    """
    Saves metadata dict to JSON file.
    """
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to write metadata file {metadata_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="FBI Wanted Criminals Image Scraper")
    parser.add_argument('-o', '--output-dir', type=str, default='data/fbi_criminals',
                        help="Directory to save downloaded images and metadata (default: data/fbi_criminals)")
    parser.add_argument('-c', '--concurrency', type=int, default=5,
                        help="Number of concurrent download threads (default: 5)")
    parser.add_argument('-p', '--max-pages', type=int, default=None,
                        help="Maximum number of API pages to query (default: all)")
    parser.add_argument('-d', '--delay', type=float, default=0.2,
                        help="Delay in seconds between pages (default: 0.2)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Enable verbose output / debug logging")
    args = parser.parse_args()

    # Configure verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure file logging
    log_file_path = os.path.join(args.output_dir, 'scraper.log')
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Starting FBI Wanted Criminals Image Scraper")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Concurrency level: {args.concurrency}")
    
    # Load existing metadata for resume logic
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    metadata_db = load_existing_metadata(metadata_path)
    logger.info(f"Loaded existing metadata records: {len(metadata_db)}")

    session = create_session(pool_connections=args.concurrency * 2, pool_maxsize=args.concurrency * 2)
    
    page = 1
    total_records = None
    processed_count = 0
    downloaded_images_count = 0
    skipped_criminals = 0
    
    while True:
        if args.max_pages and page > args.max_pages:
            logger.info(f"Reached max pages limit of {args.max_pages}. Stopping.")
            break
            
        logger.info(f"Fetching page {page} from FBI API...")
        
        try:
            response = session.get(API_URL, params={'page': page}, timeout=15)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch page {page}: {e}. Retrying might be needed, stopping loop.")
            break
            
        if total_records is None:
            total_records = data.get('total', 0)
            logger.info(f"Total criminal records found in API: {total_records}")
            
        items = data.get('items', [])
        if not items:
            logger.info("No items returned on this page. Reached the end.")
            break
            
        logger.info(f"Processing {len(items)} items from page {page}...")
        
        # Concurrently download images for items on the current page
        page_results = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            # We map each task
            futures = {
                executor.submit(process_criminal, item, args.output_dir, session): item 
                for item in items
            }
            
            # Using tqdm to show progress bar for downloads on this page
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Page {page} Downloads"):
                item = futures[future]
                try:
                    uid, criminal_meta = future.result()
                    page_results.append((uid, criminal_meta))
                except Exception as exc:
                    logger.error(f"Criminal item {item.get('title')} generated an exception: {exc}")
                    
        # Update metadata database and count stats
        for uid, criminal_meta in page_results:
            # Stats
            if uid in metadata_db:
                # Criminal already existed, count if new images are downloaded
                existing_imgs = len(metadata_db[uid].get('images', []))
                new_imgs = len(criminal_meta.get('images', []))
                if new_imgs > existing_imgs:
                    downloaded_images_count += (new_imgs - existing_imgs)
                else:
                    skipped_criminals += 1
            else:
                downloaded_images_count += len(criminal_meta.get('images', []))
            
            # Save or merge
            metadata_db[uid] = criminal_meta
            processed_count += 1
            
        # Write metadata incrementally per page
        save_metadata(metadata_db, metadata_path)
        logger.info(f"Finished page {page}. Current total database records: {len(metadata_db)}")
        
        # Increment page
        page += 1
        
        # Polite delay
        if args.delay > 0:
            time.sleep(args.delay)
            
    # Execution Summary
    logger.info("========================================")
    logger.info("Scraping Process Completed!")
    logger.info(f"Total processed records in this run: {processed_count}")
    logger.info(f"Skipped records (already fully scraped): {skipped_criminals}")
    logger.info(f"Newly downloaded images: {downloaded_images_count}")
    logger.info(f"Total records in dataset: {len(metadata_db)}")
    logger.info(f"Metadata file updated at: {metadata_path}")
    logger.info("========================================")

if __name__ == "__main__":
    main()
