import os
import time
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path

# Config
BASE_URL = "https://www.gov.br/mj/pt-br/assuntos/sua-seguranca/seguranca-publica/operacoes-integradas/projeto-captura/lista-de-procurados"
OUTPUT_DIR = os.path.join("data", "projeto_captura_criminals")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")

def sanitize_filename(name):
    """Sanitize the name to be used as a filename."""
    # Keep only alphanumeric chars and spaces
    safe_name = "".join([c for c in name if c.isalnum() or c == ' '])
    # Replace spaces with underscores, convert to lower case and remove leading/trailing spaces
    return safe_name.strip().replace(' ', '_').lower()

def scrape():
    print(f"Ensuring output directory exists: {OUTPUT_DIR}")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    session = requests.Session()
    # Use a standard User-Agent to avoid generic 403 Forbidden errors
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    })
    
    records = []
    # To avoid duplicates if a person appears multiple times
    seen_urls = set()
    
    start = 0
    page_size = 15
    
    while True:
        url = f"{BASE_URL}?b_start:int={start}"
        print(f"\n--- Fetching page starting at item {start} ---")
        print(f"URL: {url}")
        
        try:
            response = session.get(url, timeout=15)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break
            
        if response.status_code != 200:
            print(f"Failed to fetch {url}. Status code: {response.status_code}")
            break
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Plone 5+ image tags often have @@images in their src attribute for scales
        images = [img for img in soup.find_all('img') if '@@images' in img.get('src', '')]
        
        if not images:
            print("No more images found on this page. Stopping pagination.")
            break
            
        for img in images:
            img_src = img.get('src')
            name = img.get('alt', 'Unknown').strip()
            
            # Prevent processing the same profile image multiple times
            if img_src in seen_urls:
                continue
            seen_urls.add(img_src)
            
            # Find the parent anchor tag to get the profile URL
            parent_a = img.find_parent('a')
            profile_url = parent_a.get('href') if parent_a else BASE_URL
            
            if not img_src.startswith('http'):
                img_src = urljoin(BASE_URL, img_src)
                
            filename = sanitize_filename(name) + ".jpg"
            if not filename or filename == ".jpg":
                filename = f"unknown_{int(time.time())}.jpg"
                
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            if not os.path.exists(filepath):
                print(f"Downloading: {name} -> {filename}")
                try:
                    img_response = session.get(img_src, stream=True, timeout=10)
                    if img_response.status_code == 200:
                        with open(filepath, 'wb') as f:
                            for chunk in img_response.iter_content(1024):
                                f.write(chunk)
                    else:
                        print(f"  Failed to download image ({img_response.status_code})")
                except Exception as e:
                    print(f"  Error downloading image: {e}")
                
                # Sleep briefly between image requests to be polite
                time.sleep(0.5)
            else:
                print(f"Skipping already downloaded: {filename}")
                
            records.append({
                "name": name,
                "profile_url": profile_url,
                "image_url": img_src,
                "local_filename": filename
            })
            
        # Check for next page
        has_next = False
        next_param = f"b_start:int={start + page_size}"
        for a in soup.find_all('a', href=True):
            if next_param in a['href']:
                has_next = True
                break
                
        if not has_next and len(images) < page_size:
            print("Reached the end of pagination.")
            break
            
        start += page_size
        # Sleep between page requests
        time.sleep(1.5)
        
    print(f"\nSaving metadata to {METADATA_FILE}")
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=4)
        
    print(f"\nScraping complete! Total records collected: {len(records)}")

if __name__ == "__main__":
    scrape()
