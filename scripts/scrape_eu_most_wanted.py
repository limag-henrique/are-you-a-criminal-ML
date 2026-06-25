import os
import re
import sys
import json
import time
import logging
import argparse
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuração do Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

BASE_URL = "https://eumostwanted.eu/"
DEFAULT_OUTPUT_DIR = os.path.join("data", "europe_criminals")

def sanitize_filename(name):
    """Limpa o nome para uso seguro como nome de arquivo."""
    if not name:
        return "unknown"
    s = re.sub(r'[^a-zA-Z0-9_\-\s]', '', name)
    s = re.sub(r'[\s\-]+', '_', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_').lower()

def create_session(retries=5, backoff_factor=1.0):
    """Cria uma sessão requests com retries automáticos."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def download_image(session, img_url, filepath):
    """Baixa a imagem se ela não existir."""
    if os.path.exists(filepath):
        logger.debug(f"Arquivo já existe, ignorando: {filepath}")
        return True

    try:
        response = session.get(img_url, stream=True, timeout=15)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        logger.warning(f"Falha ao baixar a imagem {img_url}: {e}")
        return False

def extract_field(row, field_class):
    """Extrai texto do field-content dado uma classe do container."""
    container = row.find('div', class_=field_class)
    if container:
        content = container.find(class_='field-content')
        if content:
            return content.get_text(strip=True)
    return None

def main():
    parser = argparse.ArgumentParser(description="Scraper para o EU Most Wanted")
    parser.add_argument('-o', '--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Diretório de saída (padrão: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument('-d', '--delay', type=float, default=1.0,
                        help="Atraso entre requisições em segundos (padrão: 1.0)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Habilitar saída com mais detalhes (debug log)")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    os.makedirs(args.output_dir, exist_ok=True)
    
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    
    logger.info("Iniciando scraper do EU Most Wanted")
    session = create_session()
    
    logger.info(f"Acessando {BASE_URL}")
    try:
        response = session.get(BASE_URL, timeout=15)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Falha ao acessar o site principal: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    rows = soup.find_all('div', class_='views-row')
    
    logger.info(f"Encontrados {len(rows)} registros na página principal.")
    
    records = []
    downloaded_count = 0
    
    for row in rows:
        nid = extract_field(row, 'views-field-nid')
        name = extract_field(row, 'views-field-title')
        
        if not name or not nid:
            continue
            
        profile_path = extract_field(row, 'views-field-view-node')
        country = extract_field(row, 'views-field-field-enfast-country')
        country_code = extract_field(row, 'views-field-field-country-code')
        locations = extract_field(row, 'views-field-field-propable-locations')
        is_dangerous = extract_field(row, 'views-field-field-is-dangerous')
        status = extract_field(row, 'views-field-field-status')
        crime = extract_field(row, 'views-field-field-crime')
        state_of_case = extract_field(row, 'views-field-field-state-of-case')
        years_sentenced = extract_field(row, 'views-field-field-years-sentenced')
        picture_path = extract_field(row, 'views-field-field-picture')
        
        safe_name = sanitize_filename(name)
        
        record = {
            "id": nid,
            "name": name,
            "profile_url": urljoin(BASE_URL, profile_path) if profile_path else None,
            "country": country,
            "country_code": country_code,
            "probable_locations": locations,
            "is_dangerous": bool(is_dangerous and is_dangerous == "1"),
            "status": status,
            "crime": crime,
            "state_of_case": state_of_case,
            "years_sentenced": years_sentenced,
            "image_url": None,
            "local_filename": None
        }

        if picture_path:
            # O texto na view field picture é a rota da imagem
            img_url = urljoin(BASE_URL, picture_path)
            record['image_url'] = img_url
            
            # Tenta pegar a extensão correta da url, caso contrário usa .jpg
            parsed = urlparse(img_url)
            ext = os.path.splitext(parsed.path)[1]
            if not ext or ext.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                ext = '.jpg'
                
            filename = f"{safe_name}_{nid}{ext.lower()}"
            filepath = os.path.join(args.output_dir, filename)
            
            logger.info(f"Processando [{nid}] {name}...")
            
            if download_image(session, img_url, filepath):
                record['local_filename'] = filename
                downloaded_count += 1
            
            time.sleep(args.delay)
            
        records.append(record)

    # Salvando metadata
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4, ensure_ascii=False)
        logger.info(f"Metadados salvos com sucesso em {metadata_path}")
    except Exception as e:
        logger.error(f"Erro ao salvar arquivo de metadados: {e}")

    logger.info("========================================")
    logger.info("Scraping Concluído!")
    logger.info(f"Registros encontrados: {len(records)}")
    logger.info(f"Imagens baixadas/verificadas: {downloaded_count}")
    logger.info("========================================")

if __name__ == "__main__":
    main()
