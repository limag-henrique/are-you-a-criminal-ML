import os
import glob
import cv2
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def process_image(img_path, face_cascade, alt_cascade, profile_cascade):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return img_path, True, "Unreadable image"
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Face Detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            faces = alt_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            
        if len(faces) == 0:
            return img_path, True, "No face detected"
            
        # 2. Drawing/Sketch Detection
        # Check saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_sat = hsv[:, :, 1].mean()
        
        # Check for pure white pixels (common in sketches)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        white_ratio = cv2.countNonZero(thresh) / (gray.shape[0] * gray.shape[1])
        
        # Heuristic: If it's mostly grayscale and has a lot of pure white, it's likely a sketch
        if mean_sat < 25 and white_ratio > 0.4:
            return img_path, True, "Classified as sketch (low sat, high white background)"
            
        return img_path, False, "Valid face photo"
        
    except Exception as e:
        return img_path, True, f"Error processing: {e}"

def main():
    data_dir = 'data/fbi_criminals'
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        image_files.extend(glob.glob(os.path.join(data_dir, ext)))
        
    logging.info(f"Found {len(image_files)} images to process.")
    
    # Load cascades once
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    deleted_count = 0
    kept_count = 0
    
    # Process sequentially or with threads (OpenCV releases GIL, but cascades might have thread-safety issues in old versions. In recent versions, distinct calls are thread-safe. Let's do it sequentially to be safe and see progress easily since it's fast).
    for img_path in image_files:
        path, should_delete, reason = process_image(img_path, face_cascade, alt_cascade, profile_cascade)
        if should_delete:
            logging.info(f"DELETING {os.path.basename(path)} - Reason: {reason}")
            try:
                os.remove(path)
                deleted_count += 1
            except Exception as e:
                logging.error(f"Failed to delete {path}: {e}")
        else:
            kept_count += 1
            
    logging.info("========================================")
    logging.info("Filtration Completed!")
    logging.info(f"Total images scanned: {len(image_files)}")
    logging.info(f"Deleted non-face/drawings: {deleted_count}")
    logging.info(f"Kept valid face photos: {kept_count}")
    logging.info("========================================")
    
    # Update metadata to remove deleted images
    metadata_path = os.path.join(data_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        new_metadata = {}
        for uid, info in metadata.items():
            valid_images = []
            for img in info.get('images', []):
                local_path = os.path.join(data_dir, img.get('local_filename', ''))
                if os.path.exists(local_path):
                    valid_images.append(img)
            
            # Keep criminal record even if 0 images, or remove if no images?
            # Usually, we want criminals with at least 1 image for the database.
            if valid_images:
                info['images'] = valid_images
                new_metadata[uid] = info
                
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(new_metadata, f, indent=2, ensure_ascii=False)
            
        logging.info(f"Metadata updated. Criminals remaining with valid photos: {len(new_metadata)}")

if __name__ == '__main__':
    main()
