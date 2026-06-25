import cv2
import numpy as np
import os
from pathlib import Path

def extract_photo(image_path, output_path):
    # Use np.fromfile and cv2.imdecode to handle special characters in paths on Windows
    stream = open(image_path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    stream.close()

    if img is None:
        print(f"Error reading {image_path}")
        return False
        
    # The background is a uniform gray color. We can sample it from the top-left area 
    # just below the red banner. e.g., x=10, y=60
    bg_color = img[60, 10]
    
    # Calculate the absolute difference between the image and the background color
    diff = cv2.absdiff(img, np.full_like(img, bg_color))
    
    # Convert to grayscale and threshold to create a mask of non-background objects
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_box = None
    max_area = 0
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        
        # We know the photo is on the left (small x), it is relatively large
        if x < 200 and w > 100 and h > 100 and area > max_area:
            # Ignore the red "PROCURADO" banner at the top
            if y < 50 and w > 300:
                continue
            max_area = area
            best_box = (x, y, w, h)
            
    if best_box:
        x, y, w, h = best_box
        photo = img[y:y+h, x:x+w]
        
        # Use cv2.imencode and write to file to handle special characters on Windows
        is_success, im_buf_arr = cv2.imencode(".jpg", photo)
        if is_success:
            im_buf_arr.tofile(str(output_path))
            return True
        else:
            print(f"Error encoding {output_path}")
            return False
    else:
        print(f"Could not find photo in {image_path}")
        return False

def main():
    input_dir = Path("data/projeto_captura_criminals")
    output_dir = Path("data/projeto_captura_photos")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist.")
        return
        
    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    print(f"Found {len(images)} images to process.")
    
    success_count = 0
    for img_path in images:
        out_path = output_dir / img_path.name
        if extract_photo(img_path, out_path):
            success_count += 1
            
    print(f"Successfully extracted {success_count} photos out of {len(images)}.")

if __name__ == "__main__":
    main()
