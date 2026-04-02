from pathlib import Path
from PIL import Image
import torchvision.transforms as T

'''
PIL - import name for Pillow
PIL - Python Imaging Library - used for converting, manipulating image files.
'''

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def read_images(db_dir="dataset/database", query_dir="dataset/queries", img_size=322):
    db_paths = sorted(p for p in Path(db_dir).rglob("*") 
                      if p.suffix.lower() in IMG_EXTENSIONS)
    q_paths  = sorted(p for p in Path(query_dir).rglob("*") 
                      if p.suffix.lower() in IMG_EXTENSIONS)
    return q_paths, db_paths