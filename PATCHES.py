import rasterio
import numpy as np
from pathlib import Path
from PIL import Image

PATCH = 64

def normalize(arr):
    arr_min, arr_max = np.min(arr), np.max(arr)
    if arr_max - arr_min == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - arr_min) / (arr_max - arr_min) * 255
    return norm.astype(np.uint8)

with rasterio.open("images/VV.tiff") as vv_src:
    with rasterio.open("images/VH.tiff") as vh_src:
        vv = vv_src.read(1)
        vh = vh_src.read(1)

H, W = vv.shape
output = Path("patches_jpg")
output.mkdir(exist_ok=True)

count = 0
for y in range(0, H - PATCH, PATCH):
    for x in range(0, W - PATCH, PATCH):
        p_vv = vv[y:y+PATCH, x:x+PATCH]
        p_vh = vh[y:y+PATCH, x:x+PATCH]

        rgb_patch = np.stack([
            normalize(p_vv),
            normalize(p_vh),
            np.zeros((PATCH, PATCH), dtype=np.uint8)
        ], axis=-1)

        img = Image.fromarray(rgb_patch)
        img.save(output / f"{count}.jpg")
        count += 1

print("Створено", count, "JPEG патчей")