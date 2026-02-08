import numpy as np
from PIL import Image
from pathlib import Path
import time

PATCH = 64
PATCHES_DIR = Path("patches_jpg1")

N_Y = 260
N_X = 416
TOTAL_PATCHES = N_Y * N_X

NEW_H = N_Y * PATCH
NEW_W = N_X * PATCH

combined = np.zeros((NEW_H, NEW_W, 3), dtype=np.uint8)

start = time.time()

for idx in range(TOTAL_PATCHES):
    i = idx // N_X
    j = idx % N_X

    y = i * PATCH
    x = j * PATCH

    path = PATCHES_DIR / f"{idx}.jpg"
    if not path.exists():
        raise FileNotFoundError(path)

    patch = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    if patch.shape != (PATCH, PATCH, 3):
        continue

    combined[y:y + PATCH, x:x + PATCH] = patch

    if (idx + 1) % 5000 == 0:
        print(f"{idx + 1}/{TOTAL_PATCHES} patches, {time.time() - start:.1f}s")

Image.fromarray(combined).save("combined_output_final1.jpg", quality=95)