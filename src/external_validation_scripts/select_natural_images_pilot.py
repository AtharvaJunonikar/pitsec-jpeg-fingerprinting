from pathlib import Path
import random
import shutil

SOURCE_DIR = Path("data/external_raw/natural_images")
OUTPUT_DIR = Path("data/external_originals")
SEED = 42

# Use 20 images: 2–3 images from each of the eight categories.
PER_CATEGORY = {
    "airplane": 3,
    "car": 3,
    "cat": 3,
    "dog": 3,
    "flower": 2,
    "fruit": 2,
    "motorbike": 2,
    "person": 2,
}

random.seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Remove only prior Natural-Images pilot files; don't delete other external data.
for old_file in OUTPUT_DIR.glob("natural_*.jpg"):
    old_file.unlink()

selected = []

for category, count in PER_CATEGORY.items():
    candidates = sorted(
        path for path in SOURCE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg"}
        and path.parent.name.lower() == category
    )

    if len(candidates) < count:
        raise SystemExit(
            f"Category '{category}' has only {len(candidates)} JPEGs; need {count}."
        )

    chosen = random.sample(candidates, count)
    selected.extend((category, path) for path in chosen)

manifest_path = OUTPUT_DIR / "natural_images_pilot_manifest.csv"

with manifest_path.open("w", encoding="utf-8") as manifest:
    manifest.write("external_file,category,original_path\n")

    for index, (category, source) in enumerate(selected, start=1):
        destination = OUTPUT_DIR / f"natural_{index:03d}.jpg"
        shutil.copy2(source, destination)
        manifest.write(f'{destination.name},{category},"{source}"\n')

print(f"Copied {len(selected)} external images to: {OUTPUT_DIR}")
print(f"Manifest written to: {manifest_path}")