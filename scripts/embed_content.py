"""Builds the text and image embedding tables for every game node.

Text: sentence-transformers all-MiniLM-L6-v2 on the description, falling back to tags, then title.
Image: CLIP ViT-B/32 on each game's Steam header art.

Needs: pip install sentence-transformers transformers pillow requests
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
import torch
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", required=True, help="Folder with games.csv and games_metadata.json")
parser.add_argument("--work-dir", default="work")
parser.add_argument("--batch-size", type=int, default=64)
args = parser.parse_args()

work = Path(args.work_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
app_ids = json.loads((work / "app_ids.json").read_text())

meta = pd.read_json(Path(args.data_dir) / "games_metadata.json", lines=True).set_index("app_id")
titles = pd.read_csv(Path(args.data_dir) / "games.csv").set_index("app_id")["title"]


def text_for(app_id):
    if app_id in meta.index:
        row = meta.loc[app_id]
        if isinstance(row.get("description"), str) and row["description"].strip():
            return row["description"]
        if isinstance(row.get("tags"), list) and row["tags"]:
            return ", ".join(row["tags"])
    return str(titles.get(app_id, ""))


from sentence_transformers import SentenceTransformer  # noqa: E402

text_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
text = text_model.encode([text_for(a) for a in app_ids], batch_size=128, convert_to_tensor=True, show_progress_bar=True)
torch.save(text.cpu(), work / "text_embeddings.pt")

from transformers import CLIPModel, CLIPProcessor  # noqa: E402

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def header(app_id):
    try:
        r = requests.get(f"https://cdn.akamai.steamstatic.com/steam/apps/{app_id}/header.jpg", timeout=10)
        return Image.open(BytesIO(r.content)).convert("RGB") if r.ok else None
    except Exception:
        return None


image = torch.zeros(len(app_ids), clip.config.projection_dim)
missing = 0
with ThreadPoolExecutor(32) as pool:
    for start in range(0, len(app_ids), args.batch_size):
        batch = app_ids[start:start + args.batch_size]
        imgs = list(pool.map(header, batch))
        keep = [i for i, img in enumerate(imgs) if img is not None]
        missing += len(batch) - len(keep)
        if keep:
            inputs = processor(images=[imgs[i] for i in keep], return_tensors="pt").to(device)
            with torch.no_grad():
                feats = clip.get_image_features(**inputs).cpu()
            image[[start + i for i in keep]] = feats
torch.save(image, work / "image_embeddings.pt")
print(f"text {tuple(text.shape)}  image {tuple(image.shape)}  missing images {missing}")
