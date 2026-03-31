import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def compute_descriptor(query_images, ref_images, model, transform, 
                       batch_size=16, device="cpu"):
    def extract(paths, label):
        model.to(device).eval()
        all_desc = []
        for start in tqdm(range(0, len(paths), batch_size), desc=label):
            batch = [transform(Image.open(p).convert("RGB")) 
                     for p in paths[start:start+batch_size]]
            batch = torch.stack(batch).to(device)
            with torch.no_grad():
                feats = model(batch)
                feats = nn.functional.normalize(feats, p=2, dim=-1)
            all_desc.append(feats.cpu().numpy())
        return np.concatenate(all_desc, axis=0)

    q_desc  = extract(query_images, "Query  descriptors")
    ref_desc = extract(ref_images,  "DB descriptors")
    return q_desc, ref_desc