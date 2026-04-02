import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

'''
compute_descriptor => This function extracts L2-normalized feature vectors (descriptors) from two sets of images 
query images and reference images using a neural network model.

The compute_descriptor function takes the following inputs:
query_images, ref_images, model, transform, batch_size and device

query_images - list of "query" image paths
ref_images - list of "database" image paths
model - a neural network (a CNN or ViT backbone) that maps image to a feature vector
transform - a preprocessing pipelint (resizing, normalization)
batch_size, device - standard inference controls

The compute_descriptor function returns the following outputs:
q_desc, ref_desc

q_desc - (num_queries, D) One descriptor per query image
ref_desc - (num_refs, D) One descriptor per reference/DB image

e.g : q_desc -> shape (10, 2048) means -> 10 desccriptor one per image 
2048 -> each descriptor is a single vector of 2048 numbers

the q_desc and ref_desc are feature vectors or 
feature embedding i.e  vectors containing numeric representation of the original images
Hence we get two lists of numbers i.e q_desc, ref_desc that numerically represent the content of an image.
'''

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
                if feats.ndim == 4:
                    feats = feats.flatten(start_dim=1)
                feats = nn.functional.normalize(feats, p=2, dim=-1)
            all_desc.append(feats.cpu().numpy())
        return np.concatenate(all_desc, axis=0)

    q_desc  = extract(query_images, "Query  descriptors")
    ref_desc = extract(ref_images,  "DB descriptors")
    return q_desc, ref_desc

'''
extract function()
1. Setup : Moves model to target device and sets it to eval mode

2. Batching: Iterates over path list in chunks of batch_size, tqdm for progress bar

3. Loading & preprocessing: Each image oven with PIL(Pillow), converted to RGB to guard
against RGBA or grayscale, and passed through transform.
transform -> used for resizing + tensor conversion + normalization.

4. Inference: Batch stacked into a single tensor, moved to the device and
passed through model inside torch.no_grad() to skip gradient tracking and save
memory

5. L2 normalization: Raw features are normalized so every descriptor lies on the unit hypersphere.

6. Collection: Normalized features are moved back to CPU as NumPy arrays and accumulated,
then concatenated into a single(N, D) matrix at the end.
'''
