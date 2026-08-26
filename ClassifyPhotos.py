### get cover percentages of photos

import pandas as pd
import numpy as np
import torch
from PIL import Image
import os
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from DefineDataset import id2label, label2id

## directory of images to classify
image_dir = r"C:\Users\lhorner\Data\Tulalip\ShellfishSurvey_photos\OneDrive_1_5-18-2026\Tulalip_MissionBeach_20240816"

## directory where classified masks will by saved
mask_output_dir = r"C:\Users\lhorner\Data\Tulalip\ShellfishSurvey_photos\process_photos\Predicted_Masks_Mission1"

os.makedirs(mask_output_dir, exist_ok=True)

## directory where dataframe of % cover classes will be saved
df_directory = r"C:\Users\lhorner\Data\Tulalip\ShellfishSurvey_photos\process_photos\Predicted_Masks_Mission1\quadrat_photo_CoverClassPcts_mission1.csv"

## directory for saved model weights 

weights_dir = r"C:\Users\lhorner\Documents\Python Scripts\GrainSize_class_Stuff\ModelWeights_SegFormer_1stDraft\segformer_beach_weights_5th.pth"
 
### load saved model weights

device = torch.device("cuda")

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=8,
    ignore_mismatched_sizes=True,
    id2label=id2label,
    label2id=label2id)

model.load_state_dict(torch.load(weights_dir, map_location=device))
model = model.to(device)
model.eval()

device = torch.device("cuda")

model = model.to(device)

results = []

num_classes = 8

model.eval()

image_files = sorted([f for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))])

colors = np.array([
    [128,128,128],   # Gravel
    [210,180,140],   # MC Sand
    [255,242,168],   # FV Sand
    [255,255,255],   # Quadrat
    [92,64,51],      # Mud
    [204,51,51],     # Barnacles
    [51,170,51],     # Algae
    [217,217,217],   # Shell
], dtype=np.uint8)

results = []

tile_size = 512 # size of tiles that image will be split into

with torch.no_grad():

    for image_num, fname in enumerate(image_files):
        
        print(f"\nProcessing {image_num+1}/{len(image_files)} : {fname}")

        # Load image
        img_path = os.path.join(image_dir, fname)
        image = np.array(Image.open(img_path).convert("RGB"))
        height, width = image.shape[:2]
        
        # Crop dimensions to the nearest multiple of 512
        height_crop = (height // tile_size) * tile_size
        width_crop = (width // tile_size) * tile_size
        
        stitched_mask = np.zeros((height_crop, width_crop), dtype=np.uint8)
        
        class_counts = np.zeros(num_classes, dtype=np.int64)
        
        # loop over tiles
        for y in range(0, height_crop, tile_size):
            for x in range(0, width_crop, tile_size):
        
                tile = image[y:y+tile_size, x:x+tile_size]
                        
                # convert to tensor
                tile_tensor = (torch.tensor(tile).permute(2, 0, 1).float()
                    / 255.0).unsqueeze(0).to(device)

                # Predict
                logits = model(pixel_values=tile_tensor).logits
                
                # Upsample to 500x500
                logits = F.interpolate(
                    logits, size=(tile_size, tile_size),
                    mode="bilinear", align_corners=False,)
                
                pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
                
                # Save into stitched mask
                stitched_mask[y:y+tile_size, x:x+tile_size] = pred_mask
                              
                # count pixels
                for c in range(num_classes):
                    class_counts[c] += np.sum(pred_mask == c)
                    
        # save stiched mask
        
        rgb_mask = colors[stitched_mask]
        
        Image.fromarray(rgb_mask).save(
            os.path.join(mask_output_dir, os.path.splitext(fname)[0] + "_mask.png"))

        # convert counts to percent cover
        
        total_pixels = class_counts.sum()
        
        row = {"Filename": fname[:-4]}
        
        for class_id, class_name in id2label.items():
            
            row[class_name] = (class_counts[class_id] / total_pixels * 100)
            
        results.append(row)

df = pd.DataFrame(results)

df.to_csv(df_directory, index=False)


