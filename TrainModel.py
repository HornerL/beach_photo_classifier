from DefineDataset import BeachSegDataset, transform, id2label, label2id

### directories
TRAIN_IMAGE_DIR = r"C:\Users\lhorner\Data\Tulalip\Classification_Photos_cropped\labled_tiles\Images\Training_Images_500"
TRAIN_MASK_DIR = r"C:\Users\lhorner\Data\Tulalip\Classification_Photos_cropped\labled_tiles\Labels\Training_Labels_500"

VAL_IMAGE_DIR = r"C:\Users\lhorner\Data\Tulalip\Classification_Photos_cropped\labled_tiles\Images\Validation_500_Images"
VAL_MASK_DIR = r"C:\Users\lhorner\Data\Tulalip\Classification_Photos_cropped\labled_tiles\Labels\Validation_500_Labels"

TEST_IMAGE_DIR = r"C:\Users\lhorner\Data\Tulalip\Classification_Photos_cropped\labled_tiles\Images\Testing_500_Images"
TEST_MASK_DIR = r"C:\Users\lhorner\Data\Tulalip\Classification_Photos_cropped\labled_tiles\Labels\Testing_500_Labels"

MODEL_PATH = r"C:\Users\lhorner\Documents\Python_Scripts\GrainSize_class_Stuff\beach_photo_classifier_github\classif_model_weights.pth"



### create DataLoaders

from torch.utils.data import DataLoader

train_dataset = BeachSegDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, transform=transform)  # optionally add augmentations later

val_dataset = BeachSegDataset(VAL_IMAGE_DIR, VAL_MASK_DIR, transform=None)

test_dataset = BeachSegDataset(TEST_IMAGE_DIR, TEST_MASK_DIR, transform=None)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=2)


### Load a pre-trained SegFormer

from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=8,
    ignore_mismatched_sizes=True,
    id2label=id2label,
    label2id=label2id)

### Training Loop

import torch
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Device:", device)
print("Model device:", next(model.parameters()).device)

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(20):

    model.train()
    total_loss = 0

    for batch_idx, (images, masks) in enumerate(train_loader):

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1} | "
                f"Batch {batch_idx+1}/{len(train_loader)}")

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(pixel_values=images, labels=masks)

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}: "
        f"training loss = {avg_train_loss:.4f}")

        
### save the model weights:
torch.save(model.state_dict(), MODEL_PATH)

