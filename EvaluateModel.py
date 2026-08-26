import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import torch
import numpy as np

from DefineDataset import id2label
from TrainModel import model, test_dataset, test_loader


### test accuracy, generate confusion matrix

num_classes = 8

import torch.nn.functional as F

model.eval()

confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

model.eval()

device = torch.device("cuda")

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        logits = model(pixel_values=images).logits

        # Resize logits to match mask resolution
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        preds = torch.argmax(logits, dim=1)

        preds = preds.cpu().numpy().flatten()
        masks = masks.cpu().numpy().flatten()

        for t, p in zip(masks, preds):
            if 0 <= t < num_classes:
                confusion[t, p] += 1


### plot the confusion matrix 

print(next(model.parameters()).device)
print(model.training)


print(len(test_dataset))
print(test_dataset.img_dir)
print(test_dataset.mask_dir)



# Normalize each row (true class)
cm = confusion.astype(float)
cm = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(9,8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')


classes = [id2label[i] for i in range(num_classes)]

plt.xticks(np.arange(num_classes), classes, rotation=45, ha='right')
plt.yticks(np.arange(num_classes), classes)

plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Normalized Confusion Matrix")

# Print values in cells
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(
            j,
            i,
            f"{cm[i,j]:.2f}",
            ha="center",
            va="center",
            color="white" if cm[i,j] > 0.5 else "black",
            fontsize=9
        )

plt.tight_layout()
plt.show()



### plot side-by-side sample image and mask

image, true_mask = test_dataset[110] # indicates which image from the test_dataset you want to view

model.eval()

with torch.no_grad():
    logits = model(
        pixel_values=image.unsqueeze(0).to(device)
    ).logits

pred_mask = torch.argmax(logits, dim=1).squeeze().cpu()

colors = [
    "#808080",  # Gravel
    "#d2b48c",  # MC Sand
    "#fff2a8",  # FV Sand
    "#ffffff",  # Quadrat/Paper
    "#5c4033",  # Mud
    "#cc3333",  # Barnacles
    "#33aa33",  # Algae
    "#d9d9d9"   # Shell
]


cmap = ListedColormap(colors)

# Convert image tensor from C,H,W -> H,W,C
img = image.permute(1, 2, 0).cpu().numpy()

# Convert masks to numpy
true_mask_np = true_mask.cpu().numpy()
pred_mask_np = pred_mask.cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[0].axis("off")

#axes[1].imshow(true_mask_np, cmap=cmap, vmin=0, vmax=7)
#axes[1].set_title("True Mask")
#axes[1].axis("off")

axes[1].imshow(pred_mask_np, cmap=cmap, vmin=0, vmax=7)
axes[1].set_title("Predicted Mask")
axes[1].axis("off")

# Create legend
legend_elements = [
    Patch(facecolor=colors[i], edgecolor='black',
          label=id2label[i])
    for i in range(len(id2label))
]

fig.legend(handles=legend_elements,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.05)
)

plt.tight_layout()
plt.show()

