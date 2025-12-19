import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import SpaceNetDataset
from src.model import SimpleUNet
from src.losses import safe_dice_loss

train_dataset = SpaceNetDataset(
    image_dir=args.image_dir,
    mask_dir=args.mask_dir,
    limit=args.limit
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4
)

model = SimpleUNet().to(device)
nn.init.constant_(model.dec.bias, -5.0)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    model.train()
    epoch_loss = 0.0

    for imgs, masks, _ in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)
        loss = safe_dice_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}: loss={epoch_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "results/model.pth")

