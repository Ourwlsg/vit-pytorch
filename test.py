import torch
from vit_pytorch import ViT

v = ViT(
    image_size=256,
    patch_size=32,
    num_classes=5,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

img = torch.randn(1, 3, 256, 256)
mask = torch.ones(1, 8, 8).bool()  # optional mask, designating which patch to attend to

preds = v(img, mask=mask)  # (1, 1000)
print(preds)
