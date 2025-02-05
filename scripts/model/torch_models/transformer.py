from linformer import Linformer
from vit_pytorch.efficient import ViT

def vision_transformer(dim:int=128, image_size:int=48, num_channels:int=1, num_classes:int=7, seq_len:int=37, depth:int=12, heads:int=8, k:int=64):
    efficient_transformer = Linformer(dim=dim, seq_len=seq_len, depth=depth, heads=heads, k=k)
    return ViT(dim=dim, image_size=image_size, patch_size=16, num_classes=num_classes, transformer=efficient_transformer, channels=num_channels)