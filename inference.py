from sam.segment_anything.build_sam import _build_sam
from sam.segment_anything.utils import ResizeLongestSide
import numpy as np
import torch
sam = _build_sam(768, 4, 16, [7, 15, 23, 31])
coords = ResizeLongestSide(sam.image_encoder.img_size).apply_coords(np.array(torch.rand(10, 2))*512, (512, 512))
sam.cuda().eval()
print(sam([{'image': torch.randn(3, 512, 512).cuda(), 'original_size': (512, 512), 'point_coords':torch.from_numpy(coords)[None, :, :].cuda(), 'point_labels':torch.randn(1, 10, 768).cuda()}]*2, False)[1]["masks"])
import matplotlib.pyplot as plt

plt.imshow(sam([{'image': torch.randn(3, 512, 512).cuda(), 'original_size': (512, 512), 'point_coords':torch.from_numpy(coords)[None, :, :].cuda(), 'point_labels':torch.randn(1, 10, 768).cuda()}], False)[0]["masks"][0].permute(1, 2, 0).cpu().detach().numpy())
plt.savefig("sam.png")