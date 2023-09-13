import torch
from pytorch_pretrained_vit import ViT
from vector_quantize_pytorch import VectorQuantize
from args import args
from Data.ImageNet_dali import ImageNetDali
from trainer.amp_trainer_dali import validate_ImageNet

vq = VectorQuantize(
    dim = 256,
    codebook_size = 3000,
    codebook_dim = 8      # paper proposes setting this to 32 or as low as 8 to increase codebook usage
)

# criterion = torch.nn.CrossEntropyLoss().cuda()
# vit = ViT('B_16', pretrained=True, num_classes=1000).cuda()
# data = ImageNetDali()
# print(vit)

# acc1, acc5 = validate_ImageNet(data.val_loader, vit, criterion, args)
# print(acc1)
x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = vq(x)
print(quantized.shape, indices.shape, commit_loss.shape)