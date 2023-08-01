from . import transforms as T_
import torchvision.transforms as T
from .image_ops import GaussianBlur
from .random_aug import RandAugment

def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    if aug:
        transform = T.Compose([
            #T.Resize((height, width)),
            #T.RandomGrayscale(p=0.2),
            # T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            # T.RandomHorizontalFlip(),
            # RandAugment(n=5, m=10),
            # T.ToTensor(),

            T.Resize((height, width)),
            T.RandomHorizontalFlip(1.),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),

        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            #T.RandomHorizontalFlip(0.5),
            #T.Pad(10),
            #T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            #T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    return transform

# encoding: utf-8
#from . import transforms as T_
#import torchvision.transforms as T


# def build_transforms(img_size=(384, 128), aug=False, is_train=True):
#     height, width = img_size

#     mean = [0.48145466, 0.4578275, 0.40821073]
#     std = [0.26862954, 0.26130258, 0.27577711]

#     if not is_train:
#         transform = T.Compose([
#             T.Resize((height, width)),
#             T.ToTensor(),
#             T.Normalize(mean=mean, std=std),
#         ])
#         return transform

#     # transform for training
#     if aug:
#         transform = T.Compose([
#             T.Resize((height, width)),
#             T.RandomRotation(10),
#             #T_.RandomCrop((height, width)),
#             T.ToTensor(),
#             T.Normalize(mean=mean, std=std),
#             T.RandomErasing(scale=(0.02, 0.4), value=mean),
#         ])
#     else:
#         transform = T.Compose([
#             T.Resize((height, width)),
#             #T.RandomHorizontalFlip(0.5),
#             T.ToTensor(),
#             T.Normalize(mean=mean, std=std),
#         ])
#     return transform
