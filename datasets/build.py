from torch.utils.data import DataLoader
from .cuhkpedes_dataset import CUHKPEDESDataset, CUHKPEDESDatasetV2, CUHKPEDESDatasetMultiPrompt, CUHKPEDESDatasetV4
from .dataset import TextImageDatasetMLM, TextImageDataset
from preprocessing.transforms import build_transforms
import torch
from .sampler import RandomIdentitySampler

def multiprompt_collate_fn(batch):
    captions = torch.stack([sample['caption'] for sample in batch])
    images = torch.stack([sample['image'] for sample in batch])
    noun_chunks = [sample['noun_chunk'] for sample in batch]
    pids = torch.tensor([sample['pid'] for sample in batch])
    return {'caption': captions, 'image': images, 'noun_chunk': noun_chunks, 'pid': pids}         



def build_dataloader(cfg):

    if cfg.TRAIN.MODE == 'multiview':
        transform_train = build_transforms(img_size=cfg.DATA.SIZE, aug=False, is_train=True)
        transform_train_aug = build_transforms(img_size=cfg.DATA.SIZE, aug=True, is_train=True)

        transform_test = build_transforms(img_size=cfg.DATA.SIZE, is_train=False)

        if cfg.TRAIN.ENABLE_TEXT_AUG:
            train_data = CUHKPEDESDatasetV4(cfg.DATA, json_path=cfg.DATA.TRAIN_JSON_PATH, text_augment=True,
                                        transform=transform_train, aug_transform=transform_train_aug, split='train', mode='train')
        else:
            train_data= CUHKPEDESDatasetV2(cfg.DATA, json_path=cfg.DATA.TRAIN_JSON_PATH, 
                                        transform=transform_train, aug_transform=transform_train_aug, split='train', mode='train')
            
        trainloader = DataLoader(dataset=train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True, drop_last=True)

        val_data_text = CUHKPEDESDataset(cfg.DATA, json_path=cfg.DATA.TEST_JSON_PATH, 
                                        transform=transform_test, split='val', mode='text')
        val_data_image = CUHKPEDESDataset(cfg.DATA, json_path=cfg.DATA.TEST_JSON_PATH, 
                                        transform=transform_test, split='val', mode='image')
        
        val_text_loader = DataLoader(dataset=val_data_text, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)
        val_image_loader = DataLoader(dataset=val_data_image, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True) 
        
        return trainloader, val_text_loader, val_image_loader, train_data
    
    if 'mlm' in cfg.TRAIN.MODE: 
        transform_train = build_transforms(img_size=cfg.DATA.SIZE, aug=False, is_train=True)
        transform_train_aug = build_transforms(img_size=cfg.DATA.SIZE, aug=True, is_train=True) if 'multiview' in cfg.TRAIN.MODE else None

        transform_test = build_transforms(img_size=cfg.DATA.SIZE, is_train=False)

        
        train_data= TextImageDatasetMLM(cfg.DATA, json_path=cfg.DATA.TRAIN_JSON_PATH, text_augment='multiview' in cfg.TRAIN.MODE,
                                            transform=transform_train, aug_transform=transform_train_aug, split='train', mode='train', 
                                            multiview='multiview' in cfg.TRAIN.MODE)
            
        trainloader = DataLoader(dataset=train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True, drop_last=True)

        val_data_text = TextImageDataset(cfg.DATA, json_path=cfg.DATA.TEST_JSON_PATH, 
                                        transform=transform_test, split='test', mode='text')
        val_data_image = TextImageDataset(cfg.DATA, json_path=cfg.DATA.TEST_JSON_PATH, 
                                        transform=transform_test, split='test', mode='image')
        
        val_text_loader = DataLoader(dataset=val_data_text, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)
        val_image_loader = DataLoader(dataset=val_data_image, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True) 
        
        return trainloader, val_text_loader, val_image_loader, train_data
    
    elif cfg.TRAIN.MODE == 'clip_multi_prompt':
        transform_train = build_transforms(img_size=cfg.DATA.SIZE, aug=False, is_train=True)
        transform_test = build_transforms(img_size=cfg.DATA.SIZE, is_train=False)

        train_data= CUHKPEDESDatasetMultiPrompt(cfg.DATA, json_path=cfg.DATA.TRAIN_JSON_PATH, 
                                    transform=transform_train, split='train', mode='train')
        trainloader = DataLoader(dataset=train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=multiprompt_collate_fn)

        val_data_text = CUHKPEDESDataset(cfg.DATA, json_path=cfg.DATA.TEST_JSON_PATH, 
                                        transform=transform_test, split='val', mode='text')
        val_data_image = CUHKPEDESDataset(cfg.DATA, json_path=cfg.DATA.TEST_JSON_PATH, 
                                        transform=transform_test, split='val', mode='image')
        
        val_text_loader = DataLoader(dataset=val_data_text, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)
        val_image_loader = DataLoader(dataset=val_data_image, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True) 
        
        return trainloader, val_text_loader, val_image_loader, train_data
    
    elif cfg.TRAIN.MODE == 'clip_multiview_gnn':
        transform_train = build_transforms(img_size=cfg.DATA.SIZE, aug=False, is_train=True)
        transform_train_aug = build_transforms(img_size=cfg.DATA.SIZE, aug=True, is_train=True)
        transform_test = build_transforms(img_size=cfg.DATA.SIZE, is_train=False)
        train_data= CUHKPEDESDatasetV2(cfg.DATA, json_path=cfg.DATA.TRAIN_JSON_PATH, 
                                    transform=transform_train, aug_transform=transform_train_aug, split='train', mode='train')
        trainloader = DataLoader(dataset=train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True, drop_last=True)

        val_data_text = CUHKPEDESDataset(cfg.DATA, json_path=cfg.DATA.TEST_JSON_PATH, 
                                        transform=transform_test, split='val', mode='text')
        val_data_image = CUHKPEDESDataset(cfg.DATA, json_path=cfg.DATA.TEST_JSON_PATH, 
                                        transform=transform_test, split='val', mode='image')
        
        val_text_loader = DataLoader(dataset=val_data_text, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)
        val_image_loader = DataLoader(dataset=val_data_image, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)
        return trainloader, val_text_loader, val_image_loader, train_data
    
    elif cfg.TRAIN.MODE == 'multigrained':
        transform_train = build_transforms(img_size=cfg.DATA.SIZE, aug=False, is_train=True)
        transform_test = build_transforms(img_size=cfg.DATA.SIZE, is_train=False)

        train_data= CUHKPEDESDataset(cfg.DATA, json_path=cfg.DATA.TRAIN_JSON_PATH, 
                                    transform=transform_train, split='train', mode='train')
        trainloader = DataLoader(dataset=train_data, batch_size=cfg.TRAIN.BATCH_SIZE, 
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True, drop_last=True,
                                sampler=RandomIdentitySampler(train_data.data, cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.NUM_INSTANCE))

        val_data_text = CUHKPEDESDataset(cfg.DATA, json_path=cfg.DATA.TEST_JSON_PATH, 
                                        transform=transform_test, split='val', mode='text')
        val_data_image = CUHKPEDESDataset(cfg.DATA, json_path=cfg.DATA.TEST_JSON_PATH, 
                                        transform=transform_test, split='val', mode='image')
        
        val_text_loader = DataLoader(dataset=val_data_text, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)
        val_image_loader = DataLoader(dataset=val_data_image, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)
        
        return trainloader, val_text_loader, val_image_loader, train_data
    else:
        transform_train = build_transforms(img_size=cfg.DATA.SIZE, is_train=True)
        transform_test = build_transforms(img_size=cfg.DATA.SIZE, is_train=False)

        train_data= CUHKPEDESDataset(cfg.DATA, json_path=cfg.DATA.TRAIN_JSON_PATH, 
                                    transform=transform_train, split='train', mode='train')
        trainloader = DataLoader(dataset=train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True, drop_last=True)

        val_data_text = CUHKPEDESDataset(cfg.DATA, json_path=cfg.DATA.TEST_JSON_PATH, 
                                        transform=transform_test, split='val', mode='text')
        val_data_image = CUHKPEDESDataset(cfg.DATA, json_path=cfg.DATA.TEST_JSON_PATH, 
                                        transform=transform_test, split='val', mode='image')
        
        val_text_loader = DataLoader(dataset=val_data_text, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)
        val_image_loader = DataLoader(dataset=val_data_image, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True) 
        
        return trainloader, val_text_loader, val_image_loader, train_data

def build_test_dataloader(cfg):
    transform_test = build_transforms(img_size=cfg.DATA.SIZE, is_train=False)

    test_data_text = CUHKPEDESDataset(cfg.DATA, json_path=cfg.DATA.TEST_JSON_PATH, 
                                    transform=transform_test, split='test', mode='text')
    test_data_image = CUHKPEDESDataset(cfg.DATA, json_path=cfg.DATA.TEST_JSON_PATH, 
                                    transform=transform_test, split='test', mode='image')
    
    test_text_loader = DataLoader(dataset=test_data_text, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)
    test_image_loader = DataLoader(dataset=test_data_image, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True) 
    
    return test_text_loader, test_image_loader