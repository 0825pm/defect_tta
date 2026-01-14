"""
Severstal Steel Defect Detection - Dataset & DataLoader
========================================================
단일 불량 → 다중 불량 TTA 연구를 위한 데이터 파이프라인

Usage:
    from dataset import get_dataloaders
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root='/path/to/severstal',
        split_dir='./splits',
        batch_size=16,
        image_size=(256, 512)
    )

Split 구조:
    - Train: 단일 불량 이미지 80%
    - Val:   단일 불량 이미지 20%
    - Test:  다중 불량 이미지 전체
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, Tuple, Dict, List


# ============================================================
# RLE Encoding/Decoding
# ============================================================

def rle_decode(mask_rle: str, shape: Tuple[int, int] = (256, 1600)) -> np.ndarray:
    """
    Run-Length Encoding을 마스크로 디코딩
    
    Args:
        mask_rle: RLE 문자열 (space-separated: start1 length1 start2 length2 ...)
        shape: (height, width) 마스크 크기
    
    Returns:
        mask: Binary mask [H, W]
    """
    if pd.isna(mask_rle) or mask_rle == '':
        return np.zeros(shape, dtype=np.uint8)
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1  # 1-indexed to 0-indexed
    ends = starts + lengths
    
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(shape, order='F')


def rle_encode(mask: np.ndarray) -> str:
    """마스크를 RLE로 인코딩"""
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# ============================================================
# Augmentation Transforms
# ============================================================

def get_train_transforms(image_size: Tuple[int, int] = (256, 512)):
    """학습용 augmentation"""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.1, 
            rotate_limit=15, 
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MedianBlur(blur_limit=3),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.RandomGamma(gamma_limit=(80, 120)),
        ], p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: Tuple[int, int] = (256, 512)):
    """검증/테스트용 transform"""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


# ============================================================
# Dataset Class
# ============================================================

class SeverstalDataset(Dataset):
    """
    Severstal Steel Defect Detection Dataset
    
    Args:
        data_root: 데이터셋 루트 경로
        csv_path: split CSV 경로 (train.csv, val.csv, test.csv)
        transform: albumentations transform
        num_classes: 불량 클래스 수 (기본 4)
        return_meta: True면 메타 정보도 반환
    """
    
    def __init__(
        self,
        data_root: str,
        csv_path: str,
        transform: Optional[A.Compose] = None,
        num_classes: int = 4,
        return_meta: bool = False
    ):
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, 'train_images')
        self.transform = transform
        self.num_classes = num_classes
        self.return_meta = return_meta
        
        # CSV 로드
        self.df = pd.read_csv(csv_path)
        
        # 이미지별로 그룹화
        self.image_ids = self.df['ImageId'].unique().tolist()
        
        # 이미지별 불량 정보 딕셔너리
        self.image_defects = {}
        for image_id in self.image_ids:
            image_df = self.df[self.df['ImageId'] == image_id]
            defects = []
            for _, row in image_df.iterrows():
                defects.append({
                    'class_id': int(row['ClassId']),
                    'rle': row['EncodedPixels']
                })
            self.image_defects[image_id] = defects
        
        print(f"✓ Loaded {len(self.image_ids)} images from {csv_path}")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        image_id = self.image_ids[idx]
        defects = self.image_defects[image_id]
        
        # 이미지 로드
        image_path = os.path.join(self.image_dir, image_id)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_height, original_width = image.shape[:2]
        
        # 마스크 생성 (Multi-class: [H, W, num_classes])
        masks = np.zeros(
            (original_height, original_width, self.num_classes), 
            dtype=np.uint8
        )
        
        for defect in defects:
            class_idx = defect['class_id'] - 1  # 1-indexed to 0-indexed
            mask = rle_decode(defect['rle'], (original_height, original_width))
            masks[:, :, class_idx] = np.maximum(masks[:, :, class_idx], mask)
        
        # Transform 적용
        if self.transform:
            transformed = self.transform(image=image, mask=masks)
            image = transformed['image']
            masks = transformed['mask']
        
        # 마스크를 [num_classes, H, W]로 변환
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks).permute(2, 0, 1).float()
        else:
            masks = masks.permute(2, 0, 1).float()
        
        # 결합 마스크 (배경 포함 semantic segmentation용)
        combined_mask = torch.zeros(masks.shape[1:], dtype=torch.long)
        for c in range(self.num_classes):
            combined_mask[masks[c] > 0.5] = c + 1
        
        result = {
            'image': image,
            'masks': masks,  # [num_classes, H, W]
            'combined_mask': combined_mask,  # [H, W]
            'image_id': image_id,
            'num_defects': len(defects),
        }
        
        if self.return_meta:
            result['defect_classes'] = [d['class_id'] for d in defects]
        
        return result


class SeverstalMultiHeadDataset(Dataset):
    """
    Multi-Head 모델용 Dataset
    각 불량 클래스별로 독립적인 binary segmentation 수행
    """
    
    def __init__(
        self,
        data_root: str,
        csv_path: str,
        transform: Optional[A.Compose] = None,
        num_classes: int = 4
    ):
        self.base_dataset = SeverstalDataset(
            data_root=data_root,
            csv_path=csv_path,
            transform=transform,
            num_classes=num_classes,
            return_meta=True
        )
        self.num_classes = num_classes
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        data = self.base_dataset[idx]
        labels = (data['masks'].sum(dim=(1, 2)) > 0).float()  # [num_classes]
        
        return {
            'image': data['image'],
            'masks': data['masks'],
            'labels': labels,
            'image_id': data['image_id'],
            'num_defects': data['num_defects'],
        }


# ============================================================
# DataLoader Factory
# ============================================================

def get_dataloaders(
    data_root: str,
    split_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 512),
    pin_memory: bool = True,
    multi_head: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Train, Val, Test DataLoader 생성
    
    Args:
        data_root: Severstal 데이터셋 루트 경로
        split_dir: split CSV가 있는 디렉토리
        batch_size: 배치 크기
        num_workers: DataLoader worker 수
        image_size: (height, width) 리사이즈 크기
        pin_memory: CUDA pin memory 사용 여부
        multi_head: True면 MultiHead용 Dataset 사용
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    DatasetClass = SeverstalMultiHeadDataset if multi_head else SeverstalDataset
    
    train_dataset = DatasetClass(
        data_root=data_root,
        csv_path=os.path.join(split_dir, 'train.csv'),
        transform=train_transform
    )
    
    val_dataset = DatasetClass(
        data_root=data_root,
        csv_path=os.path.join(split_dir, 'val.csv'),
        transform=val_transform
    )
    
    test_dataset = DatasetClass(
        data_root=data_root,
        csv_path=os.path.join(split_dir, 'test.csv'),
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"\n✓ DataLoaders created:")
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def get_test_loader_for_tta(
    data_root: str,
    split_dir: str,
    batch_size: int = 1,
    image_size: Tuple[int, int] = (256, 512)
) -> DataLoader:
    """TTA용 테스트 DataLoader (batch_size=1 권장)"""
    test_dataset = SeverstalDataset(
        data_root=data_root,
        csv_path=os.path.join(split_dir, 'test.csv'),
        transform=get_val_transforms(image_size),
        return_meta=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return test_loader


# ============================================================
# Test Code
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--split_dir', type=str, default='./splits')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        image_size=(256, 512)
    )
    
    print("\n" + "="*60)
    print("📊 샘플 데이터 확인")
    print("="*60)
    
    for split_name, loader in [('Train', train_loader), ('Val', val_loader), ('Test', test_loader)]:
        batch = next(iter(loader))
        print(f"\n{split_name}:")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Masks shape: {batch['masks'].shape}")
        print(f"  Combined mask shape: {batch['combined_mask'].shape}")
        print(f"  Num defects: {batch['num_defects']}")