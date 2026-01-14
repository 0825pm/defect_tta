"""
Severstal Steel Defect Detection - Dataset Preparation
======================================================
단일 불량 이미지를 train/val로, 다중 불량 이미지를 test로 분할

Usage:
    python prepare_dataset.py --data_root /path/to/severstal --output_dir ./splits
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare Severstal dataset splits')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to Severstal dataset root')
    parser.add_argument('--output_dir', type=str, default='./splits',
                        help='Output directory for split CSVs')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation ratio from single-defect images')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def load_and_analyze_data(data_root):
    """
    train.csv 로드 및 분석
    
    Returns:
        df: 원본 DataFrame
        image_info: 이미지별 정보 (불량 개수, 클래스 목록)
    """
    train_csv_path = os.path.join(data_root, 'train.csv')
    
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"train.csv not found at {train_csv_path}")
    
    df = pd.read_csv(train_csv_path)
    print(f"✓ Loaded train.csv: {len(df)} rows")
    
    # 결측값 처리 (EncodedPixels가 NaN인 경우 = 불량 없음)
    df = df.dropna(subset=['EncodedPixels'])
    print(f"✓ After removing NaN: {len(df)} rows")
    
    # 이미지별 정보 집계
    image_info = df.groupby('ImageId').agg({
        'ClassId': list,
        'EncodedPixels': list
    }).reset_index()
    
    image_info['num_defects'] = image_info['ClassId'].apply(len)
    image_info['defect_classes'] = image_info['ClassId'].apply(lambda x: sorted(set(x)))
    
    # 단일 불량의 경우 클래스 레이블 (stratified split용)
    image_info['primary_class'] = image_info['ClassId'].apply(
        lambda x: x[0] if len(x) == 1 else -1  # 다중 불량은 -1
    )
    
    print(f"✓ Total unique images: {len(image_info)}")
    
    return df, image_info


def analyze_distribution(image_info):
    """데이터 분포 분석 및 출력"""
    print("\n" + "="*60)
    print("📊 데이터 분포 분석")
    print("="*60)
    
    # 불량 개수별 분포
    defect_counts = image_info['num_defects'].value_counts().sort_index()
    print("\n불량 개수별 이미지 수:")
    for num, count in defect_counts.items():
        pct = count / len(image_info) * 100
        print(f"  {num}개 불량: {count:,} ({pct:.1f}%)")
    
    # 단일 불량 클래스 분포
    single_defect = image_info[image_info['num_defects'] == 1]
    class_dist = single_defect['primary_class'].value_counts().sort_index()
    print("\n단일 불량 클래스 분포:")
    for cls, count in class_dist.items():
        pct = count / len(single_defect) * 100
        print(f"  Class {cls}: {count:,} ({pct:.1f}%)")
    
    # 다중 불량 조합 분포
    multi_defect = image_info[image_info['num_defects'] > 1]
    if len(multi_defect) > 0:
        combo_dist = multi_defect['defect_classes'].apply(tuple).value_counts()
        print("\n다중 불량 조합 분포 (상위 10개):")
        for combo, count in combo_dist.head(10).items():
            print(f"  {list(combo)}: {count}")
    
    return {
        'total_images': len(image_info),
        'single_defect': len(single_defect),
        'multi_defect': len(multi_defect),
        'class_distribution': class_dist.to_dict()
    }


def split_dataset(image_info, val_ratio=0.2, seed=42):
    """
    데이터셋 분할
    - 단일 불량 → train/val (stratified by class)
    - 다중 불량 → test
    """
    print("\n" + "="*60)
    print("🔀 데이터셋 분할")
    print("="*60)
    
    # 단일 불량 vs 다중 불량 분리
    single_defect = image_info[image_info['num_defects'] == 1].copy()
    multi_defect = image_info[image_info['num_defects'] > 1].copy()
    
    print(f"\n단일 불량 이미지: {len(single_defect):,}")
    print(f"다중 불량 이미지: {len(multi_defect):,}")
    
    # 단일 불량을 train/val로 stratified split
    train_images, val_images = train_test_split(
        single_defect,
        test_size=val_ratio,
        stratify=single_defect['primary_class'],
        random_state=seed
    )
    
    # Split 태그 추가
    train_images = train_images.copy()
    val_images = val_images.copy()
    multi_defect = multi_defect.copy()
    
    train_images['split'] = 'train'
    val_images['split'] = 'val'
    multi_defect['split'] = 'test'
    
    print(f"\n분할 결과:")
    print(f"  Train: {len(train_images):,} ({len(train_images)/len(single_defect)*100:.1f}%)")
    print(f"  Val:   {len(val_images):,} ({len(val_images)/len(single_defect)*100:.1f}%)")
    print(f"  Test:  {len(multi_defect):,} (다중 불량 전체)")
    
    # 클래스 분포 확인
    print("\n클래스 분포 검증:")
    for split_name, split_df in [('Train', train_images), ('Val', val_images)]:
        dist = split_df['primary_class'].value_counts().sort_index()
        dist_str = ', '.join([f"C{c}:{n}" for c, n in dist.items()])
        print(f"  {split_name}: {dist_str}")
    
    return train_images, val_images, multi_defect


def create_split_csv(df_original, train_info, val_info, test_info, output_dir):
    """
    분할 정보를 CSV로 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Split info CSV (간단 버전)
    all_info = pd.concat([train_info, val_info, test_info], ignore_index=True)
    split_info = all_info[['ImageId', 'split', 'num_defects', 'primary_class']].copy()
    split_info['defect_classes'] = all_info['defect_classes'].apply(
        lambda x: ','.join(map(str, x))
    )
    
    split_info_path = os.path.join(output_dir, 'split_info.csv')
    split_info.to_csv(split_info_path, index=False)
    print(f"\n✓ Saved: {split_info_path}")
    
    # 2-4. 각 split별 상세 CSV (원본 형식 유지)
    for split_name, info_df in [('train', train_info), ('val', val_info), ('test', test_info)]:
        image_ids = set(info_df['ImageId'].tolist())
        split_df = df_original[df_original['ImageId'].isin(image_ids)].copy()
        split_df['split'] = split_name
        
        split_path = os.path.join(output_dir, f'{split_name}.csv')
        split_df.to_csv(split_path, index=False)
        print(f"✓ Saved: {split_path} ({len(split_df)} rows, {len(image_ids)} images)")
    
    return split_info_path


def save_statistics(stats, train_info, val_info, test_info, output_dir):
    """통계 정보 JSON으로 저장"""
    stats['splits'] = {
        'train': {
            'num_images': len(train_info),
            'class_distribution': train_info['primary_class'].value_counts().sort_index().to_dict()
        },
        'val': {
            'num_images': len(val_info),
            'class_distribution': val_info['primary_class'].value_counts().sort_index().to_dict()
        },
        'test': {
            'num_images': len(test_info),
            'num_defects_distribution': test_info['num_defects'].value_counts().sort_index().to_dict()
        }
    }
    
    stats_path = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved: {stats_path}")


def main():
    args = parse_args()
    
    print("="*60)
    print("🔧 Severstal Dataset Preparation")
    print("="*60)
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Seed: {args.seed}")
    
    # 1. 데이터 로드
    df, image_info = load_and_analyze_data(args.data_root)
    
    # 2. 분포 분석
    stats = analyze_distribution(image_info)
    
    # 3. 데이터 분할
    train_info, val_info, test_info = split_dataset(
        image_info, 
        val_ratio=args.val_ratio, 
        seed=args.seed
    )
    
    # 4. CSV 저장
    create_split_csv(df, train_info, val_info, test_info, args.output_dir)
    
    # 5. 통계 저장
    save_statistics(stats, train_info, val_info, test_info, args.output_dir)
    
    print("\n" + "="*60)
    print("✅ Dataset preparation complete!")
    print("="*60)


if __name__ == '__main__':
    main()