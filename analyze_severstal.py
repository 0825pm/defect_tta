"""
============================================================================
Severstal Steel Defect Detection - 다중 불량 Co-occurrence 분석
============================================================================

데이터 구조:
    train.csv: ImageId, ClassId, EncodedPixels
    - 한 이미지가 여러 행에 있으면 다중 불량
    - ClassId: 1, 2, 3, 4 (4개 불량 유형)
    - EncodedPixels: RLE 형식 마스크 (없으면 해당 클래스 불량 없음)

사용법:
    python analyze_severstal.py --data_root /path/to/severstal_data

출력:
    - 다중 불량 통계
    - Co-occurrence Matrix
    - 조건부 확률 P(Class_j | Class_i)
    - 시각화 히트맵

============================================================================
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# 시각화 라이브러리
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("⚠️  matplotlib/seaborn 없음 - 시각화 건너뜀")


# =============================================================================
# 데이터 로드 및 전처리
# =============================================================================
def load_and_preprocess(csv_path):
    """
    train.csv 로드 및 전처리
    """
    print(f"📂 데이터 로드: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"   총 행 수: {len(df):,}")
    print(f"   컬럼: {list(df.columns)}")
    
    # 결측치 확인
    print(f"\n📊 결측치 확인:")
    print(df.isnull().sum())
    
    # 불량이 있는 행만 필터링 (EncodedPixels가 있는 경우)
    df_defect = df[df['EncodedPixels'].notna()].copy()
    print(f"\n   불량이 있는 행: {len(df_defect):,}")
    
    return df, df_defect


def analyze_multi_defect(df_defect):
    """
    이미지별 다중 불량 분석
    """
    print("\n" + "=" * 60)
    print("🔬 다중 불량 분석")
    print("=" * 60)
    
    # 이미지별 불량 클래스 집계
    image_defects = df_defect.groupby('ImageId')['ClassId'].apply(list).reset_index()
    image_defects['num_defects'] = image_defects['ClassId'].apply(len)
    image_defects['defect_set'] = image_defects['ClassId'].apply(lambda x: tuple(sorted(set(x))))
    
    # 통계
    total_images = len(image_defects)
    single_defect = (image_defects['num_defects'] == 1).sum()
    multi_defect = (image_defects['num_defects'] >= 2).sum()
    
    print(f"\n📊 기본 통계:")
    print(f"   총 불량 이미지: {total_images:,}")
    print(f"   단일 불량 이미지: {single_defect:,} ({single_defect/total_images*100:.1f}%)")
    print(f"   다중 불량 이미지: {multi_defect:,} ({multi_defect/total_images*100:.1f}%)")
    
    # 불량 개수별 분포
    print(f"\n📊 이미지당 불량 개수 분포:")
    defect_count_dist = image_defects['num_defects'].value_counts().sort_index()
    for num, count in defect_count_dist.items():
        ratio = count / total_images * 100
        marker = "  ← 다중 불량" if num >= 2 else ""
        print(f"   {num}개 불량: {count:,} ({ratio:.1f}%){marker}")
    
    # 클래스별 출현 빈도
    print(f"\n📊 클래스별 출현 빈도:")
    class_counts = df_defect['ClassId'].value_counts().sort_index()
    for cls, count in class_counts.items():
        ratio = count / total_images * 100
        print(f"   Class {cls}: {count:,} ({ratio:.1f}%)")
    
    return image_defects


# =============================================================================
# Co-occurrence 분석
# =============================================================================
def build_cooccurrence_matrix(image_defects):
    """
    Co-occurrence Matrix 생성
    """
    print("\n" + "=" * 60)
    print("📊 Co-occurrence Matrix 생성")
    print("=" * 60)
    
    classes = [1, 2, 3, 4]
    n = len(classes)
    
    # 행렬 초기화
    cooccur_matrix = np.zeros((n, n), dtype=int)
    
    # 클래스별 출현 횟수 (대각선)
    class_counts = defaultdict(int)
    
    # Co-occurrence 횟수 (비대각선)
    cooccur_counts = defaultdict(int)
    
    for _, row in image_defects.iterrows():
        defects = list(set(row['ClassId']))
        
        # 클래스별 카운트
        for d in defects:
            class_counts[d] += 1
        
        # 동시 발생 카운트
        if len(defects) >= 2:
            for d1, d2 in combinations(sorted(defects), 2):
                cooccur_counts[(d1, d2)] += 1
    
    # 행렬 채우기
    for i, c1 in enumerate(classes):
        cooccur_matrix[i, i] = class_counts[c1]  # 대각선
        for j, c2 in enumerate(classes):
            if i < j:
                count = cooccur_counts.get((c1, c2), 0)
                cooccur_matrix[i, j] = count
                cooccur_matrix[j, i] = count  # 대칭
    
    # DataFrame 변환
    class_names = [f'Class_{c}' for c in classes]
    df_cooccur = pd.DataFrame(cooccur_matrix, index=class_names, columns=class_names)
    
    print("\n📊 Co-occurrence Matrix (대각선 = 해당 클래스 총 출현 횟수):")
    print(df_cooccur)
    
    return df_cooccur, class_counts, cooccur_counts


def calculate_conditional_probability(class_counts, cooccur_counts):
    """
    조건부 확률 P(Class_j | Class_i) 계산
    """
    print("\n" + "=" * 60)
    print("📊 조건부 확률 P(Col | Row)")
    print("=" * 60)
    
    classes = [1, 2, 3, 4]
    n = len(classes)
    
    cond_prob = np.zeros((n, n))
    
    for i, c_i in enumerate(classes):
        count_i = class_counts[c_i]
        if count_i == 0:
            continue
        
        for j, c_j in enumerate(classes):
            if i == j:
                cond_prob[i, j] = 1.0
            else:
                key = tuple(sorted([c_i, c_j]))
                cooccur = cooccur_counts.get(key, 0)
                cond_prob[i, j] = cooccur / count_i
    
    class_names = [f'Class_{c}' for c in classes]
    df_cond = pd.DataFrame(cond_prob, index=class_names, columns=class_names)
    
    print("\n📊 조건부 확률 행렬:")
    print(df_cond.round(3))
    
    return df_cond


def analyze_defect_combinations(image_defects):
    """
    가장 빈번한 불량 조합 분석
    """
    print("\n" + "=" * 60)
    print("🔥 빈번한 불량 조합")
    print("=" * 60)
    
    # 다중 불량만 필터
    multi_df = image_defects[image_defects['num_defects'] >= 2].copy()
    
    if len(multi_df) == 0:
        print("   다중 불량 이미지가 없습니다.")
        return None
    
    # 조합별 빈도
    combo_counts = multi_df['defect_set'].value_counts()
    
    print(f"\n{'조합':<25} {'횟수':<10} {'비율':<10}")
    print("-" * 45)
    
    total_multi = len(multi_df)
    results = []
    
    for combo, count in combo_counts.head(15).items():
        ratio = count / total_multi * 100
        combo_str = ' + '.join([f'Class_{c}' for c in combo])
        print(f"{combo_str:<25} {count:<10} {ratio:.1f}%")
        results.append({
            'combination': combo,
            'combination_str': combo_str,
            'count': count,
            'ratio': ratio
        })
    
    return pd.DataFrame(results)


# =============================================================================
# 시각화
# =============================================================================
def plot_cooccurrence_heatmap(df_cooccur, output_dir):
    """Co-occurrence Matrix 히트맵"""
    if not HAS_PLOT:
        return
    
    plt.figure(figsize=(8, 6))
    
    # 대각선 마스크
    mask = np.eye(len(df_cooccur), dtype=bool)
    
    sns.heatmap(df_cooccur, annot=True, fmt='d', cmap='YlOrRd',
                mask=mask, square=True, linewidths=0.5,
                cbar_kws={'label': 'Co-occurrence Count'})
    
    plt.title('Severstal Steel - Defect Co-occurrence Matrix', fontsize=14)
    plt.xlabel('Defect Class')
    plt.ylabel('Defect Class')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'cooccurrence_matrix.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n📊 저장: {output_path}")


def plot_conditional_probability(df_cond, output_dir):
    """조건부 확률 히트맵"""
    if not HAS_PLOT:
        return
    
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(df_cond, annot=True, fmt='.3f', cmap='Blues',
                square=True, linewidths=0.5, vmin=0, vmax=1,
                cbar_kws={'label': 'P(Column | Row)'})
    
    plt.title('Severstal Steel - Conditional Probability P(Col|Row)', fontsize=14)
    plt.xlabel('Defect Class (Target)')
    plt.ylabel('Defect Class (Given)')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'conditional_probability.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"📊 저장: {output_path}")


def plot_defect_distribution(image_defects, output_dir):
    """불량 분포 시각화"""
    if not HAS_PLOT:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 이미지당 불량 개수 분포
    defect_counts = image_defects['num_defects'].value_counts().sort_index()
    colors = ['steelblue' if x == 1 else 'coral' for x in defect_counts.index]
    
    axes[0].bar(defect_counts.index, defect_counts.values, color=colors)
    axes[0].set_xlabel('Number of Defects per Image')
    axes[0].set_ylabel('Number of Images')
    axes[0].set_title('Distribution of Defect Count per Image')
    axes[0].set_xticks(defect_counts.index)
    
    # 비율 표시
    total = len(image_defects)
    for i, (x, y) in enumerate(zip(defect_counts.index, defect_counts.values)):
        axes[0].text(x, y + 50, f'{y/total*100:.1f}%', ha='center', fontsize=10)
    
    # 2. 단일 vs 다중 불량 파이 차트
    single = (image_defects['num_defects'] == 1).sum()
    multi = (image_defects['num_defects'] >= 2).sum()
    
    axes[1].pie([single, multi], labels=['Single Defect', 'Multi-Defect'],
                autopct='%1.1f%%', colors=['steelblue', 'coral'],
                explode=[0, 0.05], startangle=90)
    axes[1].set_title('Single vs Multi-Defect Images')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'defect_distribution.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"📊 저장: {output_path}")


# =============================================================================
# 메인 함수
# =============================================================================
def main(data_root, output_dir):
    """
    전체 분석 실행
    """
    print("=" * 70)
    print("🔬 SEVERSTAL STEEL DEFECT - 다중 불량 CO-OCCURRENCE 분석")
    print("=" * 70)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV 경로
    csv_path = os.path.join(data_root, 'train.csv')
    if not os.path.exists(csv_path):
        print(f"❌ train.csv를 찾을 수 없습니다: {csv_path}")
        return
    
    # 1. 데이터 로드
    df, df_defect = load_and_preprocess(csv_path)
    
    # 2. 다중 불량 분석
    image_defects = analyze_multi_defect(df_defect)
    
    # 3. Co-occurrence Matrix
    df_cooccur, class_counts, cooccur_counts = build_cooccurrence_matrix(image_defects)
    
    # 4. 조건부 확률
    df_cond = calculate_conditional_probability(class_counts, cooccur_counts)
    
    # 5. 빈번한 조합 분석
    df_combos = analyze_defect_combinations(image_defects)
    
    # 6. 시각화
    print("\n" + "=" * 60)
    print("📊 시각화 생성")
    print("=" * 60)
    
    plot_cooccurrence_heatmap(df_cooccur, output_dir)
    plot_conditional_probability(df_cond, output_dir)
    plot_defect_distribution(image_defects, output_dir)
    
    # 7. 결과 저장
    print("\n" + "=" * 60)
    print("💾 결과 저장")
    print("=" * 60)
    
    # CSV 저장
    df_cooccur.to_csv(os.path.join(output_dir, 'cooccurrence_matrix.csv'))
    df_cond.to_csv(os.path.join(output_dir, 'conditional_probability.csv'))
    
    if df_combos is not None:
        df_combos.to_csv(os.path.join(output_dir, 'frequent_combinations.csv'), index=False)
    
    # 다중 불량 이미지 상세
    multi_images = image_defects[image_defects['num_defects'] >= 2].copy()
    multi_images['defect_str'] = multi_images['defect_set'].apply(
        lambda x: ', '.join([f'Class_{c}' for c in x])
    )
    multi_images[['ImageId', 'num_defects', 'defect_str']].to_csv(
        os.path.join(output_dir, 'multi_defect_images.csv'), index=False
    )
    
    # 요약 JSON
    summary = {
        'total_defect_images': len(image_defects),
        'single_defect_images': int((image_defects['num_defects'] == 1).sum()),
        'multi_defect_images': int((image_defects['num_defects'] >= 2).sum()),
        'multi_defect_ratio': float((image_defects['num_defects'] >= 2).sum() / len(image_defects) * 100),
        'defect_count_distribution': image_defects['num_defects'].value_counts().to_dict(),
        'class_counts': {f'Class_{k}': int(v) for k, v in class_counts.items()},
        'cooccurrence_counts': {f'{k[0]}-{k[1]}': int(v) for k, v in cooccur_counts.items()}
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   ✅ cooccurrence_matrix.csv")
    print(f"   ✅ conditional_probability.csv")
    print(f"   ✅ frequent_combinations.csv")
    print(f"   ✅ multi_defect_images.csv")
    print(f"   ✅ summary.json")
    
    print(f"\n✅ 분석 완료! 결과는 '{output_dir}/' 폴더에 저장되었습니다.")
    
    # 연구 시사점 출력
    print("\n" + "=" * 70)
    print("💡 연구 시사점")
    print("=" * 70)
    
    multi_ratio = summary['multi_defect_ratio']
    print(f"\n1. 다중 불량 비율: {multi_ratio:.1f}%")
    
    if multi_ratio > 10:
        print("   → 다중 불량이 유의미하게 존재 → Co-occurrence 모델링 필요")
    
    # 가장 강한 상관관계 찾기
    max_cond_prob = 0
    max_pair = None
    for i in range(4):
        for j in range(4):
            if i != j:
                prob = df_cond.iloc[i, j]
                if prob > max_cond_prob:
                    max_cond_prob = prob
                    max_pair = (i+1, j+1)
    
    if max_pair:
        print(f"\n2. 가장 강한 조건부 확률:")
        print(f"   P(Class_{max_pair[1]} | Class_{max_pair[0]}) = {max_cond_prob:.3f}")
        print(f"   → Class_{max_pair[0]} 불량이 있을 때 Class_{max_pair[1]}도 {max_cond_prob*100:.1f}% 확률로 존재")
    
    print("\n3. 제안 연구 방향:")
    print("   - Co-occurrence 기반 손실 함수 설계")
    print("   - 조건부 불량 예측 모듈")
    print("   - Graph Neural Network 기반 관계 모델링")
    
    return summary


# =============================================================================
# CLI 실행
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Severstal Steel Defect 다중 불량 Co-occurrence 분석",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python analyze_severstal.py --data_root /path/to/severstal_data
  
데이터 다운로드:
  kaggle competitions download -c severstal-steel-defect-detection
        """
    )
    
    parser.add_argument(
        '--data_root', 
        type=str, 
        required=True,
        help='Severstal 데이터 루트 경로 (train.csv가 있는 폴더)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='severstal_results',
        help='결과 저장 디렉토리 (기본값: severstal_results)'
    )
    
    args = parser.parse_args()
    
    main(args.data_root, args.output)
