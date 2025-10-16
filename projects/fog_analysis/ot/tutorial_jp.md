# 霧ガウシアンデータの最適輸送解析チュートリアル

このチュートリアルでは、霧のガウシアンデータに対して最適輸送（Optimal Transport）理論を用いた解析を行う方法を学びます。

## 目次
1. [準備と環境設定](#準備と環境設定)
2. [データの理解](#データの理解)
3. [基本的な解析](#基本的な解析)
4. [可視化](#可視化)
5. [応用例](#応用例)
6. [実践的な使用例](#実践的な使用例)

---

## 準備と環境設定

### 必要なライブラリのインストール

```bash
pip install -r requirements.txt
```

### インポートとデータの準備

```python
from gaussian_ot_analysis import GaussianFogOTAnalysis
from visualization import FogOTVisualization
from applications import FogOTApplications
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データの読み込み
csv_path = "../../data/train_fog_all.csv"
analyzer = GaussianFogOTAnalysis(csv_path)
```

---

## データの理解

### 使用する特徴量

本チュートリアルでは、以下の特徴量のみを使用します：

- **opacity**: 不透明度（霧の濃度）
- **scale_x, scale_y, scale_z**: スケールパラメータ（霧の広がり）
- **f_dc_0, f_dc_1, f_dc_2**: 色係数（霧の色）

位置（position）と回転（rotation）は使用しません。

### データの確認

```python
# 霧データの基本情報を確認
print(f"霧ガウシアンの総数: {len(analyzer.fog_gaussians['opacity'])}")
print(f"不透明度の範囲: {analyzer.fog_gaussians['opacity'].min():.3f} - {analyzer.fog_gaussians['opacity'].max():.3f}")

# スケールの確認
scales = analyzer.fog_gaussians['scale']
print(f"スケールの平均: {np.mean(scales, axis=0)}")

# 色の確認
colors = analyzer.fog_gaussians['color']
print(f"色の範囲: RGB({colors.min(axis=0)}, {colors.max(axis=0)})")
```

---

## 基本的な解析

### 1. 不透明度ベースの解析

高密度霧と低密度霧の分布を最適輸送で比較します：

```python
# 不透明度による解析
opacity_result = analyzer.opacity_based_ot_analysis()

print(f"高密度霧と低密度霧間のワッサーシュタイン距離: {opacity_result['wasserstein_distance']:.4f}")
print(f"高密度霧の数: {len(opacity_result['high_opacity_values'])}")
print(f"低密度霧の数: {len(opacity_result['low_opacity_values'])}")
```

### 2. 多次元特徴解析

不透明度、スケール、色を組み合わせた解析：

```python
# 多次元特徴解析を実行
multi_result = analyzer.multi_feature_ot_analysis()
print(f"多次元ワッサーシュタイン距離: {multi_result['wasserstein_distance']:.6f}")
```

### 3. カスタム特徴解析関数

位置と回転を除外した新しい解析関数を定義：

```python
def analyze_fog_features_only(analyzer):
    """
    不透明度、スケール、色のみを使用した解析
    """
    # 特徴量の抽出
    opacities = analyzer.fog_gaussians['opacity'].reshape(-1, 1)
    scales = analyzer.fog_gaussians['scale']  # 3次元
    colors = analyzer.fog_gaussians['color']  # 3次元
    
    # スケールの大きさを計算
    scale_magnitudes = np.linalg.norm(scales, axis=1).reshape(-1, 1)
    
    # 特徴量を結合（不透明度 + スケール大きさ + 色）
    features = np.hstack([
        opacities / np.std(opacities),  # 正規化
        scale_magnitudes / np.std(scale_magnitudes),
        colors / np.std(colors, axis=0)
    ])
    
    return features

# カスタム解析の実行
features = analyze_fog_features_only(analyzer)
print(f"使用した特徴量の次元: {features.shape[1]}")
print(f"特徴量: [不透明度, スケール大きさ, R, G, B]")
```

---

## 可視化

### 1. 基本的な可視化

```python
# 可視化オブジェクトの作成
visualizer = FogOTVisualization(analyzer)

# 不透明度解析の可視化
opacity_result = analyzer.opacity_based_ot_analysis()
fig_opacity = visualizer.plot_opacity_analysis(opacity_result)
plt.show()
```

### 2. カスタム可視化関数

```python
def plot_fog_features_analysis(analyzer):
    """
    不透明度、スケール、色の分布を可視化
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 不透明度分布
    axes[0, 0].hist(analyzer.fog_gaussians['opacity'], bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('不透明度')
    axes[0, 0].set_ylabel('頻度')
    axes[0, 0].set_title('不透明度の分布')
    axes[0, 0].grid(True, alpha=0.3)
    
    # スケール分布
    scales = analyzer.fog_gaussians['scale']
    scale_magnitudes = np.linalg.norm(scales, axis=1)
    axes[0, 1].hist(scale_magnitudes, bins=50, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('スケール大きさ')
    axes[0, 1].set_ylabel('頻度')
    axes[0, 1].set_title('スケールの分布')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 色分布（RGB）
    colors = analyzer.fog_gaussians['color']
    axes[1, 0].hist(colors[:, 0], bins=50, alpha=0.7, color='red', label='R')
    axes[1, 0].hist(colors[:, 1], bins=50, alpha=0.7, color='green', label='G')
    axes[1, 0].hist(colors[:, 2], bins=50, alpha=0.7, color='blue', label='B')
    axes[1, 0].set_xlabel('色値')
    axes[1, 0].set_ylabel('頻度')
    axes[1, 0].set_title('色の分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 不透明度とスケールの関係
    axes[1, 1].scatter(analyzer.fog_gaussians['opacity'], scale_magnitudes, 
                       alpha=0.6, s=10)
    axes[1, 1].set_xlabel('不透明度')
    axes[1, 1].set_ylabel('スケール大きさ')
    axes[1, 1].set_title('不透明度とスケールの関係')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# カスタム可視化の実行
fig_custom = plot_fog_features_analysis(analyzer)
plt.show()
```

---

## 応用例

### 1. 霧の品質評価（特徴量限定版）

```python
class FogQualityAssessmentLimited:
    """
    不透明度、スケール、色のみを使用した霧品質評価
    """
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def assess_fog_consistency(self):
        """霧の一貫性評価"""
        # 不透明度の一貫性
        opacities = self.analyzer.fog_gaussians['opacity']
        opacity_consistency = 1.0 / (1.0 + np.std(opacities) / np.mean(opacities))
        
        # スケールの一貫性
        scales = self.analyzer.fog_gaussians['scale']
        scale_magnitudes = np.linalg.norm(scales, axis=1)
        scale_consistency = 1.0 / (1.0 + np.std(scale_magnitudes) / np.mean(scale_magnitudes))
        
        # 色の一貫性
        colors = self.analyzer.fog_gaussians['color']
        color_std = np.mean(np.std(colors, axis=0))
        color_consistency = 1.0 / (1.0 + color_std)
        
        # 総合スコア
        overall_score = (opacity_consistency + scale_consistency + color_consistency) / 3
        
        return {
            'opacity_consistency': opacity_consistency,
            'scale_consistency': scale_consistency,
            'color_consistency': color_consistency,
            'overall_score': overall_score
        }

# 品質評価の実行
quality_assessor = FogQualityAssessmentLimited(analyzer)
quality_result = quality_assessor.assess_fog_consistency()

print("=== 霧品質評価結果 ===")
print(f"不透明度の一貫性: {quality_result['opacity_consistency']:.4f}")
print(f"スケールの一貫性: {quality_result['scale_consistency']:.4f}")
print(f"色の一貫性: {quality_result['color_consistency']:.4f}")
print(f"総合スコア: {quality_result['overall_score']:.4f}")
```

### 2. 特徴量ベースの異常検出

```python
def detect_fog_anomalies_by_features(analyzer, contamination=0.05):
    """
    不透明度、スケール、色を使用した異常検出
    """
    # 特徴量の準備
    opacities = analyzer.fog_gaussians['opacity'].reshape(-1, 1)
    scales = analyzer.fog_gaussians['scale']
    colors = analyzer.fog_gaussians['color']
    
    # 特徴量を結合
    features = np.hstack([
        opacities / np.std(opacities),
        scales / np.std(scales, axis=0),
        colors / np.std(colors, axis=0)
    ])
    
    # 各サンプルの異常度を計算（他のサンプルとの平均距離）
    n_samples = len(features)
    anomaly_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        distances = np.linalg.norm(features - features[i], axis=1)
        anomaly_scores[i] = np.mean(distances)
    
    # 閾値の設定
    threshold = np.percentile(anomaly_scores, (1 - contamination) * 100)
    anomalies = anomaly_scores > threshold
    
    return {
        'anomaly_scores': anomaly_scores,
        'anomalies': anomalies,
        'threshold': threshold,
        'anomaly_indices': np.where(anomalies)[0]
    }

# 異常検出の実行
anomaly_result = detect_fog_anomalies_by_features(analyzer)
n_anomalies = len(anomaly_result['anomaly_indices'])

print(f"検出された異常な霧パターン: {n_anomalies}個")
print(f"異常率: {n_anomalies/len(analyzer.fog_gaussians['opacity'])*100:.2f}%")
print(f"異常度閾値: {anomaly_result['threshold']:.4f}")
```

### 3. 霧の分類

```python
def classify_fog_by_characteristics(analyzer, n_classes=4):
    """
    特徴量による霧の分類
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # 特徴量の準備
    opacities = analyzer.fog_gaussians['opacity'].reshape(-1, 1)
    scales = analyzer.fog_gaussians['scale']
    colors = analyzer.fog_gaussians['color']
    
    # 特徴量を結合
    features = np.hstack([opacities, scales, colors])
    
    # 標準化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # クラスタリング
    kmeans = KMeans(n_clusters=n_classes, random_state=42)
    labels = kmeans.fit_predict(features_scaled)
    
    # 各クラスタの特徴を分析
    cluster_stats = {}
    for i in range(n_classes):
        mask = labels == i
        if np.any(mask):
            cluster_stats[f'クラスタ{i+1}'] = {
                '数': np.sum(mask),
                '平均不透明度': np.mean(analyzer.fog_gaussians['opacity'][mask]),
                '平均スケール': np.mean(np.linalg.norm(scales[mask], axis=1)),
                '平均色': np.mean(colors[mask], axis=0)
            }
    
    return labels, cluster_stats

# 分類の実行
labels, cluster_stats = classify_fog_by_characteristics(analyzer)

print("=== 霧の分類結果 ===")
for cluster_name, stats in cluster_stats.items():
    print(f"\n{cluster_name}:")
    print(f"  数: {stats['数']}")
    print(f"  平均不透明度: {stats['平均不透明度']:.4f}")
    print(f"  平均スケール: {stats['平均スケール']:.4f}")
    print(f"  平均色 (RGB): [{stats['平均色'][0]:.3f}, {stats['平均色'][1]:.3f}, {stats['平均色'][2]:.3f}]")
```

---

## 実践的な使用例

### 完全な解析パイプライン

```python
def complete_fog_analysis_pipeline(csv_path):
    """
    霧データの完全な解析パイプライン
    """
    print("=== 霧ガウシアンデータ解析 ===\n")
    
    # 1. データの読み込み
    print("1. データを読み込み中...")
    analyzer = GaussianFogOTAnalysis(csv_path)
    
    # 2. 基本統計
    print("2. 基本統計情報:")
    opacities = analyzer.fog_gaussians['opacity']
    scales = analyzer.fog_gaussians['scale']
    colors = analyzer.fog_gaussians['color']
    
    print(f"   霧ガウシアン数: {len(opacities)}")
    print(f"   不透明度: 平均={np.mean(opacities):.4f}, 標準偏差={np.std(opacities):.4f}")
    print(f"   スケール: 平均={np.mean(np.linalg.norm(scales, axis=1)):.4f}")
    print(f"   色: R={np.mean(colors[:, 0]):.3f}, G={np.mean(colors[:, 1]):.3f}, B={np.mean(colors[:, 2]):.3f}")
    
    # 3. 品質評価
    print("\n3. 品質評価:")
    quality_assessor = FogQualityAssessmentLimited(analyzer)
    quality_result = quality_assessor.assess_fog_consistency()
    print(f"   総合品質スコア: {quality_result['overall_score']:.4f}")
    
    # 4. 異常検出
    print("\n4. 異常検出:")
    anomaly_result = detect_fog_anomalies_by_features(analyzer)
    n_anomalies = len(anomaly_result['anomaly_indices'])
    print(f"   異常パターン: {n_anomalies}個 ({n_anomalies/len(opacities)*100:.2f}%)")
    
    # 5. 分類
    print("\n5. 霧の分類:")
    labels, cluster_stats = classify_fog_by_characteristics(analyzer)
    for cluster_name, stats in cluster_stats.items():
        print(f"   {cluster_name}: {stats['数']}個")
    
    # 6. 可視化
    print("\n6. 可視化を生成中...")
    fig = plot_fog_features_analysis(analyzer)
    fig.savefig('fog_analysis/ot/fog_features_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("\n解析完了！結果は fog_analysis/ot/ に保存されました。")
    
    return {
        'analyzer': analyzer,
        'quality': quality_result,
        'anomalies': anomaly_result,
        'classification': (labels, cluster_stats)
    }

# パイプラインの実行
if __name__ == "__main__":
    results = complete_fog_analysis_pipeline("../../data/train_fog_all.csv")
```

---

## 注意事項とヒント

### 1. パフォーマンスの最適化

```python
# 大きなデータセットの場合はサブサンプリングを使用
def subsample_fog_data(analyzer, sample_size=10000):
    """霧データのサブサンプリング"""
    n_total = len(analyzer.fog_gaussians['opacity'])
    if n_total > sample_size:
        indices = np.random.choice(n_total, sample_size, replace=False)
        for key in analyzer.fog_gaussians:
            if len(analyzer.fog_gaussians[key].shape) == 1:
                analyzer.fog_gaussians[key] = analyzer.fog_gaussians[key][indices]
            else:
                analyzer.fog_gaussians[key] = analyzer.fog_gaussians[key][indices]
    return analyzer
```

### 2. 結果の保存

```python
def save_analysis_results(results, output_dir="fog_analysis/ot/results"):
    """解析結果をCSVファイルに保存"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 品質評価結果
    quality_df = pd.DataFrame([results['quality']])
    quality_df.to_csv(f"{output_dir}/quality_assessment.csv", index=False)
    
    # 異常検出結果
    anomaly_df = pd.DataFrame({
        'anomaly_score': results['anomalies']['anomaly_scores'],
        'is_anomaly': results['anomalies']['anomalies']
    })
    anomaly_df.to_csv(f"{output_dir}/anomaly_detection.csv", index=False)
    
    print(f"結果を {output_dir} に保存しました。")
```

---

## まとめ

このチュートリアルでは、霧のガウシアンデータの不透明度、スケール、色特徴量を使用した最適輸送解析の方法を学びました。主な学習内容：

1. **基本解析**: 不透明度ベースの比較と多次元特徴解析
2. **可視化**: 特徴量分布の可視化と関係性の分析
3. **応用**: 品質評価、異常検出、分類
4. **実践**: 完全な解析パイプラインの構築

これらの手法を使用することで、霧データの特性を深く理解し、品質改善や異常パターンの検出に活用できます。