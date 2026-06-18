"""피처 선택 근거 분석 + 심화 분석.

찐최종EDA.ipynb 기반: 인라인 코드를 함수화.
모든 함수는 df를 인자로 받아 자체 완결적으로 동작.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, kruskal
from scipy import stats
from itertools import combinations


# 조회수와의 관련성을 검정할 분석 대상 피처 목록
NUMERIC_FEATURES = [
    'like_ratio', 'comment_ratio',
    'tag_count', 'days_since_published', 'video_order',
    'f0_cv', 'speech_density',
    'final_score',
    'jnn_like_count_momentum', 'jnn_like_count_cv',
]

CAT_FEATURES = ['concept', 'group', 'new_category_id', 'category_sub']


def eda_spearman_analysis(df, features=None):
    """수치형 피처 Spearman 상관계수 분석 + 막대 + 산점도.

    Returns
    -------
    result_df : DataFrame  (다른 함수에서 재사용 가능)
    """
    if features is None:
        features = [f for f in NUMERIC_FEATURES if f in df.columns]

    df_num = df[features + ['view_count']].copy()
    df_num.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_num['log_view_count'] = np.log1p(df_num['view_count'])

    results = []
    for col in features:
        tmp = df_num[[col, 'view_count', 'log_view_count']].dropna()
        n = len(tmp)
        r_sp, _ = spearmanr(tmp[col], tmp['view_count'])
        r_pe_log, _ = pearsonr(tmp[col], tmp['log_view_count'])
        results.append({
            '피처': col, 'n': n,
            'Spearman r': round(r_sp, 4),
            'Pearson r (log)': round(r_pe_log, 4),
        })

    result_df = (pd.DataFrame(results)
                 .sort_values('Spearman r', key=abs, ascending=False)
                 .reset_index(drop=True))

    print("Spearman r 해석: |r|≥0.2 강함 / 0.1~0.2 약함 / <0.1 매우 약함")

    # ── 막대 그래프 ──
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#E74C3C' if r < 0 else '#2E86C1' for r in result_df['Spearman r']]
    bars = ax.barh(result_df['피처'], result_df['Spearman r'], color=colors)
    ax.axvline(0, color='black', lw=0.8)
    ax.axvline(0.1, color='gray', lw=0.6, ls='--', alpha=0.5)
    ax.axvline(-0.1, color='gray', lw=0.6, ls='--', alpha=0.5)
    for bar, val in zip(bars, result_df['Spearman r']):
        offset = 0.005 if val >= 0 else -0.005
        ha = 'left' if val >= 0 else 'right'
        ax.text(val + offset, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', ha=ha, fontsize=9)
    ax.set_xlabel('Spearman r (vs view_count)')
    ax.set_title('수치형 피처 Spearman 상관계수')
    ax.set_xlim(-0.5, 0.5)
    plt.tight_layout(); plt.show()

    # ── 산점도 ──
    ncols = 4
    nrows = -(-len(features) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(result_df['피처']):
        ax = axes[i]
        tmp = df_num[[col, 'log_view_count']].dropna()
        ax.scatter(tmp[col], tmp['log_view_count'],
                   alpha=0.08, s=8, color='steelblue', rasterized=True)
        m, b = np.polyfit(tmp[col], tmp['log_view_count'], 1)
        x_line = np.linspace(tmp[col].min(), tmp[col].max(), 200)
        ax.plot(x_line, m * x_line + b, color='crimson', lw=1.5)

        r_val = result_df.loc[result_df['피처'] == col, 'Spearman r'].values[0]
        ax.set_title(f'{col}\nr={r_val:.4f}', fontsize=9)
        ax.set_xlabel(col, fontsize=8); ax.set_ylabel('log(view)', fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('수치형 피처 vs log(view_count) 산점도', fontsize=13, y=1.01)
    plt.tight_layout(); plt.show()


def eda_group_defense(df):
    """group 피처 방어 분석: 그룹 내 CV로 '단순 암기' 반박."""
    grp = df.groupby('group')['view_count'].agg(['mean', 'std', 'median', 'count'])
    grp['CV'] = (grp['std'] / grp['mean']).round(4)
    grp = grp.sort_values('mean', ascending=False).rename(columns={
        'mean': '평균 조회수', 'std': '표준편차', 'median': '중앙값', 'count': '영상 수'
    })
    print(f"CV 평균: {grp['CV'].mean():.4f} (1 이상이면 그룹 내 분산 > 평균)")
    print(f"CV > 1인 그룹: {(grp['CV'] > 1).sum()}개 / {len(grp)}개")

    # 시각화
    order = grp.index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    data_by_group = [np.log1p(df[df['group'] == g]['view_count'].dropna()) for g in order]
    ax.boxplot(data_by_group, vert=False, patch_artist=True,
               boxprops=dict(facecolor='steelblue', alpha=0.6))
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel('log(view_count)')
    ax.set_title('group별 조회수 분포 (log)')

    ax = axes[1]
    cv_vals = grp.loc[order, 'CV']
    bars = ax.barh(order, cv_vals, color='steelblue', alpha=0.7)
    ax.axvline(1.0, color='crimson', lw=1.2, ls='--', label='CV=1 기준선')
    for bar, val in zip(bars, cv_vals):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=9)
    ax.set_xlabel('CV (표준편차 / 평균)')
    ax.set_title('group별 변동계수(CV)')
    ax.legend(fontsize=10)

    plt.tight_layout(); plt.show()


def eda_kruskal_wallis(df, features=None):
    """범주형 피처 Kruskal-Wallis 검정 + 박스플롯.

    Returns
    -------
    kw_df : DataFrame
    """
    if features is None:
        features = [f for f in CAT_FEATURES if f in df.columns]

    kw_results = []
    for col in features:
        tmp = df[[col, 'view_count']].dropna()
        groups = [tmp[tmp[col] == v]['view_count'].values for v in tmp[col].unique()]
        groups = [g for g in groups if len(g) > 0]
        H, p = kruskal(*groups)
        med = tmp.groupby(col)['view_count'].median().sort_values(ascending=False)
        kw_results.append({
            '피처': col, '범주 수': tmp[col].nunique(),
            'H-statistic': round(H, 2), 'p-value': f'{p:.3e}',
            '유의 (p<0.05)': '✓' if p < 0.05 else '✗',
            '중앙값 최대 범주': med.index[0],
            '중앙값 최솟값': int(med.min()), '중앙값 최댓값': int(med.max()),
        })

    kw_df = pd.DataFrame(kw_results)

    # 박스플롯 (범주 20개 초과 시 중앙값 상위 20개만 표시)
    MAX_CAT = 20
    n = len(features)
    ncols = min(n, 2)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(features):
        ax = axes[i]
        tmp = df[[col, 'view_count']].dropna().copy()
        tmp['log_view'] = np.log1p(tmp['view_count'])
        order = (tmp.groupby(col)['log_view'].median()
                 .sort_values(ascending=False).index.tolist())

        truncated = len(order) > MAX_CAT
        if truncated:
            order = order[:MAX_CAT]
            tmp = tmp[tmp[col].isin(order)]

        data = [tmp[tmp[col] == v]['log_view'].values for v in order]
        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.6),
                   medianprops=dict(color='crimson', lw=2),
                   flierprops=dict(marker='o', markersize=3, alpha=0.3))
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('log(view_count)')

        row = kw_df[kw_df['피처'] == col].iloc[0]
        suffix = f'  (상위 {MAX_CAT}개만 표시)' if truncated else ''
        ax.set_title(f'{col}  (범주={row["범주 수"]}, H={row["H-statistic"]}){suffix}')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('범주형 피처별 log(view_count) 분포', fontsize=13, y=1.01)
    plt.tight_layout(); plt.show()


# 심화 분석: 타깃 분포 정규성, 비선형 패턴, group 교호작용, 피처 독립성

def eda_target_distribution(df):
    """view_count 분포 + QQ-plot + Spearman 선택 근거."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 원본 분포
    ax = axes[0]
    clip_val = df['view_count'].quantile(0.99)
    ax.hist(df['view_count'].clip(upper=clip_val), bins=60,
            color='steelblue', alpha=0.7, edgecolor='white', lw=0.3)
    skew_raw = df['view_count'].skew()
    ax.set_title(f'view_count 원본 분포 (상위 1% 클리핑)')
    ax.set_xlabel('view_count')
    ax.text(0.97, 0.95, f'왜도={skew_raw:.2f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # log 분포
    ax = axes[1]
    log_vc = np.log1p(df['view_count'])
    ax.hist(log_vc, bins=60, color='steelblue', alpha=0.7, edgecolor='white', lw=0.3)
    skew_log = log_vc.skew()
    ax.set_title('log(view_count) 분포')
    ax.set_xlabel('log(view_count)')
    ax.text(0.97, 0.95, f'왜도={skew_log:.2f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # QQ-plot
    ax = axes[2]
    (osm, osr), (slope, intercept, r) = stats.probplot(log_vc)
    ax.scatter(osm, osr, color='steelblue', s=8, alpha=0.5)
    ax.plot(osm, slope * np.array(osm) + intercept, color='crimson', lw=1.5)
    ax.set_title(f'QQ-Plot (r={r:.3f})')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')

    plt.suptitle('view_count 정규성 검토 (원본 vs log 변환)', fontsize=12, y=1.02)
    plt.tight_layout(); plt.show()

    print(f"원본 왜도: {skew_raw:.2f} → log 변환 후: {skew_log:.2f}")


def eda_speech_density_nonlinear(df):
    """speech_density 구간별 조회수 중앙값 (비선형 패턴 탐색)."""
    if 'speech_density' not in df.columns:
        print("speech_density 컬럼 없음"); return

    tmp = df[['speech_density', 'view_count']].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    tmp['bin'] = pd.qcut(tmp['speech_density'], q=8, duplicates='drop')
    sp_binned = tmp.groupby('bin', observed=True).agg(
        median=('view_count', 'median'), count=('view_count', 'count')
    ).reset_index()
    sp_binned['label'] = sp_binned['bin'].apply(lambda x: f"{x.left:.0f}~{x.right:.0f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(sp_binned)), sp_binned['median'],
           color='steelblue', alpha=0.7, edgecolor='white')
    overall_med = tmp['view_count'].median()
    ax.axhline(overall_med, color='crimson', lw=1.5, ls='--',
               label=f'전체 중앙값 ({int(overall_med):,})')
    ax.set_xticks(range(len(sp_binned)))
    ax.set_xticklabels(sp_binned['label'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('view_count 중앙값')
    ax.set_title('speech_density 구간별 조회수 중앙값\n(비선형 패턴 탐색)')
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()


def eda_interaction(df, feature='like_ratio', top_n=8):
    """group × 피처 교호작용 시각화."""
    top_groups = df.groupby('group')['view_count'].count().nlargest(top_n).index.tolist()
    tmp = df[df['group'].isin(top_groups)][[
        'group', feature, 'view_count']].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    tmp['log_view'] = np.log1p(tmp['view_count'])

    group_r = {}
    for g in top_groups:
        sub = tmp[tmp['group'] == g]
        if len(sub) > 10:
            r, _ = spearmanr(sub[feature], sub['log_view'])
            group_r[g] = round(r, 3)

    ncols = min(top_n, 4)
    nrows = -(-top_n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    palette = plt.cm.tab10.colors
    for i, g in enumerate(top_groups):
        ax = axes[i]
        sub = tmp[tmp['group'] == g]
        ax.scatter(sub[feature], sub['log_view'],
                   alpha=0.3, s=15, color=palette[i % 10], rasterized=True)
        if len(sub) > 5:
            m, b = np.polyfit(sub[feature], sub['log_view'], 1)
            x_line = np.linspace(sub[feature].min(), sub[feature].max(), 100)
            ax.plot(x_line, m * x_line + b, color='black', lw=1.5)

        r_val = group_r.get(g, float('nan'))
        ax.set_title(f'{g}\nr={r_val:.3f} (n={len(sub)})', fontsize=9)
        ax.set_xlabel(feature, fontsize=8)
        ax.set_ylabel('log(view)', fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'교호작용: group × {feature}', fontsize=12, y=1.02)
    plt.tight_layout(); plt.show()

    print(f"그룹별 {feature} ↔ log(view) Spearman r:")
    for g, r in sorted(group_r.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {g:20s}: r = {r:+.3f}")
    overall_r = spearmanr(tmp[feature], tmp['log_view'])[0]
    print(f"\n전체 Spearman r = {overall_r:.3f}")


def eda_feature_independence(df, features=None):
    """SHAP 상위 피처 간 독립성 확인 (Spearman 히트맵 + 산점도)."""
    if features is None:
        features = ['like_ratio', 'comment_ratio']
    features = [f for f in features if f in df.columns]
    if len(features) < 2:
        print("피처 2개 이상 필요"); return

    df_sub = df[features + ['view_count']].copy()
    df_sub.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_sub['log_view'] = np.log1p(df_sub['view_count'])

    feat_cols = features + ['log_view']
    corr_matrix = pd.DataFrame(index=feat_cols, columns=feat_cols, dtype=float)
    for c1 in feat_cols:
        for c2 in feat_cols:
            tmp = df_sub[[c1, c2]].dropna()
            r = spearmanr(tmp[c1], tmp[c2]).statistic
            if np.ndim(r) == 2:
                r = r[0, 1]
            corr_matrix.loc[c1, c2] = round(float(r), 3)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr_matrix.values.astype(float), cmap='RdBu_r', vmin=-1, vmax=1)
    n = len(feat_cols)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(feat_cols, fontsize=9)
    ax.set_yticklabels(feat_cols, fontsize=9)
    plt.colorbar(im, ax=ax)
    for i in range(n):
        for j in range(n):
            val = float(corr_matrix.iloc[i, j])
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=11, color=color, fontweight='bold')
    ax.set_title('Spearman 상관 히트맵\n|r| < 0.3 → 독립')
    plt.tight_layout(); plt.show()

    print("=== 피처 간 Spearman r ===")
    for c1, c2 in combinations(features, 2):
        r = float(corr_matrix.loc[c1, c2])
        flag = ("✓ 독립" if abs(r) < 0.3
                else "△ 약한 관련" if abs(r) < 0.5
                else "⚠ 공선성 주의")
        print(f"  {c1} ↔ {c2}: r={r:+.3f}  {flag}")


# JNN(이웃 채널 기반) 파생 피처 EDA

def eda_jnn_analysis(df, jnn_cols=None, q=10):
    """JNN 피처 분포 + 분위수별 log(view_count) 트렌드.

    Parameters
    ----------
    df       : make_jnn_features 적용 후 DataFrame (view_count 포함)
    jnn_cols : 시각화할 JNN 컬럼 리스트 (None이면 'jnn_' 접두사 컬럼 자동 탐색)
    q        : 분위수 구간 수 (기본 10)
    """
    if jnn_cols is None:
        jnn_cols = [c for c in df.columns if c.startswith('jnn_')]
    jnn_cols = [c for c in jnn_cols if c in df.columns]

    if not jnn_cols:
        print("JNN 피처 없음 — make_jnn_features 먼저 실행 필요"); return
    if 'view_count' not in df.columns:
        print("view_count 컬럼 없음"); return

    n = len(jnn_cols)
    _, axes = plt.subplots(n, 2, figsize=(14, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, col in enumerate(jnn_cols):
        # ── 왼쪽: 분포 ──
        ax = axes[i, 0]
        data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(data, bins=50, color='steelblue', alpha=0.75, edgecolor='white', lw=0.3)
        ax.axvline(data.mean(),   color='crimson', lw=1.5, ls='--', label=f'mean={data.mean():.3f}')
        ax.axvline(data.median(), color='orange',  lw=1.5, ls=':',  label=f'median={data.median():.3f}')
        ax.set_title(f'{col}  |  분포')
        ax.set_xlabel(col); ax.set_ylabel('count'); ax.legend(fontsize=8)

        # ── 오른쪽: 분위수별 log(view_count) 트렌드 ──
        ax = axes[i, 1]
        tmp = df[[col, 'view_count']].replace([np.inf, -np.inf], np.nan).dropna().copy()
        tmp['log_view'] = np.log1p(tmp['view_count'])
        tmp['bucket'] = pd.qcut(tmp[col], q=q, duplicates='drop')
        agg = tmp.groupby('bucket', observed=True)['log_view'].agg(['mean', 'median', 'count']).reset_index()
        x = range(len(agg))
        ax.plot(x, agg['mean'],   'o-',  color='steelblue', label='mean')
        ax.plot(x, agg['median'], 's--', color='orange',    label='median', alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(b) for b in agg['bucket']], rotation=45, ha='right', fontsize=7)
        ax.set_title(f'{col}  |  분위수별 log(view_count) 트렌드')
        ax.set_xlabel('분위수 구간'); ax.set_ylabel('log(view_count)'); ax.legend(fontsize=8)

        # 구간별 샘플 수 보조 텍스트
        for xi, cnt in zip(x, agg['count']):
            ax.text(xi, ax.get_ylim()[0], f'n={cnt}', ha='center', va='bottom',
                    fontsize=6, color='gray', rotation=90)

    plt.suptitle('JNN 피처 EDA', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()
