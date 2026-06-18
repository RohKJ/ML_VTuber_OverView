"""모델 결과 EDA: select_columns가 남긴 모델 피처와 view_count의 관계를 본다."""
import pandas as pd

from eda_selection import eda_spearman_analysis, eda_kruskal_wallis

# view_count와의 관계를 볼 때 피처로 취급하지 않을 컬럼 (식별자 + 타깃 + 채널 규모 보조)
_SKIP_COLS = {"video_id", "channel_id", "view_count", "subscriber_count"}


def run_result_eda(train, test=None):
    """모델이 선택한 피처(select_columns 결과)와 view_count의 관계를 분석한다.

    train에 실제로 남아있는 컬럼을 그대로 피처로 보고, dtype에 따라
    수치형은 Spearman, 범주형은 Kruskal-Wallis로 나눠 살펴본다.
    test를 주면 train → test 순으로 같이 출력한다.
    """
    datasets = [("TRAIN", train)] + ([("TEST", test)] if test is not None else [])
    for label, df in datasets:
        feats = [c for c in df.columns if c not in _SKIP_COLS]
        # 수치형/범주형은 dtype으로 갈라 각각 맞는 검정으로 보낸다
        num_feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
        cat_feats = [c for c in feats
                     if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() >= 2]
        print(f"\n[{label}] 수치형 {len(num_feats)}개 · 범주형 {len(cat_feats)}개")
        if num_feats:
            eda_spearman_analysis(df, features=num_feats)
        if cat_feats:
            eda_kruskal_wallis(df, features=cat_feats)
