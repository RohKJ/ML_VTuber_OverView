"""
JNN (Jeonghoon's Nearest Neighbors)
===================================
피처 엔지니어링 모듈.

K 대신 J, 공간적 이웃 대신 시간적 이웃(과거 N개 영상)을 참조하여
채널 단위 통계 피처를 생성한다.
"""

import numpy as np
import pandas as pd


def _trend(values):
    """선형 회귀 기울기 (추세). 유효값이 2개 미만이면 0.0 반환."""
    v = values[~np.isnan(values)]
    if len(v) < 2:
        return 0.0
    x = np.arange(len(v))
    try:
        return np.polyfit(x, v, 1)[0]
    except np.linalg.LinAlgError:
        return 0.0


def _cv(values):
    """변동계수 (Coefficient of Variation). std / mean, mean이 0이면 0.0 반환."""
    v = values[~np.isnan(values)]
    m = np.mean(v)
    if m == 0:
        return 0.0
    return np.std(v) / m


_AGG_FUNC_MAP = {
    'mean': np.mean,
    'std': np.std,
    'trend': _trend,
    'cv': _cv,
    'momentum': 'momentum',  # 후처리: rolling_mean / channel_mean
}

# momentum은 기본에서 제외 (명시적으로 요청해야 계산)
_DEFAULT_AGG_FUNCS = {
    'mean': np.mean,
    'std': np.std,
    'trend': _trend,
}


def make_jnn_features(df, N=5, target_cols=None, agg_funcs=None,
                      drop_insufficient=False, drop_na=True):
    """
    채널 단위로 과거 N개 영상의 통계 피처를 생성한다.

    - 과거 영상이 N개 이상 있으면: 과거 N개 사용
    - 과거 영상이 부족하면: 있는 과거만 사용 (또는 drop)
    - 과거 영상이 0개면: NaN → drop_na=True이면 해당 행 제거

    Parameters
    ----------
    df : DataFrame (published_at, channel_id 컬럼 필수)
    N  : 참조할 영상 수 (기본 5)
    target_cols : 통계를 구할 컬럼 리스트
                  (기본: ['view_count', 'like_count', 'comment_count'])
    agg_funcs : 적용할 집계 함수 dict 또는 함수명 리스트
                (기본: {'mean': np.mean, 'std': np.std, 'trend': _trend})
    drop_insufficient : bool (기본 False)
                True  → 과거 영상이 N개 미만인 행을 제거
                False → 있는 만큼만 계산
    drop_na : bool (기본 True)
                True  → JNN 피처에 NaN이 하나라도 있는 행을 제거
                False → NaN 유지

    Returns
    -------
    result_df : 원본 df에 JNN 피처가 추가된 DataFrame
    """
    if target_cols is None:
        target_cols = [
            'view_count', 'like_count', 'comment_count', '긍정', '부정', '중립']
    if agg_funcs is None:
        agg_funcs = _DEFAULT_AGG_FUNCS
    elif isinstance(agg_funcs, list):
        agg_funcs = {k: _AGG_FUNC_MAP[k] for k in agg_funcs}

    df = df.copy()
    df['published_at'] = pd.to_datetime(df['published_at'])
    df.sort_values(['channel_id', 'published_at'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    jnn_cols = []
    for col in target_cols:
        # shift(1): 현재 행 제외, 과거만 참조
        shifted = df.groupby('channel_id')[col].shift(1)

        roller = shifted.groupby(df['channel_id']).rolling(
            window=N, min_periods=1
        )

        for func_name, func in agg_funcs.items():
            if func_name == 'momentum':
                continue  # 후처리에서 계산
            col_name = f'jnn_{col}_{func_name}'
            if func is np.mean:
                df[col_name] = roller.mean().droplevel(0)
            elif func is np.std:
                df[col_name] = roller.std().droplevel(0)
            else:
                df[col_name] = roller.apply(func, raw=True).droplevel(0)
            jnn_cols.append(col_name)

    # momentum: 채널 평균 대비 최근 수준 (rolling_mean / channel_mean)
    if 'momentum' in agg_funcs:
        for col in target_cols:
            mean_col = f'jnn_{col}_mean'
            momentum_col = f'jnn_{col}_momentum'
            # mean이 아직 없으면 임시 계산
            if mean_col not in df.columns:
                shifted = df.groupby('channel_id')[col].shift(1)
                tmp_roller = shifted.groupby(df['channel_id']).rolling(
                    window=N, min_periods=1)
                df[mean_col] = tmp_roller.mean().droplevel(0)
            # 채널 전체 평균
            ch_mean = df.groupby('channel_id')[col].transform('mean')
            df[momentum_col] = df[mean_col] / ch_mean.replace(0, np.nan)
            jnn_cols.append(momentum_col)

    if drop_insufficient:
        rank = df.groupby('channel_id').cumcount()  # 0-based
        df = df[rank >= N].reset_index(drop=True)

    if drop_na and jnn_cols:
        n_before = len(df)
        df = df.dropna(subset=jnn_cols).reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"[make_jnn_features] NaN 행 {n_dropped}개 제거 ({n_before} → {len(df)})")

    return df