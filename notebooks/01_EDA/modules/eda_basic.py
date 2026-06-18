"""기초 EDA: 데이터 개요, 품질 점검, 필터 유틸리티."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from _utils import MAIN_COLOR, SUB_COLOR, fmt_kr


# 조회수 퍼널 단계(노출→클릭→체류→확산)별로 묶어둔 분석 대상 피처
EXPOSURE_VARS  = ["video_order", "days_since_published"]
CTR_VARS       = ["concept"]
RETENTION_VARS = ["speech_density", "f0_cv", "final_score", "group", "collaboration"]
SPREAD_VARS    = ["like_ratio", "comment_ratio", "tag_count"]


def _get_num_cat(df):
    """위 피처 목록 중 df에 있는 것을 수치형/범주형으로 나눠 반환."""
    all_num = (EXPOSURE_VARS + CTR_VARS + RETENTION_VARS + SPREAD_VARS)
    num_vars = [c for c in all_num if c in df.columns and df[c].dtype != 'object']
    cat_vars = [c for c in all_num if c in df.columns and df[c].dtype == 'object']
    return num_vars, cat_vars


def plot_outlier_summary(df):
    """IQR 기반 이상값 비율."""
    num_vars, _ = _get_num_cat(df)
    if not num_vars:
        print("수치형 피처 없음"); return

    outlier_rates = {}
    for var in num_vars:
        q1, q3 = df[var].quantile(0.25), df[var].quantile(0.75)
        iqr = q3 - q1
        outlier_rates[var] = ((df[var] < q1 - 1.5 * iqr) |
                              (df[var] > q3 + 1.5 * iqr)).mean() * 100

    rates = pd.Series(outlier_rates).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#ef4444" if v > 5 else "#4361ee" for v in rates.values]
    ax.bar(rates.index, rates.values, color=colors, alpha=0.85)
    ax.axhline(5, color="#f59e0b", lw=1.2, ls="--", label="5% 기준선")
    ax.set_title("IQR 기반 이상값 비율 (%)", fontsize=13)
    ax.set_ylabel("%")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.legend()
    plt.tight_layout(); plt.show()


def eda_yearly_videos(df):
    """연도별 영상 수 / 고유 채널 수 시각화."""
    year = pd.to_datetime(df["published_at"]).dt.year
    video_counts = year.value_counts().sort_index()
    channel_counts = df.groupby(year)["channel_id"].nunique()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bars = ax.bar(video_counts.index.astype(str), video_counts.values,
                  color=MAIN_COLOR, alpha=0.8)
    for bar, val in zip(bars, video_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
    ax.set_xlabel("연도"); ax.set_ylabel("영상 수")
    ax.set_title("연도별 영상 수")

    ax = axes[1]
    bars = ax.bar(channel_counts.index.astype(str), channel_counts.values,
                  color=SUB_COLOR, alpha=0.8)
    for bar, val in zip(bars, channel_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel("연도"); ax.set_ylabel("고유 채널 수")
    ax.set_title("연도별 고유 채널 수")

    plt.tight_layout(); plt.show()
