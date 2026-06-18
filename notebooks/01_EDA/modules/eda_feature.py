"""피처 vs 조회수 분석: duration, comment, like, category, tag, upload_time."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from IPython.display import display

from _utils import MAIN_COLOR, SUB_COLOR, fmt_kr, add_stat_box, add_regression

# 영상 길이·태그 수·매력도 점수를 구간으로 묶을 때 쓰는 경계값과 라벨
DURATION_BINS   = [0, 60, 180, 300, 600, 1200, 1800, 3600, np.inf]
DURATION_LABELS = ["0~1분", "1~3분", "3~5분", "5~10분",
                   "10~20분", "20~30분", "30~60분", "1시간+"]

TAG_BINS   = [-1, 0, 3, 7, 12, 20, 30, np.inf]
TAG_LABELS = ["0", "1-3", "4-7", "8-12", "13-20", "21-30", "31+"]

SCORE_BINS   = [0, 30, 40, 50, 60, 70, 80, 100]
SCORE_LABELS = ["~30", "30~40", "40~50", "50~60", "60~70", "70~80", "80~"]

CATEGORY_MAP = {
    1: "Animation & Film", 2: "Automotive", 10: "Music",
    15: "Animals", 17: "Sports", 18: "Short Films", 19: "Travel",
    20: "Gaming", 21: "Vlog", 22: "Lifestyle / Talk", 23: "Comedy",
    24: "Entertainment / Broadcast", 25: "News", 26: "Tutorials",
    27: "Education", 28: "IT / Science",
}


def eda_duration_vs_views(df):
    """영상 길이 1분 단위 구간별 조회수 (3~12분, 중앙값 바차트 + log boxplot)."""
    d = df.dropna(subset=["duration", "view_count"]).copy()
    d = d[(d["duration"] > 0) & (d["view_count"] > 0)]

    # 1분 단위 bins: 3~12분 (180~720초)
    bins = list(range(180, 721, 60))  # [180, 240, 300, ..., 720]
    labels = [f"{m}분" for m in range(3, 12)]  # ["3분", "4분", ..., "11분"]
    d = d[(d["duration"] >= 180) & (d["duration"] < 720)]
    d["dur_group"] = pd.cut(d["duration"], bins=bins, labels=labels, right=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 중앙값 barplot
    ax = axes[0]
    grouped = (d.groupby("dur_group", observed=False)["view_count"]
                .agg(["median", "count"]).reset_index())
    bars = ax.bar(grouped["dur_group"].astype(str), grouped["median"],
                  color=MAIN_COLOR, alpha=0.8)
    for bar, n in zip(bars, grouped["count"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'N={n:,}', ha='center', va='bottom', fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
    ax.set_title("영상 길이 1분 단위 조회수 중앙값 (3~12분)")
    ax.set_xlabel("영상 길이"); ax.set_ylabel("조회수 (중앙값)")

    # log boxplot
    ax = axes[1]
    sns.boxplot(data=d, x="dur_group", y="view_count",
                hue="dur_group", palette="Set2", legend=False,
                showfliers=False, ax=ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
    ax.set_title("영상 길이 1분 단위 조회수 분포 (Log, 3~12분)")
    ax.set_xlabel("영상 길이"); ax.set_ylabel("조회수")

    plt.tight_layout(); plt.show()


def eda_comment_vs_views(df):
    """채널 평균 댓글수 vs 평균 조회수 (로그 산점도 + 댓글률 분포)."""
    d = df.dropna(subset=["comment_count", "view_count"]).copy()
    ch = d.groupby("channel_id").agg(
        avg_comment=("comment_count", "mean"),
        avg_view=("view_count", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    v = ch[(ch["avg_comment"] > 0) & (ch["avg_view"] > 0)]
    ax.scatter(v["avg_comment"], v["avg_view"],
               alpha=0.6, s=50, color=MAIN_COLOR, edgecolors="white", lw=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
    ax.set_xlabel("평균 댓글수"); ax.set_ylabel("평균 조회수")
    ax.set_title("채널 평균 댓글수 vs 평균 조회수 (Log)")
    r2, sp = add_regression(ax, v["avg_comment"], v["avg_view"], log=True)
    if r2 is not None:
        add_stat_box(ax, f"R² = {r2:.3f}\nSpearman ρ = {sp:.3f}\nN = {len(v)}")

    ax = axes[1]
    v2 = ch[(ch["avg_view"] > 0) & (ch["avg_comment"] > 0)].copy()
    v2["comment_rate"] = v2["avg_comment"] / v2["avg_view"] * 100
    sns.histplot(np.log10(v2["comment_rate"]), kde=True, bins=25,
                 color=MAIN_COLOR, alpha=0.7, ax=ax)
    med = v2["comment_rate"].median()
    ax.axvline(np.log10(med), color=SUB_COLOR, ls="--", lw=2,
               label=f"중앙값: {med:.2f}%")
    ax.legend(fontsize=9)
    ax.set_title("채널별 댓글률 분포")
    ax.set_xlabel("댓글률 log₁₀(%)"); ax.set_ylabel("채널 수")

    plt.tight_layout(); plt.show()


def eda_like_vs_views(df):
    """채널 평균 좋아요수 vs 평균 조회수 (로그 산점도 + 좋아요률 분포)."""
    d = df.dropna(subset=["like_count", "view_count"]).copy()
    ch = d.groupby("channel_id").agg(
        avg_like=("like_count", "mean"),
        avg_view=("view_count", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    v = ch[(ch["avg_like"] > 0) & (ch["avg_view"] > 0)]
    ax.scatter(v["avg_like"], v["avg_view"],
               alpha=0.6, s=50, color=MAIN_COLOR, edgecolors="white", lw=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
    ax.set_xlabel("평균 좋아요수"); ax.set_ylabel("평균 조회수")
    ax.set_title("채널 평균 좋아요수 vs 평균 조회수 (Log)")
    r2, sp = add_regression(ax, v["avg_like"], v["avg_view"], log=True)
    if r2 is not None:
        add_stat_box(ax, f"R² = {r2:.3f}\nSpearman ρ = {sp:.3f}\nN = {len(v)}")

    ax = axes[1]
    v2 = ch[(ch["avg_view"] > 0) & (ch["avg_like"] > 0)].copy()
    v2["like_rate"] = v2["avg_like"] / v2["avg_view"] * 100
    sns.histplot(np.log10(v2["like_rate"]), kde=True, bins=25,
                 color=MAIN_COLOR, alpha=0.7, ax=ax)
    med = v2["like_rate"].median()
    ax.axvline(np.log10(med), color=SUB_COLOR, ls="--", lw=2,
               label=f"중앙값: {med:.2f}%")
    ax.legend(fontsize=9)
    ax.set_title("채널별 좋아요률 분포")
    ax.set_xlabel("좋아요률 log₁₀(%)"); ax.set_ylabel("채널 수")

    plt.tight_layout(); plt.show()


def eda_category_vs_views(df):
    """카테고리별 조회수 (요약 테이블 + log boxplot)."""
    d = df.dropna(subset=["view_count"]).copy()
    d = d[d["view_count"] > 0]
    d["category_name"] = d["category_id"].map(CATEGORY_MAP).fillna("Unknown")

    cat_summary = (
        d.groupby(["category_id", "category_name"], dropna=False)
         .agg(n=("video_id", "count"),
              median_view=("view_count", "median"),
              mean_view=("view_count", "mean"))
         .reset_index()
         .sort_values("median_view", ascending=False)
    )
    display(cat_summary)

    cat_order = cat_summary["category_name"].tolist()
    n_map = dict(zip(cat_summary["category_name"], cat_summary["n"]))
    cat_labels = [f"{name}  (N={n_map[name]:,})" for name in cat_order]

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=d, y="category_name", x="view_count", order=cat_order,
                hue="category_name", palette="Set2", legend=False,
                showfliers=False, ax=ax)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
    ax.set_yticklabels(cat_labels)
    ax.set_ylabel("카테고리"); ax.set_xlabel("조회수 (Log)")
    ax.set_title("카테고리별 조회수 분포")
    plt.tight_layout(); plt.show()


def eda_tag_count_vs_views(df):
    """태그 개수 vs 조회수 (평균+중앙값 꺾은선 + 구간별 log boxplot)."""
    d = df.dropna(subset=["view_count"]).copy()
    d = d[d["view_count"] > 0]
    d["tag_group"] = pd.cut(d["tag_count"], bins=TAG_BINS, labels=TAG_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 평균+중앙값 꺾은선
    ax = axes[0]
    tag_stats = (d.groupby("tag_count")["view_count"]
                  .agg(["count", "mean", "median", "std"]).reset_index())
    p = tag_stats[tag_stats["count"] >= 30].copy()
    p["se"] = p["std"] / np.sqrt(p["count"])
    p["ci_lo"] = (p["mean"] - 1.96 * p["se"]).clip(lower=0)
    p["ci_hi"] = p["mean"] + 1.96 * p["se"]

    ax.plot(p["tag_count"], p["mean"], marker="o", ms=4, color=MAIN_COLOR,
            label="평균", zorder=3)
    ax.fill_between(p["tag_count"], p["ci_lo"], p["ci_hi"],
                    alpha=0.15, color=MAIN_COLOR, label="95% CI")
    ax.plot(p["tag_count"], p["median"], marker="s", ms=4, color=SUB_COLOR,
            ls="--", label="중앙값", zorder=3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
    ax.set_xlabel("태그 수"); ax.set_ylabel("조회수")
    ax.set_title("태그 수별 평균·중앙값 조회수 (N≥30)")
    ax.legend(fontsize=9)

    # 구간별 boxplot
    ax = axes[1]
    grp_n = d.groupby("tag_group", observed=False)["view_count"].count()
    tag_labels = [f"{lbl}\n(N={grp_n.get(lbl, 0):,})" for lbl in TAG_LABELS]
    sns.boxplot(data=d, x="tag_group", y="view_count",
                hue="tag_group", palette="Set2", legend=False,
                order=TAG_LABELS, showfliers=False, ax=ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
    ax.set_xticklabels(tag_labels, fontsize=9)
    ax.set_xlabel("태그 수 구간"); ax.set_ylabel("조회수 (Log)")
    ax.set_title("태그 구간별 조회수 분포")

    plt.tight_layout(); plt.show()


def eda_upload_time_vs_views(df):
    """업로드 시간대별 평균+중앙값 조회수 + 업로드 주기별 log boxplot."""
    d = df.dropna(subset=["upload_hour", "avg_upload_interval_min"]).copy()
    d = d[d["view_count"] > 0]
    d["upload_hour"] = d["upload_hour"].astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 시간대별
    ax = axes[0]
    hourly = (d.groupby("upload_hour")["view_count"]
               .agg(["mean", "median", "count"]).reset_index())
    ax.plot(hourly["upload_hour"], hourly["mean"], marker="o", ms=5,
            color=MAIN_COLOR, label="평균", zorder=3)
    ax.plot(hourly["upload_hour"], hourly["median"], marker="s", ms=5,
            color=SUB_COLOR, ls="--", label="중앙값", zorder=3)
    ax.set_xticks(range(0, 24))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
    ax.set_xlabel("업로드 시간 (시)"); ax.set_ylabel("조회수")
    ax.set_title("업로드 시간대별 조회수")
    ax.legend(fontsize=9)

    # 주기별
    ax = axes[1]
    interval_bins = [0, 1, 3, 7, 14, 30, 60, np.inf]
    interval_labels = ["~1일", "1~3일", "3~7일", "7~14일",
                       "14~30일", "30~60일", "60일+"]
    d["interval_group"] = pd.cut(d["avg_upload_interval_min"] / 1440,
                                  bins=interval_bins, labels=interval_labels)
    sns.boxplot(data=d, x="interval_group", y="view_count",
                hue="interval_group", palette="Set2", legend=False,
                showfliers=False, ax=ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
    ax.set_xlabel("평균 업로드 주기"); ax.set_ylabel("조회수 (Log)")
    ax.set_title("업로드 주기별 조회수 분포")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout(); plt.show()


def eda_appearance_vs_views(df):
    """외모(매력도) 점수 vs 조회수 (구간별 log boxplot + 채널별 산점도)."""
    d = df.dropna(subset=["final_score", "view_count"]).copy()
    d = d[d["view_count"] > 0]
    d["score_group"] = pd.cut(d["final_score"], bins=SCORE_BINS,
                               labels=SCORE_LABELS, right=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    grp_n = d.groupby("score_group", observed=False)["view_count"].count()
    score_labels = [f"{lbl}\n(N={grp_n.get(lbl, 0):,})" for lbl in SCORE_LABELS]
    sns.boxplot(data=d, x="score_group", y="view_count",
                hue="score_group", palette="Set2", legend=False,
                order=SCORE_LABELS, showfliers=False, ax=ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
    ax.set_xticklabels(score_labels, fontsize=9)
    ax.set_title("매력도 점수 구간별 조회수 분포 (Log Scale)")
    ax.set_xlabel("매력도 점수")
    ax.set_ylabel("조회수 (Log Scale)")

    ax = axes[1]
    ch = d.groupby("channel_id").agg(
        final_score=("final_score", "first"),
        avg_view=("view_count", "mean"),
    ).reset_index()
    ch = ch[(ch["avg_view"] > 0) & (ch["final_score"] > 0)]
    ax.scatter(ch["final_score"], ch["avg_view"],
               alpha=0.6, s=50, color=MAIN_COLOR, edgecolors="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
    ax.set_xlabel("매력도 점수")
    ax.set_ylabel("평균 조회수 (Log Scale)")
    ax.set_title("채널별 매력도 점수 vs 평균 조회수")
    r2, sp = add_regression(ax, ch["final_score"], ch["avg_view"], log=True)
    if r2 is not None:
        add_stat_box(ax, f"R² = {r2:.3f}\nSpearman ρ = {sp:.3f}\nN = {len(ch)}")

    plt.tight_layout(); plt.show()
