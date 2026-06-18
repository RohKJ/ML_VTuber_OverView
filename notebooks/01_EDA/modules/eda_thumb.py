"""썸네일 피처 EDA: 분포, 상관관계, 조회수 비교, 얼굴 분석, 성별/그룹별 비교."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats

from _utils import MAIN_COLOR, SUB_COLOR, fmt_kr, add_stat_box, add_regression, sig_stars

# 영문 컬럼명 → 차트에 쓸 한글 라벨
THUMB_FEATURES = {
    "avg_brightness":  "평균 밝기",
    "avg_saturation":  "평균 채도",
    "contrast":        "명암 대비",
    "text_area_ratio": "텍스트 면적비율",
    "color_variance":  "색 분산도",
}

# 각 썸네일 피처를 구간으로 나눌 때 쓰는 경계값과 라벨
BRIGHT_BINS   = [0, 0.25, 0.35, 0.45, 0.55, 0.65, 1.01]
BRIGHT_LABELS = ["~0.25", "0.25~0.35", "0.35~0.45", "0.45~0.55", "0.55~0.65", "0.65+"]
SAT_BINS      = [0, 0.1, 0.2, 0.3, 0.4, 1.01]
SAT_LABELS    = ["~0.1", "0.1~0.2", "0.2~0.3", "0.3~0.4", "0.4+"]
TEXT_BINS      = [0, 0.02, 0.05, 0.1, 0.2, 1.01]
TEXT_LABELS    = ["~2%", "2~5%", "5~10%", "10~20%", "20%+"]
CVAR_BINS      = [0, 2.5, 3.5, 4.0, 4.5, 7.0]
CVAR_LABELS    = ["~2.5", "2.5~3.5", "3.5~4.0", "4.0~4.5", "4.5+"]


def eda_thumb_overview(df):
    """썸네일 피처 기초통계 + 히스토그램."""
    feat_cols = [c for c in THUMB_FEATURES if c in df.columns]
    print(f"[썸네일 피처 개요] N={len(df):,}")

    ncols = 3
    nrows = (len(feat_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(feat_cols):
        ax = axes[i]
        data = df[col].dropna()
        sns.histplot(data, kde=True, bins=40, color=MAIN_COLOR, alpha=0.7, ax=ax)
        med = data.median()
        ax.axvline(med, color=SUB_COLOR, ls="--", lw=2, label=f"중앙값: {med:.3f}")
        ax.set_title(THUMB_FEATURES[col])
        ax.set_xlabel(col)
        ax.legend(fontsize=8)
        add_stat_box(ax, f"N={len(data):,}\n평균={data.mean():.3f}\n"
                         f"중앙값={med:.3f}\nStd={data.std():.3f}",
                     loc="upper left" if data.skew() > 0 else "upper right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("썸네일 피처 분포", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.show()


def eda_thumb_correlation(df):
    """썸네일 피처 간 + vs 조회수/좋아요 Spearman 상관관계."""
    feat_cols = [c for c in THUMB_FEATURES if c in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    corr = df[feat_cols].corr(method="spearman")
    labels = [THUMB_FEATURES.get(c, c) for c in feat_cols]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title("썸네일 피처 간 Spearman 상관관계")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    ax = axes[1]
    targets = {"view_count": "조회수", "like_count": "좋아요"}
    corr_data = []
    for feat in feat_cols:
        for target, label in targets.items():
            valid = df[[feat, target]].dropna()
            valid = valid[valid[target] > 0]
            if len(valid) >= 30:
                sp_r, sp_p = stats.spearmanr(valid[feat], valid[target])
                corr_data.append({
                    "피처": THUMB_FEATURES.get(feat, feat),
                    "대상": label, "Spearman ρ": sp_r, "p-value": sp_p,
                })

    df_corr = pd.DataFrame(corr_data)
    if len(df_corr) > 0:
        pivot = df_corr.pivot(index="피처", columns="대상", values="Spearman ρ")
        pivot.plot(kind="barh", ax=ax, color=[MAIN_COLOR, SUB_COLOR], alpha=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title("썸네일 피처 vs 조회수/좋아요 (Spearman ρ)")
        ax.set_xlabel("Spearman ρ")
        ax.legend(fontsize=9)

    plt.tight_layout(); plt.show()



def eda_thumb_vs_views(df):
    """주요 썸네일 피처 4개 vs 조회수 (산점도+회귀선 + 구간별 boxplot)."""
    targets = [
        ("avg_brightness",  BRIGHT_BINS, BRIGHT_LABELS, "평균 밝기"),
        ("avg_saturation",  SAT_BINS,    SAT_LABELS,    "평균 채도"),
        ("text_area_ratio", TEXT_BINS,    TEXT_LABELS,   "텍스트 면적비율"),
        ("color_variance",  CVAR_BINS,   CVAR_LABELS,   "색 분산도"),
    ]

    for feat, bins, labels, feat_name in targets:
        d = df.dropna(subset=[feat, "view_count"]).copy()
        d = d[d["view_count"] > 0]
        d["group_bin"] = pd.cut(d[feat], bins=bins, labels=labels, right=False)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        ax = axes[0]
        ax.scatter(d[feat], d["view_count"], alpha=0.15, s=10,
                   color=MAIN_COLOR, edgecolors="none")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
        ax.set_xlabel(feat_name)
        ax.set_ylabel("조회수 (Log Scale)")
        ax.set_title(f"{feat_name} vs 조회수")
        r2, sp = add_regression(ax, d[feat], d["view_count"], log=True)
        if r2 is not None:
            add_stat_box(ax, f"R² = {r2:.4f}\nSpearman ρ = {sp:.4f}\nN = {len(d):,}")

        ax = axes[1]
        grp_n = d.groupby("group_bin", observed=False)["view_count"].count()
        tick_labels = [f"{lbl}\n(N={grp_n.get(lbl, 0):,})" for lbl in labels]
        sns.boxplot(data=d, x="group_bin", y="view_count",
                    hue="group_bin", palette="Set2", legend=False,
                    order=labels, showfliers=False, ax=ax)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_xlabel(feat_name)
        ax.set_ylabel("조회수 (Log Scale)")
        ax.set_title(f"{feat_name} 구간별 조회수 분포")

        plt.tight_layout(); plt.show()


def eda_thumb_channel_level(df):
    """채널별 평균 썸네일 피처 vs 평균 조회수."""
    feat_cols = ["avg_brightness", "avg_saturation", "contrast",
                 "text_area_ratio", "color_variance"]

    agg_dict = {feat: "mean" for feat in feat_cols}
    agg_dict.update({"view_count": "mean", "like_count": "mean",
                     "video_id": "count", "channel_title": "first"})

    df_ch = df.groupby("channel_id").agg(agg_dict).reset_index()
    df_ch = df_ch.rename(columns={"video_id": "n_videos"})
    df_ch = df_ch[df_ch["view_count"] > 0]

    print(f"[채널 단위 집계] 채널 수: {len(df_ch)}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, feat in enumerate(feat_cols):
        ax = axes[i]
        ax.scatter(df_ch[feat], df_ch["view_count"], alpha=0.6, s=40,
                   color=MAIN_COLOR, edgecolors="white", linewidth=0.5)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
        ax.set_xlabel(THUMB_FEATURES.get(feat, feat))
        ax.set_ylabel("채널 평균 조회수")
        r2, sp = add_regression(ax, df_ch[feat], df_ch["view_count"], log=True)
        if r2 is not None:
            add_stat_box(ax, f"R² = {r2:.3f}\nSpearman ρ = {sp:.3f}\nN = {len(df_ch)}")
        ax.set_title(THUMB_FEATURES.get(feat, feat))

    plt.suptitle("채널 단위: 평균 썸네일 피처 vs 평균 조회수", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.show()


def eda_thumb_combo_explore(df):
    """썸네일 피처 조합 vs 단일 피처 비교 (Shorts 제외)."""
    dn = df[df["is_shorts"] == False].copy()
    print(f"[썸네일 조합 피처 탐색] 일반 영상 N={len(dn):,}")

    combos = {
        "밝기×채도 (선명도)":       dn["avg_brightness"] * dn["avg_saturation"],
        "대비×채도 (시각 임팩트)":   dn["contrast"] * dn["avg_saturation"],
        "대비×밝기":                dn["contrast"] * dn["avg_brightness"],
        "텍스트×대비 (가독성)":     dn["text_area_ratio"] * dn["contrast"],
        "색분산×대비 (시각 복잡도)": dn["color_variance"] * dn["contrast"],
        "(1-텍스트)×밝기 (깔끔도)": (1 - dn["text_area_ratio"]) * dn["avg_brightness"],
        "채도×색분산":              dn["avg_saturation"] * dn["color_variance"],
    }

    singles = {
        "평균 밝기": dn["avg_brightness"], "평균 채도": dn["avg_saturation"],
        "명암 대비": dn["contrast"], "텍스트 비율": dn["text_area_ratio"],
        "색 분산도": dn["color_variance"],
    }

    results = []
    for name, series in {**singles, **combos}.items():
        for target, tlabel in [("view_count", "조회수"), ("like_count", "좋아요")]:
            valid = pd.DataFrame({"feat": series, "target": dn[target]}).dropna()
            valid = valid[valid["target"] > 0]
            if len(valid) >= 30:
                r, p = stats.spearmanr(valid["feat"], valid["target"])
                results.append({
                    "피처": name, "유형": "단일" if name in singles else "조합",
                    "대상": tlabel, "ρ": r, "p": p,
                    "유의": sig_stars(p),
                })

    df_res = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for idx, target in enumerate(["조회수", "좋아요"]):
        ax = axes[idx]
        sub = df_res[df_res["대상"] == target].sort_values("ρ", ascending=True)
        colors = [MAIN_COLOR if t == "단일" else SUB_COLOR for t in sub["유형"]]
        ax.barh(range(len(sub)), sub["ρ"], color=colors, alpha=0.8)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["피처"], fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Spearman ρ")
        ax.set_title(f"썸네일 피처 vs {target} (일반 영상)")

        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=MAIN_COLOR, label="단일"),
                           Patch(color=SUB_COLOR, label="조합")], fontsize=8)

    plt.suptitle("썸네일 피처 조합 탐색", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.show()



def run_thumb_eda(df, sections=None):
    """썸네일 EDA 전체 또는 선택 실행."""
    all_sections = {
        "overview":      eda_thumb_overview,
        "correlation":   eda_thumb_correlation,
        "vs_views":      eda_thumb_vs_views,
        "channel":       eda_thumb_channel_level,
        "combo":         eda_thumb_combo_explore,
    }
    targets = sections if sections else list(all_sections.keys())
    for name in targets:
        if name in all_sections:
            print(f"\n{'='*60}\n  {name}\n{'='*60}")
            all_sections[name](df)
