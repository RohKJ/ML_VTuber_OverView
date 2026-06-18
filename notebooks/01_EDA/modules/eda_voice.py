"""음성 피처 EDA: 분포, 상관관계, 조회수 비교, 성별/그룹별 비교, 채널 프로파일링."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats

from _utils import (MAIN_COLOR, SUB_COLOR, ACCENT_COLOR, fmt_kr,
                    add_stat_box, add_regression, sig_stars, to_group_type)

# 영문 컬럼명 → 차트에 쓸 한글 라벨
VOICE_FEATURES = {
    "f0_mean":        "평균 피치 (Hz)",
    "f0_std":         "피치 변동성 (Hz)",
    "f0_range":       "피치 범위 (Hz)",
    "speech_ratio":   "발화 비율",
    "wpm":            "말 속도 (WPM)",
    "hnr_mean":       "음성 명료도 (HNR)",
    "voiced_rate":    "유성음 비율",
    "lufs":           "음량 (LUFS)",
    "mean_pause_sec": "평균 쉼 길이 (초)",
}

# 피치·말속도·발화비율·명료도를 구간으로 나눌 때 쓰는 경계값과 라벨
F0_BINS    = [0, 150, 200, 250, 300, 350, 500]
F0_LABELS  = ["~150", "150~200", "200~250", "250~300", "300~350", "350+"]
WPM_BINS   = [0, 60, 90, 120, 150, 200]
WPM_LABELS = ["~60", "60~90", "90~120", "120~150", "150+"]
SR_BINS    = [0, 0.7, 0.8, 0.9, 0.95, 1.01]
SR_LABELS  = ["~70%", "70~80%", "80~90%", "90~95%", "95%+"]
HNR_BINS   = [0, 10, 12, 14, 16, 25]
HNR_LABELS = ["~10", "10~12", "12~14", "14~16", "16+"]


def load_voice_data(df, voice_csv_path, encoding="cp949"):
    """음성 raw CSV를 로드하고 df의 채널별 집계와 병합하여 df_va 반환."""
    df_voice = pd.read_csv(voice_csv_path, encoding=encoding)
    df_voice = df_voice.dropna(subset=["channel_id"])

    ch_agg = df.groupby("channel_id").agg(
        avg_view=("view_count", "mean"),
        median_view=("view_count", "median"),
        total_view=("view_count", "sum"),
        subscriber_count=("subscriber_count", "first"),
        video_count=("video_id", "nunique"),
        avg_like=("like_count", "mean"),
        avg_comment=("comment_count", "mean"),
        sex=("sex", "first"),
        group=("group", "first"),
        channel_title=("channel_title", "first"),
        final_score=("final_score", "first"),
    ).reset_index()

    df_va = df_voice.merge(ch_agg, on="channel_id", how="inner")
    print(f"음성 데이터 병합 완료: {df_va.shape[0]}개 채널")
    return df_va


def eda_voice_overview(df_va):
    """음성 데이터 기초통계 + 주요 피처 분포 히스토그램."""
    feat_cols = [c for c in VOICE_FEATURES if c in df_va.columns]
    print(f"[음성 데이터] shape: {df_va.shape}")

    n = len(feat_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(feat_cols):
        ax = axes[i]
        data = df_va[col].dropna()
        sns.histplot(data, kde=True, bins=25, color=MAIN_COLOR, alpha=0.7, ax=ax)
        med = data.median()
        ax.axvline(med, color=SUB_COLOR, ls="--", lw=2, label=f"중앙값: {med:.1f}")
        ax.set_title(VOICE_FEATURES[col])
        ax.set_xlabel(col)
        ax.legend(fontsize=8)
        add_stat_box(ax, f"N={len(data)}\n평균={data.mean():.1f}\n"
                         f"중앙값={med:.1f}\nStd={data.std():.1f}",
                     loc="upper left" if data.skew() > 0 else "upper right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("음성 피처 분포", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.show()


def eda_voice_correlation(df_va):
    """음성 피처 간 상관관계 히트맵 + 조회수/구독자와의 상관계수 바차트."""
    feat_cols = [c for c in VOICE_FEATURES if c in df_va.columns]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    corr = df_va[feat_cols].corr(method="spearman")
    labels = [VOICE_FEATURES.get(c, c) for c in feat_cols]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title("음성 피처 간 Spearman 상관관계")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    ax = axes[1]
    target_cols = {"avg_view": "평균 조회수", "subscriber_count": "구독자수"}
    corr_data = []
    for feat in feat_cols:
        for target, target_label in target_cols.items():
            valid = df_va[[feat, target]].dropna()
            valid = valid[valid[target] > 0]
            if len(valid) >= 10:
                sp_r, sp_p = stats.spearmanr(valid[feat], valid[target])
                corr_data.append({
                    "피처": VOICE_FEATURES.get(feat, feat),
                    "대상": target_label,
                    "Spearman ρ": sp_r,
                    "p-value": sp_p,
                })

    df_corr = pd.DataFrame(corr_data)
    if len(df_corr) > 0:
        pivot = df_corr.pivot(index="피처", columns="대상", values="Spearman ρ")
        pivot.plot(kind="barh", ax=ax, color=[MAIN_COLOR, SUB_COLOR], alpha=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title("음성 피처 vs 조회수/구독자수 (Spearman ρ)")
        ax.set_xlabel("Spearman ρ")
        ax.legend(fontsize=9)

    plt.tight_layout(); plt.show()


def eda_voice_vs_views(df_va):
    """주요 음성 피처 4개 vs 평균 조회수 (산점도+회귀선 + 구간별 boxplot)."""
    targets = [
        ("f0_mean",      F0_BINS,  F0_LABELS,  "평균 피치 (Hz)"),
        ("wpm",          WPM_BINS, WPM_LABELS, "말 속도 (WPM)"),
        ("speech_ratio", SR_BINS,  SR_LABELS,  "발화 비율"),
        ("hnr_mean",     HNR_BINS, HNR_LABELS, "음성 명료도 (HNR)"),
    ]

    for feat, bins, labels, feat_name in targets:
        d = df_va.dropna(subset=[feat, "avg_view"]).copy()
        d = d[d["avg_view"] > 0]
        d["group_bin"] = pd.cut(d[feat], bins=bins, labels=labels, right=False)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        ax = axes[0]
        ax.scatter(d[feat], d["avg_view"], alpha=0.6, s=50,
                   color=MAIN_COLOR, edgecolors="white", linewidth=0.5)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
        ax.set_xlabel(feat_name)
        ax.set_ylabel("평균 조회수 (Log Scale)")
        ax.set_title(f"{feat_name} vs 평균 조회수")
        r2, sp = add_regression(ax, d[feat], d["avg_view"], log=True)
        if r2 is not None:
            add_stat_box(ax, f"R² = {r2:.3f}\nSpearman ρ = {sp:.3f}\nN = {len(d)}")

        ax = axes[1]
        grp_n = d.groupby("group_bin", observed=False)["avg_view"].count()
        tick_labels = [f"{lbl}\n(N={grp_n.get(lbl, 0):,})" for lbl in labels]
        sns.boxplot(data=d, x="group_bin", y="avg_view",
                    hue="group_bin", palette="Set2", legend=False,
                    order=labels, showfliers=False, ax=ax)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_title(f"{feat_name} 구간별 평균 조회수 분포")
        ax.set_xlabel(feat_name)
        ax.set_ylabel("평균 조회수 (Log Scale)")

        plt.tight_layout(); plt.show()


def eda_voice_by_gender_group(df_va):
    """성별·그룹별 주요 음성 피처 비교 (바이올린 + Mann-Whitney)."""
    feat_list = ["f0_mean", "wpm", "speech_ratio", "hnr_mean", "lufs", "voiced_rate"]

    # 성별 비교
    d_sex = df_va[df_va["sex"].isin(["남자", "여자"])].copy()
    if len(d_sex) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for i, feat in enumerate(feat_list):
            ax = axes[i]
            sns.violinplot(data=d_sex, x="sex", y=feat, hue="sex",
                           palette=[MAIN_COLOR, SUB_COLOR], legend=False,
                           inner="box", ax=ax)
            ax.set_title(VOICE_FEATURES.get(feat, feat))
            ax.set_xlabel("")
            g1 = d_sex[d_sex["sex"] == "남자"][feat].dropna()
            g2 = d_sex[d_sex["sex"] == "여자"][feat].dropna()
            if len(g1) >= 5 and len(g2) >= 5:
                _, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                sig = sig_stars(p_val)
                ax.set_xlabel(f"Mann-Whitney p={p_val:.4f} ({sig})", fontsize=9)
        plt.suptitle("성별 음성 피처 비교", fontsize=16, fontweight="bold", y=1.01)
        plt.tight_layout(); plt.show()

    # 솔로 vs 그룹 비교
    d_grp = df_va.copy()
    d_grp["group_type"] = to_group_type(d_grp["group"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, feat in enumerate(feat_list):
        ax = axes[i]
        sns.violinplot(data=d_grp, x="group_type", y=feat, hue="group_type",
                       palette=[ACCENT_COLOR, SUB_COLOR], legend=False,
                       inner="box", ax=ax)
        ax.set_title(VOICE_FEATURES.get(feat, feat))
        ax.set_xlabel("")
        g1 = d_grp[d_grp["group_type"] == "솔로"][feat].dropna()
        g2 = d_grp[d_grp["group_type"] == "그룹"][feat].dropna()
        if len(g1) >= 5 and len(g2) >= 5:
            _, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            sig = sig_stars(p_val)
            ax.set_xlabel(f"Mann-Whitney p={p_val:.4f} ({sig})", fontsize=9)
    plt.suptitle("솔로 vs 그룹 음성 피처 비교", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.show()



def eda_voice_vs_views_by_gender(df_va):
    """성별 교란 제거: 남/여 각각에서 음성 피처 vs 조회수 상관분석."""
    feat_cols = ["f0_mean", "f0_std", "wpm", "speech_ratio", "hnr_mean",
                 "lufs", "voiced_rate", "mean_pause_sec"]
    feat_names = {c: VOICE_FEATURES.get(c, c) for c in feat_cols}

    results = []
    for sex in ["남자", "여자"]:
        d = df_va[(df_va["sex"] == sex) & (df_va["avg_view"] > 0)].copy()
        for feat in feat_cols:
            valid = d[[feat, "avg_view"]].dropna()
            if len(valid) >= 10:
                sp_r, sp_p = stats.spearmanr(valid[feat], valid["avg_view"])
                results.append({
                    "성별": sex, "피처": feat_names[feat],
                    "N": len(valid), "Spearman ρ": round(sp_r, 3),
                    "p-value": round(sp_p, 4),
                    "유의": sig_stars(sp_p),
                })

    df_res = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df_res.pivot(index="피처", columns="성별", values="Spearman ρ")
    pivot.plot(kind="barh", ax=ax, color=[MAIN_COLOR, SUB_COLOR], alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("성별 통제 후 음성 피처 vs 평균 조회수 (Spearman ρ)")
    ax.set_xlabel("Spearman ρ")
    ax.legend(fontsize=10)
    plt.tight_layout(); plt.show()


def eda_voice_vs_engagement(df_va):
    """음성 피처 vs 좋아요율·댓글율 상관분석 + 유의 피처 산점도."""
    feat_cols = ["f0_mean", "f0_std", "wpm", "speech_ratio", "hnr_mean",
                 "lufs", "voiced_rate", "mean_pause_sec"]
    feat_names = {c: VOICE_FEATURES.get(c, c) for c in feat_cols}

    d = df_va.copy()
    d["like_rate"] = d["avg_like"] / d["avg_view"] * 100
    d["comment_rate"] = d["avg_comment"] / d["avg_view"] * 100
    d = d[(d["avg_view"] > 0) & (d["like_rate"] > 0) & (d["comment_rate"] > 0)]

    targets = {"like_rate": "좋아요율 (%)", "comment_rate": "댓글율 (%)"}
    results = []
    for feat in feat_cols:
        for target, target_name in targets.items():
            valid = d[[feat, target]].dropna()
            if len(valid) >= 10:
                sp_r, sp_p = stats.spearmanr(valid[feat], valid[target])
                results.append({
                    "피처": feat_names[feat], "대상": target_name,
                    "N": len(valid), "Spearman ρ": round(sp_r, 3),
                    "p-value": round(sp_p, 4),
                    "유의": sig_stars(sp_p),
                })

    df_res = pd.DataFrame(results)

    # 바차트
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df_res.pivot(index="피처", columns="대상", values="Spearman ρ")
    pivot.plot(kind="barh", ax=ax, color=[MAIN_COLOR, ACCENT_COLOR], alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("음성 피처 vs 참여율 (Spearman ρ)")
    ax.set_xlabel("Spearman ρ")
    ax.legend(fontsize=10)
    plt.tight_layout(); plt.show()

    # 유의한 피처 산점도
    sig = df_res[df_res["유의"] != "ns"]
    if len(sig) > 0:
        print(f"\n[유의한 상관 {len(sig)}건 — 개별 산점도]")
        for _, row in sig.iterrows():
            feat_kr = row["피처"]
            feat_en = next(k for k, v in feat_names.items() if v == feat_kr)
            target_en = next(k for k, v in targets.items() if v == row["대상"])

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(d[feat_en], d[target_en], alpha=0.6, s=50,
                       color=MAIN_COLOR, edgecolors="white", linewidth=0.5)
            ax.set_xlabel(feat_kr)
            ax.set_ylabel(row["대상"])
            ax.set_title(f"{feat_kr} vs {row['대상']} (ρ={row['Spearman ρ']:.3f} {row['유의']})")

            mask = np.isfinite(d[feat_en]) & np.isfinite(d[target_en])
            x, y = d.loc[mask, feat_en], d.loc[mask, target_en]
            if len(x) >= 10:
                slope, intercept, *_ = stats.linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, slope * x_line + intercept,
                        color=SUB_COLOR, lw=2, ls="--", alpha=0.8)

            add_stat_box(ax, f"Spearman ρ = {row['Spearman ρ']:.3f}\n"
                             f"p = {row['p-value']:.4f}\nN = {row['N']}")
            plt.tight_layout(); plt.show()


def eda_voice_x_appearance(df_va):
    """음성(피치/WPM) × 외모점수 교차 그룹별 조회수 히트맵."""
    d = df_va.dropna(subset=["f0_mean", "final_score", "avg_view"]).copy()
    d = d[d["avg_view"] > 0]

    d["pitch_group"] = pd.cut(d["f0_mean"],
                               bins=[0, 200, 275, 350, 500],
                               labels=["저음(~200)", "중음(200~275)", "고음(275~350)", "초고음(350+)"])
    d["score_group"] = pd.cut(d["final_score"],
                               bins=[0, 40, 55, 70, 100],
                               labels=["하(~40)", "중(40~55)", "상(55~70)", "최상(70+)"])

    cross = d.groupby(["pitch_group", "score_group"], observed=False).agg(
        median_view=("avg_view", "median"), n=("avg_view", "count"),
    ).reset_index()

    pivot_view = cross.pivot(index="pitch_group", columns="score_group", values="median_view")
    pivot_n = cross.pivot(index="pitch_group", columns="score_group", values="n")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    annot = pivot_view.map(lambda x: fmt_kr(x) if pd.notna(x) else "")
    sns.heatmap(pivot_view, annot=annot, fmt="", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("피치 × 외모점수 교차 — 중앙값 조회수")
    ax.set_xlabel("외모 점수"); ax.set_ylabel("피치 구간")

    ax = axes[1]
    sns.heatmap(pivot_n, annot=True, fmt=".0f", cmap="Blues", ax=ax, linewidths=0.5)
    ax.set_title("피치 × 외모점수 교차 — 채널 수 (N)")
    ax.set_xlabel("외모 점수"); ax.set_ylabel("피치 구간")

    plt.tight_layout(); plt.show()

    # WPM × 외모점수
    d2 = df_va.dropna(subset=["wpm", "final_score", "avg_view"]).copy()
    d2 = d2[d2["avg_view"] > 0]
    d2["wpm_group"] = pd.cut(d2["wpm"], bins=[0, 70, 100, 130, 200],
                              labels=["느림(~70)", "보통(70~100)", "빠름(100~130)", "매우빠름(130+)"])
    d2["score_group"] = pd.cut(d2["final_score"], bins=[0, 40, 55, 70, 100],
                                labels=["하(~40)", "중(40~55)", "상(55~70)", "최상(70+)"])

    cross2 = d2.groupby(["wpm_group", "score_group"], observed=False).agg(
        median_view=("avg_view", "median"), n=("avg_view", "count"),
    ).reset_index()

    pivot_view2 = cross2.pivot(index="wpm_group", columns="score_group", values="median_view")
    pivot_n2 = cross2.pivot(index="wpm_group", columns="score_group", values="n")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    annot2 = pivot_view2.map(lambda x: fmt_kr(x) if pd.notna(x) else "")
    sns.heatmap(pivot_view2, annot=annot2, fmt="", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("말속도 × 외모점수 교차 — 중앙값 조회수")
    ax.set_xlabel("외모 점수"); ax.set_ylabel("말속도 구간")

    ax = axes[1]
    sns.heatmap(pivot_n2, annot=True, fmt=".0f", cmap="Blues", ax=ax, linewidths=0.5)
    ax.set_title("말속도 × 외모점수 교차 — 채널 수 (N)")
    ax.set_xlabel("외모 점수"); ax.set_ylabel("말속도 구간")

    plt.tight_layout(); plt.show()


def eda_voice_combo_explore(df_va):
    """음성 피처 조합 vs 단일 피처 비교."""
    dv = df_va.copy()
    dv["like_rate"] = dv["avg_like"] / dv["avg_view"].replace(0, np.nan)
    dv["comment_rate"] = dv["avg_comment"] / dv["avg_view"].replace(0, np.nan)
    print(f"[음성 조합 피처 탐색] N={len(dv)}")

    combos = {
        "상대적 음역대 (f0_range/f0_mean)":   dv["f0_range"] / dv["f0_mean"].replace(0, np.nan),
        "피치 변동계수 (f0_std/f0_mean)":      dv["f0_std"] / dv["f0_mean"].replace(0, np.nan),
        "표현력 (f0_range×f0_std)":            dv["f0_range"] * dv["f0_std"],
        "음성품질×말비율 (hnr×speech_ratio)":  dv["hnr_mean"] * dv["speech_ratio"],
        "실질 말밀도 (wpm×speech_ratio)":      dv["wpm"] * dv["speech_ratio"],
        "속도/쉼 비율 (wpm/mean_pause)":       dv["wpm"] / dv["mean_pause_sec"].replace(0, np.nan),
        "명료도/속도 (hnr/wpm)":               dv["hnr_mean"] / dv["wpm"].replace(0, np.nan),
    }

    singles = {
        "평균 피치": dv["f0_mean"], "피치 변동성": dv["f0_std"],
        "피치 범위": dv["f0_range"], "말 비율": dv["speech_ratio"],
        "말 속도": dv["wpm"], "음질(HNR)": dv["hnr_mean"],
        "유성음 비율": dv["voiced_rate"], "평균 쉼": dv["mean_pause_sec"],
    }

    results = []
    for name, series in {**singles, **combos}.items():
        for target, tlabel in [("avg_view", "평균 조회수"), ("like_rate", "좋아요율"), ("comment_rate", "댓글율")]:
            valid = pd.DataFrame({"feat": series, "target": dv[target]}).dropna()
            valid = valid[np.isfinite(valid["feat"]) & np.isfinite(valid["target"]) & (valid["target"] > 0)]
            if len(valid) >= 10:
                r, p = stats.spearmanr(valid["feat"], valid["target"])
                results.append({
                    "피처": name, "유형": "단일" if name in singles else "조합",
                    "대상": tlabel, "ρ": r, "p": p,
                    "유의": sig_stars(p),
                })

    df_res = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for idx, target in enumerate(["평균 조회수", "좋아요율", "댓글율"]):
        ax = axes[idx]
        sub = df_res[df_res["대상"] == target].sort_values("ρ", ascending=True)
        colors = [MAIN_COLOR if t == "단일" else SUB_COLOR for t in sub["유형"]]
        ax.barh(range(len(sub)), sub["ρ"], color=colors, alpha=0.8)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["피처"], fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Spearman ρ")
        ax.set_title(f"음성 피처 vs {target}")

        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=MAIN_COLOR, label="단일"),
                           Patch(color=SUB_COLOR, label="조합")], fontsize=8)

    plt.suptitle("음성 피처 조합 탐색", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.show()


def eda_voice_gender_channel_size(df_va):
    """성별 채널 규모·조회수 분포 비교 (구독자수, 평균/중앙값 조회수)."""
    d = df_va[df_va["sex"].isin(["남자", "여자"])].copy()
    if len(d) == 0:
        print("성별 데이터 없음"); return

    palette = {"남자": MAIN_COLOR, "여자": SUB_COLOR}
    metrics = [
        ("subscriber_count", "구독자 수"),
        ("avg_view",         "평균 조회수"),
        ("median_view",      "중앙값 조회수"),
    ]

    # ── 행 1: 지표별 boxplot (log scale) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (col, label) in zip(axes, metrics):
        tmp = d[["sex", col]].dropna()
        tmp = tmp[tmp[col] > 0]
        sns.boxplot(data=tmp, x="sex", y=col, hue="sex",
                    palette=palette, legend=False,
                    width=0.45, linewidth=1.5, showfliers=False, ax=ax)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
        ax.set_xlabel("")
        ax.set_ylabel(f"{label} (log scale)")
        ax.set_title(label)

        # 중앙값 레이블
        for i, sex in enumerate(["남자", "여자"]):
            med = tmp[tmp["sex"] == sex][col].median()
            ax.text(i, med * 1.6, fmt_kr(med), ha="center", fontsize=10,
                    fontweight="bold", color=palette[sex])

        # Mann-Whitney
        g1 = tmp[tmp["sex"] == "남자"][col]
        g2 = tmp[tmp["sex"] == "여자"][col]
        if len(g1) >= 5 and len(g2) >= 5:
            _, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            sig = sig_stars(p)
            ax.set_xlabel(f"Mann-Whitney p={p:.4f} ({sig})", fontsize=9)

    plt.suptitle("성별 채널 규모 & 조회수 분포", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.show()

    # ── 행 2: 구독자 vs 평균 조회수 산점도 (성별 색 구분) ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for sex, color in palette.items():
        sub = d[d["sex"] == sex].dropna(subset=["subscriber_count", "avg_view"])
        sub = sub[(sub["subscriber_count"] > 0) & (sub["avg_view"] > 0)]
        ax.scatter(sub["subscriber_count"], sub["avg_view"],
                   color=color, alpha=0.7, s=60, label=sex,
                   edgecolors="white", linewidth=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
    ax.set_xlabel("구독자 수 (log)"); ax.set_ylabel("평균 조회수 (log)")
    ax.set_title("구독자 수 vs 평균 조회수 (성별)")
    ax.legend(fontsize=10)

    ax = axes[1]
    summary = d.groupby("sex")[["subscriber_count", "avg_view", "median_view"]].median()
    x = range(len(summary.columns))
    width = 0.3
    for j, (sex, color) in enumerate(palette.items()):
        vals = summary.loc[sex]
        bars = ax.bar([xi + j * width for xi in x], vals.values,
                      width=width, color=color, alpha=0.8, label=sex)
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                    fmt_kr(val), ha="center", fontsize=9, fontweight="bold", color=color)
    ax.set_xticks([xi + width / 2 for xi in x])
    ax.set_xticklabels(["구독자 수", "평균 조회수", "중앙값 조회수"], fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_kr))
    ax.set_title("성별 중앙값 비교")
    ax.legend(fontsize=10)

    plt.suptitle("성별 채널 규모 상세", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.show()

    # ── 요약 출력 ──
    print("\n[성별 채널 규모 요약]")
    summary_full = d.groupby("sex").agg(
        채널수=("channel_id", "count"),
        구독자수_중앙값=("subscriber_count", "median"),
        평균조회수_중앙값=("avg_view", "median"),
        중앙값조회수_중앙값=("median_view", "median"),
    )
    print(summary_full.to_string())


def run_voice_eda(df_va, sections=None):
    """음성 EDA 전체 또는 선택 실행."""
    all_sections = {
        "overview":       eda_voice_overview,
        "correlation":    eda_voice_correlation,
        "vs_views":       eda_voice_vs_views,
        "gender_group":   eda_voice_by_gender_group,
        "gender_channel": eda_voice_gender_channel_size,
        "gender_control": eda_voice_vs_views_by_gender,
        "engagement":     eda_voice_vs_engagement,
        "appearance":     eda_voice_x_appearance,
        "combo":          eda_voice_combo_explore,
    }
    targets = sections if sections else list(all_sections.keys())
    for name in targets:
        if name in all_sections:
            print(f"\n{'='*60}\n  {name}\n{'='*60}")
            all_sections[name](df_va)
