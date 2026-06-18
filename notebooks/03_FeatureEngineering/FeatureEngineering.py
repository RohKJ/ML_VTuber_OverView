"""
FeatureEngineering.py
========================
파생변수 생성 유틸리티 모듈

사용법:
    import FeatureEngineering as fe

    df = fe.run_feature_engineering(df)
    df = fe.select_columns(df)

    # train/test 분할 후 채널단위 피처 구간화 (누수 방지)
    train, test = fe.bin_channel_features(train, test, q=5)
"""

import numpy as np
import pandas as pd


# JNN 피처 선택 (get_jnn_params에서 사용)
JNN_COLS = [
    "jnn_like_count_momentum",
    "jnn_like_count_cv",
]
_JNN_KNOWN_FUNCS = ["mean", "std", "trend", "momentum", "cv"]

# 채널단위 구간화 대상 컬럼 (bin_channel_features에서 사용)
CHANNEL_BIN_COLS = [
    "like_ratio",              # 채널단위
    "comment_ratio",           # 채널단위
    # "f0_cv",                   # 채널단위
    "speech_density",          # 채널단위
    "final_score",             # 채널단위
    "vtuber_features_score",   # 채널단위
    # "vtuber_vibe_score",       # 채널단위
    # "vtuber_styling_score",    # 채널단위
    "vtuber_technical_score",  # 채널단위
    "vtuber_rendering_score",  # 채널단위
    "vtuber_finish_score",     # 채널단위
    "vtuber_beauty_score",     # 채널단위
    "vtuber_qual_score",       # 채널단위
    "vtuber_appeal_score",     # 채널단위
]


# 파생변수 (Feature Engineering)

def add_video_order(df, channel_col="channel_id", date_col="published_at"):
    """채널별로 영상 게시 순서를 매기는 함수 (1부터 시작)"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["video_order"] = df.sort_values(date_col).groupby(channel_col).cumcount() + 1
    return df


def add_upload_interval(df, channel_col="channel_id", date_col="published_at"):
    """채널별 업로드 주기(분)를 구하는 함수. 첫 영상은 0."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([channel_col, date_col])
    df["upload_interval_min"] = (
        df.groupby(channel_col)[date_col]
        .diff()
        .dt.total_seconds()
        .div(60)
        .fillna(0)
        .astype(int)
    )
    return df


def add_tag_count(df, tag_col="tags"):
    """tags 리스트의 원소 개수를 세는 함수"""
    df = df.copy()
    tags = df[tag_col]
    is_empty = tags.isna() | (tags == "[]")
    df["tag_count"] = tags.str.count("', '").add(1).where(~is_empty, 0).astype(int)
    return df


def add_days_since_published(df, date_col="published_at", collection_date="2026-02-06", drop_same_day=False):
    """published_at 기준으로 데이터 수집 시점 대비 경과 일수를 추가

    Args:
        df: 데이터프레임
        date_col: 업로드 날짜 컬럼명
        collection_date: 데이터 수집일 (기본: "2026-02-06")
        drop_same_day: True이면 수집일 당일 업로드된 영상 제거 (기본: False)
    """
    df = df.copy()

    dt = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.normalize()
    collection_dt = pd.to_datetime(collection_date, utc=True).normalize()

    df["days_since_published"] = (collection_dt - dt).dt.days

    if drop_same_day:
        n_before = len(df)
        df = df[df["days_since_published"] > 0].reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"[add_days_since_published] 수집일 당일 업로드 {n_dropped}행 제거 "
                  f"({n_before} → {len(df)})")

    return df


def add_avg_like_by_channel(df, channel_col="channel_id"):
    """채널별 평균 좋아요수를 각 영상 행에 추가"""
    df = df.copy()
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce")
    df["avg_like_count"] = (
        df.groupby(channel_col)["like_count"]
        .transform("mean")
        .round(0).astype(int)
    )
    return df


def add_avg_comment_by_channel(df, channel_col="channel_id"):
    """채널별 평균 댓글수를 각 영상 행에 추가"""
    df = df.copy()
    df["comment_count"] = pd.to_numeric(df["comment_count"], errors="coerce")
    df["avg_comment_count"] = (
        df.groupby(channel_col)["comment_count"]
        .transform("mean")
        .round(0).astype(int)
    )
    return df


def add_avg_upload_interval(df, channel_col="channel_id", date_col="published_at"):
    """채널별 평균 업로드 주기(분)를 각 영상 행에 추가"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([channel_col, date_col])

    interval = (
        df.groupby(channel_col)[date_col]
        .diff()
        .dt.total_seconds()
        .div(60)
        .fillna(0)
    )
    df["avg_upload_interval_min"] = (
        interval.groupby(df[channel_col])
        .transform("mean")
        .round(0).astype(int)
    )
    return df


# 채널단위 DFS 집계 피처
# featuretools DFS 결과 중 조회수 상관 |corr| ≥ 0.2인 집계 피처만 직접 구현:
#   duration의 MIN/MEAN/STD, contrast MEAN, avg_saturation STD, avg_brightness MEAN

def add_duration_stats_by_channel(df, channel_col="channel_id", src_col="duration"):
    """채널별 영상 길이의 MIN/MEAN/STD (초 단위, 원본 유지)"""
    if src_col not in df.columns:
        return df
    df = df.copy()
    dur = pd.to_numeric(df[src_col], errors="coerce")
    grp = dur.groupby(df[channel_col])
    df["duration_ch_min"]  = grp.transform("min")
    df["duration_ch_mean"] = grp.transform("mean")
    df["duration_ch_std"]  = grp.transform("std").fillna(0)
    return df


def add_avg_contrast_by_channel(df, channel_col="channel_id", src_col="contrast"):
    """채널별 영상 대비(contrast) 평균"""
    if src_col not in df.columns:
        return df
    df = df.copy()
    df["contrast_ch_mean"] = (
        pd.to_numeric(df[src_col], errors="coerce")
        .groupby(df[channel_col]).transform("mean")
    )
    return df


def add_std_saturation_by_channel(df, channel_col="channel_id", src_col="avg_saturation"):
    """채널별 영상 채도(avg_saturation) 표준편차"""
    if src_col not in df.columns:
        return df
    df = df.copy()
    df["saturation_ch_std"] = (
        pd.to_numeric(df[src_col], errors="coerce")
        .groupby(df[channel_col]).transform("std")
        .fillna(0)
    )
    return df


def add_avg_brightness_by_channel(df, channel_col="channel_id", src_col="avg_brightness"):
    """채널별 영상 밝기(avg_brightness) 평균"""
    if src_col not in df.columns:
        return df
    df = df.copy()
    df["brightness_ch_mean"] = (
        pd.to_numeric(df[src_col], errors="coerce")
        .groupby(df[channel_col]).transform("mean")
    )
    return df


# 다중공선성 완화용 파생변수

def add_like_ratio(df):
    """채널 좋아요 참여율: avg_like_count / subscriber_count"""
    df = df.copy()
    subs = pd.to_numeric(df["subscriber_count"], errors="coerce").replace(0, np.nan)
    df["like_ratio"] = pd.to_numeric(df["avg_like_count"], errors="coerce") / subs
    return df


def add_comment_ratio(df):
    """채널 댓글 참여율: avg_comment_count / subscriber_count"""
    df = df.copy()
    subs = pd.to_numeric(df["subscriber_count"], errors="coerce").replace(0, np.nan)
    df["comment_ratio"] = pd.to_numeric(df["avg_comment_count"], errors="coerce") / subs
    return df


def add_f0_cv(df):
    """음높이 변동계수: f0_std / f0_mean (2개→1개 축약)"""
    df = df.copy()
    f0_mean = pd.to_numeric(df["f0_mean"], errors="coerce").replace(0, np.nan)
    df["f0_cv"] = pd.to_numeric(df["f0_std"], errors="coerce") / f0_mean
    return df


def add_speech_density(df):
    """정보 전달 밀도: speech_ratio × wpm (2개→1개 축약)"""
    df = df.copy()
    df["speech_density"] = (
        pd.to_numeric(df["speech_ratio"], errors="coerce")
        * pd.to_numeric(df["wpm"], errors="coerce")
    )
    return df


def add_has_affiliated_group(df, col="affiliated_group"):
    """소속사 유무 이진 변수: unknown/개인 → 0, 나머지 → 1"""
    if col not in df.columns:
        return df
    df = df.copy()
    no_group = {"unknown", "개인"}
    df["has_affiliated_group"] = (~df[col].isin(no_group)).astype(int)
    return df


# 희소 범주 병합 & 이상치 클리핑

def merge_rare_categories(df):
    """희소 범주를 유사 범주로 병합하고, 이상치를 클리핑"""
    df = df.copy()

    # 보조 장르 병합
    if "content_category_secondary_genre" in df.columns:
        mapping = {"토크/소통": "just_chatting", "gaming_mmo": "gaming_rpg"}
        df["content_category_secondary_genre"] = (
            df["content_category_secondary_genre"]
            .replace(mapping)
            .replace(["karaoke", "song_medley"], "cover_song")
        )

    # 핵심 장르 병합
    if "content_category_primary_genre" in df.columns:
        mapping = {"토크/소통": "just_chatting", "gaming_mmo": "gaming_rpg"}
        df["content_category_primary_genre"] = (
            df["content_category_primary_genre"]
            .replace(mapping)
            .replace(["karaoke", "song_medley"], "cover_song")
        )

    # 하이라이트 소스 포맷 병합
    if "highlight_status_source_format" in df.columns:
        df["highlight_status_source_format"] = df["highlight_status_source_format"].replace(
            ["best_moments_compilation", "short_cut"], "hot_clip"
        ).replace("full_original", "highlight_edit")

    # 썸네일 캐릭터 수 클리핑 (5 이상 → 5)
    if "thumb_character_total_character_count" in df.columns:
        df.loc[
            df["thumb_character_total_character_count"] >= 5,
            "thumb_character_total_character_count",
        ] = 5

    return df


# 일괄 실행

def run_feature_engineering(df):
    """파생변수를 일괄 적용하는 헬퍼 함수"""
    df = add_video_order(df)
    df = add_upload_interval(df)
    df = add_tag_count(df)
    df = add_avg_like_by_channel(df)
    df = add_avg_comment_by_channel(df)
    df = add_avg_upload_interval(df)
    df = add_days_since_published(df, drop_same_day=True)
    # featuretools DFS로 채택한 채널단위 집계 피처
    df = add_duration_stats_by_channel(df)
    df = add_avg_contrast_by_channel(df)
    df = add_std_saturation_by_channel(df)
    df = add_avg_brightness_by_channel(df)
    df = add_like_ratio(df)
    df = add_comment_ratio(df)
    df = add_f0_cv(df)
    df = add_speech_density(df)
    df = add_has_affiliated_group(df)
    df = merge_rare_categories(df)
    return df


def map_channel_features(train, test, channel_col="channel_id"):
    """train의 채널단위 집계 피처를 test에 덮어쓰기"""
    test = test.copy()
    cols = [
        "avg_like_count", "avg_comment_count", "avg_upload_interval_min",
        "like_ratio", "comment_ratio",
        # featuretools DFS로 채택한 채널단위 집계 피처
        "duration_ch_min", "duration_ch_mean", "duration_ch_std",
        "contrast_ch_mean", "saturation_ch_std", "brightness_ch_mean",
    ]
    cols = [c for c in cols if c in train.columns]
    # 채널단위 집계 피처는 채널 내 모든 행이 동일값 → first()로 충분
    channel_avg = train.groupby(channel_col)[cols].first()
    for col in cols:
        mapped = test[channel_col].map(channel_avg[col]).fillna(0)
        test[col] = mapped.astype(int) if train[col].dtype.kind == 'i' else mapped
    return test


def map_video_order(train, test, channel_col="channel_id"):
    """train의 채널별 max video_order를 offset으로 test에 이어붙이기"""
    test = test.copy()
    max_order = train.groupby(channel_col)["video_order"].max()
    offset = test[channel_col].map(max_order).fillna(0).astype(int)
    test["video_order"] = test["video_order"] + offset
    return test


def map_jnn_features(train, test, channel_col="channel_id", date_col="published_at",
                     drop_missing=True):
    """train의 채널별 마지막(최신) 영상의 JNN 피처를 test에 덮어쓰기

    drop_missing : True면 train에 없는 채널의 행을 제거
    """
    test = test.copy()
    jnn_cols = [c for c in train.columns if c.startswith("jnn_")]
    if not jnn_cols:
        return test
    last_jnn = (
        train.sort_values(date_col)
        .groupby(channel_col)[jnn_cols]
        .last()
    )
    for col in jnn_cols:
        test[col] = test[channel_col].map(last_jnn[col])

    if drop_missing:
        n_before = len(test)
        test = test.dropna(subset=jnn_cols, how="all").reset_index(drop=True)
        n_dropped = n_before - len(test)
        if n_dropped:
            print(f"[map_jnn_features] train에 없는 채널 {n_dropped}행 제거 "
                  f"({n_before} → {len(test)})")
    return test


def bin_channel_features(train, test, cols=None, q=5):
    """채널단위 수치 피처를 등빈도(q분위) 구간화.

    train 기준으로 분위수 bins 계산 → test에 동일 bins 적용 (누수 방지).
    원본 컬럼을 구간 인덱스(0 ~ q-1)로 덮어씀.

    Args:
        train : 학습 데이터프레임
        test  : 테스트 데이터프레임
        cols  : 구간화할 컬럼 리스트 (None이면 CHANNEL_BIN_COLS 사용)
        q     : 구간 수 (기본 5)

    Returns:
        (train, test) 튜플
    """
    if cols is None:
        cols = [c for c in CHANNEL_BIN_COLS if c in train.columns]

    train, test = train.copy(), test.copy()

    for col in cols:
        series = pd.to_numeric(train[col], errors="coerce")
        _, bins = pd.qcut(series, q=q, retbins=True, duplicates="drop")
        bins[0], bins[-1] = -np.inf, np.inf

        train[col] = pd.cut(series, bins=bins, labels=False)
        test[col]  = pd.cut(
            pd.to_numeric(test[col], errors="coerce"), bins=bins, labels=False
        )

    return train, test


def get_jnn_params():
    """JNN_COLS를 파싱해 (target_cols, agg_func_names) 반환.

    make_jnn_features(df, target_cols=tc, agg_funcs=af) 로 전달하면
    JNN_COLS에 선택된 피처만 계산한다.
    """
    target_cols, func_names = [], []
    for col in JNN_COLS:
        name = col[4:]  # 'jnn_' 제거
        for func in _JNN_KNOWN_FUNCS:
            if name.endswith(f"_{func}"):
                target = name[: -len(f"_{func}")]
                if target not in target_cols:
                    target_cols.append(target)
                if func not in func_names:
                    func_names.append(func)
                break
    return target_cols, func_names


# 컬럼 선택

def select_columns(df):
    """모델링에 필요한 컬럼만 선택해서 반환"""
    cols = [
        # 식별자
        "video_id", "channel_id",

        # 타겟
        "view_count",

        # 1. 채널 참여율 (subscriber 대비 비율로 공선성 완화)
        "like_count",
        # "subscriber_count",  # DFS 단일 최강(|corr|=0.67), 날것 유지 (tree 모델)
        # "like_ratio",    # 채널단위
        # "comment_ratio", # 채널단위

        # 2. 영상 메타 / 채널 행동
        "tag_count",
        # "duration",
        # "days_since_published",
        # "video_order",
        # "upload_interval_min",
        # "avg_upload_interval_min",

        # 3. 음성 (축약: f0_std/mean, speech_ratio×wpm)
        "f0_cv",          # 채널단위
        # "speech_density", # 채널단위

        # 4. 버츄얼 캐릭터 이미지
        # "concept",     # 채널단위
        # "group",       # 채널단위
        # "final_score", # 채널단위

        # 4-1. Gemini 채널 피처 (channel_gemini.csv, 채널단위)
        "worldview_type",          # 세계관 유형 (판타지/현대/없음 등)
        "concept_age_group_est",   # 추정 연령대 (10대/20대/30대이상)
        # "has_affiliated_group",    # 소속사 유무 (0/1)

        # 5. 썸네일
        # "avg_saturation",
        # "text_area_ratio",

        # 5-1. DFS 검증 채택 피처 (채널단위 집계)
        # "duration_ch_min",     # |corr|=0.50 (최강 집계)
        # "duration_ch_mean",    # |corr|=0.45
        # "duration_ch_std",     # |corr|=0.45 (길이 일관성 신호)
        # "contrast_ch_mean",    # |corr|=0.34
        # "saturation_ch_std",   # |corr|=0.23
        # "brightness_ch_mean",  # |corr|=0.21

        # 6. JNN (아래의 JNN_COLS에서 선택)
        # *JNN_COLS,

        # Gemini 컬럼 (P5)

        # 7-1. 상호작용 (Interaction)
        "chat_interaction_frequency",       # enum: high/low/none

        # 7-2. 외형 (Appearance)
        # "suggestive_intent",                       # 선정성 의도
        # "body_exposure_exposure_level",            # 신체 노출 수준
        # "color_vibrancy_color_diversity_level",    # 색상 다양성

        # 7-3. 썸네일 텍스트 (Thumbnail Text)
        "thumbnail_text_provocativeness_level",    # 자극성 수준

        # 7-4. 썸네일 비주얼 (Thumbnail Visual)
        # "thumb_visual_impact_severity_level",      # 시각 자극 강도
        "thumb_character_total_character_count",   # 총 캐릭터 수
        # "thumb_character_has_guest",               # 게스트 유무
        # "thumb_character_sd_presence",             # SD 캐릭터 유무
        # "thumb_chat_staging_chat_extract_present", # 채팅 캡처 유무
        # "thumb_chat_staging_staging_method",       # 연출 방식

        # 7-5. 콘텐츠 (Content)
        "content_category_primary_genre",          # 핵심 장르
        "content_category_secondary_genre",        # 보조 장르
        "collaboration_collaboration_format",      # 협업 형태
        # "collaboration_partners",                # array(JSON) → 제외
        # "real_camera_real_cam_level",              # 실사 카메라 수준
        # "highlight_status_source_format",          # 원본/편집 형식
        # "content_provocativeness_has_profanity",   # 욕설 유무
        # "event_info_is_event_content",             # 이벤트 콘텐츠 여부
        # "animation_fantasy_has_fantastical_elements",  # 판타지 요소 유무

        # 7-6. 의외성 (Surprise)
        "surprise_factor_content_surprise_level",  # 콘텐츠 의외성
        # "surprise_factor_content_surprise_desc", # free text → 제외
        # "surprise_factor_character_consistency",    # 캐릭터 일관성
        # "surprise_factor_reaction_surprise_level",  # 리액션 의외성
        # "surprise_factor_situation_surprise_level", # 상황 의외성
        # "surprise_factor_overall_surprise_score",   # 종합 의외성 점수

        # 8. 버츄얼 이미지 점수 (vtuber_scores.xlsx, 채널단위)
        "vtuber_features_score",           # 디자인 특징
        "vtuber_vibe_score",               # 분위기
        "vtuber_styling_score",            # 스타일링
        # "vtuber_technical_score",          # 기술적 완성도
        # "vtuber_rendering_score",          # 렌더링 품질
        # "vtuber_finish_score",             # 마감 품질
        # "vtuber_beauty_score",             # 미적 점수
        # "vtuber_qual_score",               # 종합 품질
        # "vtuber_appeal_score",             # 어필 점수
    ]

    available = [c for c in cols if c in df.columns]
    return df[available]