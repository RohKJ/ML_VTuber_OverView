"""plot_style.py
========================
공통 matplotlib + Plotly 스타일 & 컬러 유틸리티

사용법:
    import plot_style
    plot_style.apply()                                # matplotlib + plotly 일괄 적용
    colors = plot_style.gradient_colors(n, "Oranges") # 그라데이션 색상 n개
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ============================================================
# 공통 팔레트 (matplotlib colorway / plotly colorway 공유)
# ============================================================
COLORWAY = [
    "#DD8452",  # 주황  (버츄얼 캐릭터)
    "#4C72B0",  # 파랑  (썸네일)
    "#55A868",  # 초록  (Gemini)
    "#C44E52",  # 빨강
    "#8172B2",  # 보라
    "#937860",  # 브라운
    "#DA8BC3",  # 핑크
    "#8C8C8C",  # 회색
    "#CCB974",  # 머스타드
    "#64B5CD",  # 청록
]

# ============================================================
# 그룹별 colormap (feature importance 시각화 등)
# ============================================================
GROUP_CMAPS = {
    "버츄얼 캐릭터":    "Oranges",
    "썸네일":           "Blues",
    "Gemini 영상 분석": "Greens",
}


def apply():
    """matplotlib + Plotly 공통 스타일 일괄 적용 — 노트북 시작 시 한 번만 호출."""
    _apply_matplotlib()
    _apply_plotly()


def _apply_matplotlib():
    """matplotlib rcParams 설정."""
    mpl.rcParams.update({
        # 폰트 / 유니코드
        "font.family": "Malgun Gothic",
        "axes.unicode_minus": False,

        # 그리드 (막대 뒤로)
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#cccccc",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,

        # 스파인
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,

        # 타이틀 / 축 라벨 (진하게)
        "axes.titlesize":   13,
        "axes.titleweight": "bold",
        "axes.titlepad":    12,
        "axes.labelsize":   11,
        "axes.labelweight": "bold",

        # 틱
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": "#222222",
        "ytick.color": "#222222",

        # 배경 / 출력
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "figure.dpi": 100,
        "savefig.dpi": 120,

        # 범례
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


def _apply_plotly():
    """Plotly 전역 템플릿 설정 — Optuna / 기타 plotly 그래프 일괄 적용."""
    try:
        import plotly.io as pio
        import plotly.graph_objects as go
    except ImportError:
        return

    # plotly_white 베이스로 커스텀 템플릿 생성
    t = go.layout.Template(pio.templates["plotly_white"])

    # 폰트 (matplotlib 스타일과 동일 계열)
    t.layout.font.family = "Malgun Gothic, Nanum Gothic, sans-serif"
    t.layout.font.size = 12
    t.layout.font.color = "#222222"

    # 타이틀 (진한 색 + 좌측 정렬)
    t.layout.title.font.size = 15
    t.layout.title.font.color = "#111111"
    t.layout.title.x = 0.03
    t.layout.title.xanchor = "left"

    # 배경
    t.layout.paper_bgcolor = "white"
    t.layout.plot_bgcolor = "white"

    # 축: 그리드 흐리게, 스파인 얇게
    for axis in (t.layout.xaxis, t.layout.yaxis):
        axis.gridcolor = "#e5e5e5"
        axis.gridwidth = 1
        axis.zerolinecolor = "#cccccc"
        axis.linecolor = "#333333"
        axis.ticks = "outside"
        axis.tickcolor = "#888888"
        axis.tickfont.color = "#333333"
        axis.title.font.size = 12

    # 색상 순서 (matplotlib colorway와 동일)
    t.layout.colorway = COLORWAY

    # 여백
    t.layout.margin = dict(l=70, r=30, t=60, b=55)

    # 범례 (틀 없이)
    t.layout.legend.bgcolor = "rgba(255,255,255,0.8)"
    t.layout.legend.bordercolor = "rgba(0,0,0,0)"

    pio.templates["compact_theme"] = t
    pio.templates.default = "compact_theme"


def gradient_colors(n, cmap_name, start=0.35, end=0.95):
    """colormap에서 균일 간격으로 n개 색상 추출.

    작은 값(start) → 옅음, 큰 값(end) → 진함.
    ascending 정렬된 가로막대에 쓰면 위로 갈수록(값 큼) 짙은 색.
    """
    cmap = plt.get_cmap(cmap_name)
    return [cmap(v) for v in np.linspace(start, end, n)]
