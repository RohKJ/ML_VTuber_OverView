"""공통 유틸리티: 색상 상수, 포맷 함수, 시각화 헬퍼."""
import numpy as np
from scipy import stats
from scipy.stats import spearmanr

# 차트 전반에 공통으로 쓰는 색상 (주/보조/강조)
MAIN_COLOR   = "#4C72B0"
SUB_COLOR    = "#DD8452"
ACCENT_COLOR = "#55A868"


def fmt_kr(x, pos=None):
    """축 라벨용: 숫자를 한국식(만/억) 표기로 변환."""
    if abs(x) >= 1e8:
        return f'{x/1e8:.1f}억'
    elif abs(x) >= 1e4:
        return f'{x/1e4:.0f}만'
    elif abs(x) >= 1:
        return f'{x:,.0f}'
    return f'{x:.1f}'


def add_stat_box(ax, text, loc='upper right'):
    """차트에 통계 요약 텍스트 박스 추가."""
    box = dict(boxstyle='round,pad=0.4', facecolor='white',
               edgecolor='gray', alpha=0.85)
    ha = 'right' if 'right' in loc else 'left'
    x = 0.97 if 'right' in loc else 0.03
    ax.text(x, 0.95, text, transform=ax.transAxes,
            fontsize=9.5, verticalalignment='top', horizontalalignment=ha,
            bbox=box, linespacing=1.5)


def add_regression(ax, x, y, color=None, log=False):
    """산점도 위에 회귀선을 그리고 R²와 Spearman ρ를 반환한다."""
    if color is None:
        color = SUB_COLOR
    # 양수이면서 유한한 값만 사용 (log 변환과 회귀가 깨지지 않도록)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    xc, yc = x[mask].values, y[mask].values
    if len(xc) < 10:
        return None, None

    # log 옵션이면 log10 공간에서 회귀하고, 그릴 때만 원래 스케일로 되돌린다
    xf = np.log10(xc) if log else xc
    yf = np.log10(yc) if log else yc

    slope, intercept, r, p, se = stats.linregress(xf, yf)
    x_line = np.linspace(xf.min(), xf.max(), 100)
    y_line = slope * x_line + intercept

    if log:
        ax.plot(10**x_line, 10**y_line, color=color, lw=2, ls='--',
                alpha=0.8, zorder=5)
    else:
        ax.plot(x_line, y_line, color=color, lw=2, ls='--',
                alpha=0.8, zorder=5)

    sp_r, _ = spearmanr(xc, yc)
    return r**2, sp_r


def sig_stars(p):
    """p값을 유의성 별표로 변환 (0.001/0.01/0.05 기준, 아니면 ns)."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def to_group_type(group):
    """그룹명을 솔로/그룹 두 종류로 단순화. '솔로'만 남기고 나머지는 모두 '그룹'으로 묶는다."""
    return np.where(group == "솔로", "솔로", "그룹")

