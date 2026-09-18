"""
AI CUP 2026 玉山挑戰賽 — 150檔投資池市場環境探索分析
純描述性分析，不含選股/策略邏輯。產出 12 張 seaborn 圖到 figures/，
並把每張圖對應的關鍵數字印到 stdout，供撰寫 summary.md 時引用真實數字。
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ---------------------------------------------------------------- setup ----
sns.set_theme(style="whitegrid")
# sns.set_theme() 若帶 font= 會直接鎖死 font.family 成單一字型、不會走 fallback；
# 改成設定 font.sans-serif 清單。WenQuanYi Zen Hei 是原分析機器(Linux)上的字型，
# 補上 Heiti TC(macOS內建)當備援，避免沒有該字型的環境把中文字元顯示成方框。
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Heiti TC", "PingFang TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 路徑相對於repo根目錄，執行時請從repo根目錄下指令：python3 src/eda_market_profile.py
DATA_PATH = "data/price_volume.csv"
FIG_DIR = "analysis/figures"

UNIVERSE = """2330 2454 2308 2317 3711 2881 2383 2303 2882 3037 2891 1303 2345 2382 2408 7769 2412 2327 6669 3017
2885 2887 2360 2886 2059 6505 2884 2880 2357 2890 2883 3443 3653 2395 2301 6446 4958 2603 5880 1216
3045 2368 3665 4904 3481 2379 1301 1326 3189 3034 2207 2002 2801 2449 1590 3661 6770 3036 2615 2618
8046 2344 3231 2892 3008 2356 2404 6515 3533 2313 2337 5871 3044 3702 1101 2409 6239 2609 2834 6139
2912 4938 2376 5876 1519 6415 6919 2324 2347 1504 7750 1402 1605 2610 6789 1802 8210 2451 2812 2377
5274 6223 6488 8299 6274 5347 3293 8069 3529 3081 3260 3105 5289 5536 5483 6147 7734 6187 3264 3324
6510 8358 3374 3131 4749 3491 6121 6548 1785 3363 7828 4979 6182 3163 3211 4966 4991 5903 1815 3680
8415 6290 7751 6023 4772 8932 6584 3227 4123 3718""".split()
assert len(UNIVERSE) == 150

O, H, L, C, V = "開盤價(元)", "最高價(元)", "最低價(元)", "收盤價(元)", "成交量(千股)"
TICKER, NAME, DATE, SECTOR = "公司簡稱", "名稱", "年月日", "TSE產業_名稱"

RECENT_N = 252          # 近一年交易日數（橫斷面統計共同窗口）
MIN_OBS_RECENT = 60     # 近期窗口資料不足的門檻
WINDOW_START = (10, 26)  # 比賽視窗（月, 日）
WINDOW_END = (11, 27)


def sec(msg):
    print(f"\n{'='*10} {msg} {'='*10}")


# ------------------------------------------------------------ load/prep ----
def load_universe():
    sec("Loading & filtering to 150-ticker universe")
    df = pd.read_csv(DATA_PATH, dtype={TICKER: "string", NAME: "string", SECTOR: "string"})
    df[DATE] = pd.to_datetime(df[DATE].astype(str), format="%Y%m%d")
    uni = df[df[TICKER].isin(UNIVERSE)].copy()
    del df
    uni = uni.sort_values([TICKER, DATE]).reset_index(drop=True)
    uni["ret"] = uni.groupby(TICKER)[C].pct_change()
    uni["turnover"] = uni[C] * uni[V] * 1000.0  # V是千股 -> 股數 x 收盤價 = 成交金額(元)
    print(f"universe rows: {len(uni):,}  date range: {uni[DATE].min().date()} ~ {uni[DATE].max().date()}")
    return uni


def sector_map(uni):
    return uni.sort_values(DATE).groupby(TICKER)[SECTOR].last()


def name_map(uni):
    return uni.groupby(TICKER)[NAME].first()


def ascii_log_formatter():
    """LogFormatterSciNotation 在 mathtext 關閉情境下仍會插入 U+2212 負號，
    中文字型沒有該字符會顯示成方框；改用純 ASCII 字串的自訂 formatter 避開問題。"""
    def _fmt(x, pos):
        if x <= 0:
            return ""
        exp = int(round(np.log10(x)))
        return f"$10^{{{exp}}}$" if exp != 0 else "1"
    return matplotlib.ticker.FuncFormatter(_fmt)


def savefig(fig, filename):
    path = f"{FIG_DIR}/{filename}"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved -> {path}")


# ============================================================= A. 投資池輪廓 ====
def plot_01_sector_composition(uni):
    sec("01 產業分布")
    smap = sector_map(uni)
    counts = smap.value_counts().sort_values(ascending=False)
    print(counts.to_string())

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, legend=False,
                palette="crest_r", ax=ax)
    ax.set_xlabel("檔數")
    ax.set_ylabel("TSE產業")
    ax.set_title("150檔比賽池 - 產業分布")
    for i, v in enumerate(counts.values):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=9)
    savefig(fig, "01_sector_composition.png")


def plot_02_history_length(uni):
    sec("02 上市歷史/資料長度分布")
    g = uni.groupby(TICKER)[DATE].agg(["min", "max", "count"])
    last_date = uni[DATE].max()
    g["years"] = (last_date - g["min"]).dt.days / 365.25
    short = g[g["years"] < 3].sort_values("years")
    nm = name_map(uni)
    print(f"資料<3年的股票數: {len(short)}")
    for tk, row in short.iterrows():
        print(f"  {tk} {nm.get(tk,'')}: {row['years']:.2f}年 (首筆 {row['min'].date()}, {int(row['count'])}筆)")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.histplot(g["years"], bins=26, color="#3b6ea5", ax=ax)
    ax.axvline(3, color="crimson", ls="--", lw=1.5, label="3年門檻")
    ax.set_xlabel("資料年數（至最新一筆資料）")
    ax.set_ylabel("檔數")
    ax.set_title(f"150檔資料歷史長度分布（資料<3年: {len(short)}檔）")
    ax.legend()
    savefig(fig, "02_history_length.png")


def recent_window_dates(uni):
    all_dates = sorted(uni[DATE].unique())
    return set(all_dates[-RECENT_N:])


def plot_03_liquidity(uni, rdates):
    sec("03 流動性分布（近252交易日日均成交金額）")
    recent = uni[uni[DATE].isin(rdates)]
    n_obs = recent.groupby(TICKER).size()
    insufficient = n_obs[n_obs < MIN_OBS_RECENT].index.tolist()
    avg_turnover = recent.groupby(TICKER)["turnover"].mean()
    smap = sector_map(uni)
    nm = name_map(uni)

    d = pd.DataFrame({"turnover": avg_turnover, "sector": smap, "n_obs": n_obs}).dropna()
    d = d[~d.index.isin(insufficient)]
    d["log_turnover_億"] = np.log10(d["turnover"] / 1e8)

    print(f"資料不足({MIN_OBS_RECENT}筆)被排除: {insufficient} -> {[nm.get(t,'') for t in insufficient]}")
    print("日均成交金額(億元) 描述統計:")
    print((d["turnover"] / 1e8).describe())
    lowest = (d["turnover"] / 1e8).sort_values().head(10)
    print("\n流動性最低10檔（億元/日）:")
    for tk, v in lowest.items():
        print(f"  {tk} {nm.get(tk,'')}: {v:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 1.3]})
    sns.histplot(d["turnover"] / 1e8, bins=30, log_scale=True, color="#3b6ea5", ax=axes[0])
    axes[0].xaxis.set_major_formatter(ascii_log_formatter())
    axes[0].set_xlabel("日均成交金額（億元，log scale）")
    axes[0].set_ylabel("檔數")
    axes[0].set_title("近1年日均成交金額分布")

    order = d.groupby("sector")["turnover"].median().sort_values(ascending=False).index
    sns.boxplot(data=d, x="turnover", y="sector", order=order, hue="sector", legend=False,
                palette="crest_r", ax=axes[1])
    axes[1].set_xscale("log")
    axes[1].xaxis.set_major_formatter(ascii_log_formatter())
    axes[1].set_xlabel("日均成交金額（元，log scale）")
    axes[1].set_ylabel("")
    axes[1].set_title("依產業分組的流動性分布")
    savefig(fig, "03_liquidity_distribution.png")
    return insufficient


def plot_04_volatility(uni, rdates, insufficient):
    sec("04 波動度分布（近252交易日年化）")
    recent = uni[uni[DATE].isin(rdates)]
    vol = recent.groupby(TICKER)["ret"].std() * np.sqrt(252) * 100
    smap = sector_map(uni)
    nm = name_map(uni)
    d = pd.DataFrame({"vol": vol, "sector": smap}).dropna()
    d = d[~d.index.isin(insufficient)]

    print("近1年年化波動度(%) 描述統計:")
    print(d["vol"].describe())
    highest = d["vol"].sort_values(ascending=False).head(10)
    print("\n波動度最高10檔:")
    for tk, v in highest.items():
        print(f"  {tk} {nm.get(tk,'')}: {v:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 8))
    order = d.groupby("sector")["vol"].median().sort_values(ascending=False).index
    sns.boxplot(data=d, x="vol", y="sector", order=order, hue="sector", legend=False,
                palette="flare", ax=ax)
    ax.set_xlabel("年化波動度 (%)")
    ax.set_ylabel("")
    ax.set_title("近1年各產業波動度分布")
    savefig(fig, "04_volatility_distribution.png")


# ============================================================= B. 近期市場狀態 ====
def build_equal_weight_index(uni):
    pivot = uni.pivot_table(index=DATE, columns=TICKER, values="ret")
    index_ret = pivot.mean(axis=1, skipna=True)
    return pivot, index_ret


def plot_05_rolling_vol(index_ret):
    sec("05 代理指數滾動波動度（近3年）")
    rolling_vol = index_ret.rolling(60).std() * np.sqrt(252) * 100
    last3y = rolling_vol.tail(252 * 3)
    current = last3y.iloc[-1]
    pct_rank = (last3y < current).mean() * 100
    print(f"目前(最新交易日)60日滾動年化波動度: {current:.2f}%")
    print(f"在近3年區間中的百分位: {pct_rank:.1f}%")
    print(last3y.describe())

    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.lineplot(x=last3y.index, y=last3y.values, color="#3b6ea5", ax=ax)
    ax.axhline(current, color="crimson", ls="--", lw=1,
               label=f"最新值 {current:.1f}% (歷史百分位 {pct_rank:.0f}%)")
    ax.set_xlabel("日期")
    ax.set_ylabel("60日滾動年化波動度 (%)")
    ax.set_title("150檔等權重代理指數 - 近3年滾動波動度")
    ax.legend()
    savefig(fig, "05_rolling_volatility.png")


def plot_06_momentum_snapshot(uni):
    sec("06 動能橫斷面快照（1M/3M/6M）")
    pivot_c = uni.pivot_table(index=DATE, columns=TICKER, values=C)
    last_date = pivot_c.index.max()
    rows = []
    for label, n in [("1個月(21日)", 21), ("3個月(63日)", 63), ("6個月(126日)", 126)]:
        if len(pivot_c) <= n:
            continue
        ret = pivot_c.iloc[-1] / pivot_c.iloc[-1 - n] - 1
        for tk, v in ret.dropna().items():
            rows.append({"window": label, "ticker": tk, "ret": v * 100})
    d = pd.DataFrame(rows)
    print(f"基準日: {last_date.date()}")
    print(d.groupby("window")["ret"].describe())

    fig, ax = plt.subplots(figsize=(9, 6))
    order = ["1個月(21日)", "3個月(63日)", "6個月(126日)"]
    sns.violinplot(data=d, x="window", y="ret", order=order, hue="window", legend=False,
                    palette="crest", inner="quartile", ax=ax)
    ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel("回顧窗口")
    ax.set_ylabel("報酬率 (%)")
    ax.set_title(f"150檔動能橫斷面分布（基準日 {last_date.date()}）")
    savefig(fig, "06_momentum_snapshot.png")


def plot_07_correlation(uni, rdates):
    sec("07 報酬相關性結構（近252交易日）")
    recent = uni[uni[DATE].isin(rdates)]
    pivot = recent.pivot_table(index=DATE, columns=TICKER, values="ret")
    valid_cols = pivot.columns[pivot.notna().sum() >= MIN_OBS_RECENT]
    dropped = sorted(set(pivot.columns) - set(valid_cols))
    nm = name_map(uni)
    print(f"因資料不足排除於相關性分析: {[(t, nm.get(t,'')) for t in dropped]}")
    corr = pivot[valid_cols].corr()
    corr.index.name = None
    corr.columns.name = None
    off_diag = corr.values[np.triu_indices_from(corr.values, k=1)]
    print(f"相關係數矩陣: {corr.shape}, 平均兩兩相關係數: {off_diag.mean():.3f}, "
          f"中位數: {np.median(off_diag):.3f}")

    g = sns.clustermap(corr, cmap="vlag", center=0, figsize=(14, 14),
                        xticklabels=False, yticklabels=False,
                        cbar_pos=(0.02, 0.83, 0.03, 0.15))
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.fig.suptitle("150檔近1年日報酬相關係數（階層聚類）", y=1.02)
    g.savefig(f"{FIG_DIR}/07_correlation_clustermap.png", dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    print(f"saved -> {FIG_DIR}/07_correlation_clustermap.png")


# ============================================================= C. 比賽視窗季節性 ====
def in_contest_window(dates):
    md = pd.Series([(d.month, d.day) for d in dates])
    return md.apply(lambda x: WINDOW_START <= x <= WINDOW_END).values


def plot_08_window_vs_rest(index_ret):
    sec("08 比賽視窗 vs 全年其餘 報酬分布")
    idx = index_ret.dropna()
    in_win = in_contest_window(idx.index)
    pre2015 = idx.index.year < 2015

    groups = pd.Series(np.where(~in_win, "全年(非視窗)",
                        np.where(pre2015, "視窗期(±7%制度, 2000-2014)", "視窗期(±10%制度, 2015-)")),
                        index=idx.index)
    d = pd.DataFrame({"ret": idx.values * 100, "group": groups.values})
    print(d.groupby("group")["ret"].agg(["count", "mean", "std", "median"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    order = ["全年(非視窗)", "視窗期(±7%制度, 2000-2014)", "視窗期(±10%制度, 2015-)"]
    sns.boxplot(data=d, x="group", y="ret", order=order, hue="group", legend=False,
                palette=["#8fa8c7", "#e0a458", "#c0435b"], showfliers=False, ax=ax)
    ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel("")
    ax.set_ylabel("代理指數日報酬率 (%)")
    ax.set_title("比賽視窗(10/26-11/27) vs 全年其餘 - 日報酬分布")
    savefig(fig, "08_window_vs_rest_return.png")


def contest_phase(dates):
    md = [(d.month, d.day) for d in dates]
    out = []
    for m, dd in md:
        if (m, dd) < WINDOW_START or (m, dd) > WINDOW_END:
            out.append("全年其他")
        elif (m, dd) <= (11, 7):
            out.append("Ⅰ資訊真空期(10/26-11/7)")
        elif (m, dd) <= (11, 14):
            out.append("Ⅱ財報密集期(11/10-11/14)")
        else:
            out.append("Ⅲ財報後期(11/17-11/27)")
    return out


def plot_09_phase_dispersion(pivot):
    sec("09 財報三階段 - 個股橫斷面離散度")
    cross_std = pivot.std(axis=1, skipna=True) * 100
    cross_std = cross_std.dropna()
    phases = contest_phase(cross_std.index)
    d = pd.DataFrame({"cross_std": cross_std.values, "phase": phases}, index=cross_std.index)
    print(d.groupby("phase")["cross_std"].agg(["count", "mean", "median", "std"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    order = ["全年其他", "Ⅰ資訊真空期(10/26-11/7)", "Ⅱ財報密集期(11/10-11/14)", "Ⅲ財報後期(11/17-11/27)"]
    sns.boxplot(data=d, x="phase", y="cross_std", order=order, hue="phase", legend=False,
                palette=["#8fa8c7", "#7fb37f", "#c0435b", "#e0a458"], showfliers=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("當日150檔報酬橫斷面標準差 (%)")
    ax.set_title("財報季三階段 - 個股間報酬分歧程度（2000-2025歷年匯總）")
    ax.tick_params(axis="x", rotation=15)
    savefig(fig, "09_phase_cross_sectional_dispersion.png")


def plot_10_window_liquidity(uni):
    sec("10 比賽視窗 vs 全年其餘 - 流動性比較")
    daily_turnover = uni.groupby(DATE)["turnover"].mean()
    in_win = in_contest_window(daily_turnover.index)
    d = pd.DataFrame({"turnover_億": daily_turnover.values / 1e8,
                       "group": np.where(in_win, "視窗期(10/26-11/27)", "全年(非視窗)")},
                      index=daily_turnover.index)
    print(d.groupby("group")["turnover_億"].agg(["count", "mean", "median"]))

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(data=d, x="group", y="turnover_億", hue="group", legend=False,
                order=["全年(非視窗)", "視窗期(10/26-11/27)"],
                palette=["#8fa8c7", "#c0435b"], ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("個股日均成交金額（億元）")
    ax.set_title("比賽視窗 vs 全年其餘 - 流動性（含95% CI）")
    savefig(fig, "10_window_vs_rest_liquidity.png")


# ============================================================= D. 資料限制回顧 ====
def plot_11_zero_volume_heatmap(uni):
    sec("11 零成交量日熱力圖（ticker x 年）")
    uni2 = uni.copy()
    uni2["year"] = uni2[DATE].dt.year
    zero = uni2[uni2[V] == 0]
    pivot = zero.pivot_table(index=TICKER, columns="year", values=V, aggfunc="count", fill_value=0)
    all_tickers = uni2[TICKER].unique()
    pivot = pivot.reindex(all_tickers, fill_value=0)
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total", ascending=False)
    top_total = pivot["_total"]
    pivot = pivot.drop(columns="_total")
    nonzero = pivot[top_total > 0]
    print(f"有零量紀錄的股票數: {len(nonzero)}/150；零量紀錄最多前10檔:")
    print(top_total[top_total > 0].head(10))

    nm = name_map(uni)
    ylabels = [f"{tk} {nm.get(tk,'')}" for tk in nonzero.index]
    fig_h = max(6, 0.18 * len(nonzero))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    sns.heatmap(nonzero, cmap="rocket_r", ax=ax, yticklabels=ylabels,
                cbar_kws={"label": "零成交量天數"})
    ax.set_xlabel("年")
    ax.set_ylabel("")
    ax.set_title(f"150檔零成交量日分布（{len(nonzero)}檔有紀錄，依總天數排序）")
    savefig(fig, "11_zero_volume_heatmap.png")


def plot_12_coverage_timeline(uni):
    sec("12 資料涵蓋度時間軸")
    g = uni.groupby(TICKER)[DATE].agg(["min", "max"]).sort_values("min")
    nm = name_map(uni)
    fig, ax = plt.subplots(figsize=(11, 20))
    y = np.arange(len(g))
    ax.hlines(y, g["min"], g["max"], color="#3b6ea5", lw=2.2)
    sns.scatterplot(x=g["min"], y=y, color="#2c5f8a", s=14, ax=ax, zorder=3, legend=False)
    sns.scatterplot(x=g["max"], y=y, color="#c0435b", s=14, ax=ax, zorder=3, legend=False)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{tk} {nm.get(tk,'')}" for tk in g.index], fontsize=6)
    ax.set_xlabel("日期")
    ax.set_title("150檔資料涵蓋期間（藍點=首筆, 紅點=末筆）")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    savefig(fig, "12_coverage_timeline.png")


# ==================================================================== main ====
def main():
    import os
    os.makedirs(FIG_DIR, exist_ok=True)

    uni = load_universe()

    plot_01_sector_composition(uni)
    plot_02_history_length(uni)

    rdates = recent_window_dates(uni)
    insufficient = plot_03_liquidity(uni, rdates)
    plot_04_volatility(uni, rdates, insufficient)

    pivot_all, index_ret = build_equal_weight_index(uni)
    plot_05_rolling_vol(index_ret)
    plot_06_momentum_snapshot(uni)
    plot_07_correlation(uni, rdates)

    plot_08_window_vs_rest(index_ret)
    plot_09_phase_dispersion(pivot_all)
    plot_10_window_liquidity(uni)

    plot_11_zero_volume_heatmap(uni)
    plot_12_coverage_timeline(uni)

    sec("DONE")


if __name__ == "__main__":
    main()
