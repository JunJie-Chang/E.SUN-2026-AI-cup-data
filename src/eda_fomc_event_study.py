"""
AI CUP 2026 玉山挑戰賽 — FOMC利率決策對台股(150檔投資池)的影響（event study）
純描述性分析，不含選股/策略邏輯。

背景：比賽視窗(10/26-11/27)涵蓋 2026/10/27-28 FOMC會議(台灣時間10/29公布結果)。
這裡跟 eda_q3_earnings_event_study.py 用同一套方法論（cross-sectional 分歧度、
成交量），但額外加上「市場整體方向性幅度」(|150檔等權重平均報酬|)，因為
FOMC屬於系統性(systematic)總體衝擊、理論上比較可能讓「全市場一起動」，
而財報屬於個股異質性(idiosyncratic)衝擊、比較可能拉開「個股間分歧」——
兩種事件在dispersion這個指標上不見得可比，所以這裡兩種指標都測。

FOMC會議日期來源：直接爬 federalreserve.gov 官方歷史頁面
(fomchistoricalYYYY.htm 2008-2020, fomccalendars.htm 2021-2026)，
用規則式解析(regex)取得「decision day」(每次會議的最後一天，即公布決策當天)，
不是靠AI摘要，日期經過交叉檢查、134筆(扣除2020年3月因COVID臨時加開的
非例行會議)。決策公布時間是美東下午2點，換算台灣時間是次日凌晨，
所以事件錨點day0設定為「決策後第一個台灣交易日」。
"""

import warnings
warnings.filterwarnings("ignore")

import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Heiti TC", "PingFang TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PRICE_PATH = "data/price_volume.csv"
FIG_DIR = "analysis/figures"

O, H, L, C, V = "開盤價(元)", "最高價(元)", "最低價(元)", "收盤價(元)", "成交量(千股)"
TICKER, NAME, DATE, SECTOR = "公司簡稱", "名稱", "年月日", "TSE產業_名稱"

UNIVERSE = """2330 2454 2308 2317 3711 2881 2383 2303 2882 3037 2891 1303 2345 2382 2408 7769 2412 2327 6669 3017
2885 2887 2360 2886 2059 6505 2884 2880 2357 2890 2883 3443 3653 2395 2301 6446 4958 2603 5880 1216
3045 2368 3665 4904 3481 2379 1301 1326 3189 3034 2207 2002 2801 2449 1590 3661 6770 3036 2615 2618
8046 2344 3231 2892 3008 2356 2404 6515 3533 2313 2337 5871 3044 3702 1101 2409 6239 2609 2834 6139
2912 4938 2376 5876 1519 6415 6919 2324 2347 1504 7750 1402 1605 2610 6789 1802 8210 2451 2812 2377
5274 6223 6488 8299 6274 5347 3293 8069 3529 3081 3260 3105 5289 5536 5483 6147 7734 6187 3264 3324
6510 8358 3374 3131 4749 3491 6121 6548 1785 3363 7828 4979 6182 3163 3211 4966 4991 5903 1815 3680
8415 6290 7751 6023 4772 8932 6584 3227 4123 3718""".split()
UNIVERSE = [int(x) for x in UNIVERSE]
assert len(UNIVERSE) == 150

# 2008-2026 FOMC decision days（每次會議最後一天/公布政策聲明當天，美東時間）。
# 來源：federalreserve.gov 官方歷史行事曆頁面，規則式解析，2026-09 抓取。
# 2020年3月因COVID臨時加開的3次非例行會議(3/3, 3/15, 3/23, 3/31)不計入，
# 只保留例行排定的8次會議，避免跟疫情本身的極端波動混在一起。
FOMC_DECISION_DATES_STR = """
2008-01-30 2008-03-18 2008-04-30 2008-06-25 2008-08-05 2008-09-16 2008-10-29 2008-12-16
2009-01-28 2009-03-18 2009-04-29 2009-06-24 2009-08-12 2009-09-23 2009-11-04 2009-12-16
2010-01-27 2010-03-16 2010-04-28 2010-06-23 2010-08-10 2010-09-21 2010-11-03 2010-12-14
2011-01-26 2011-03-15 2011-04-27 2011-06-22 2011-08-09 2011-09-21 2011-11-02 2011-12-13
2012-01-25 2012-03-13 2012-04-25 2012-06-20 2012-08-01 2012-09-13 2012-10-24 2012-12-12
2013-01-30 2013-03-20 2013-05-01 2013-06-19 2013-07-31 2013-09-18 2013-10-30 2013-12-18
2014-01-29 2014-03-19 2014-04-30 2014-06-18 2014-07-30 2014-09-17 2014-10-29 2014-12-17
2015-01-28 2015-03-18 2015-04-29 2015-06-17 2015-07-29 2015-09-17 2015-10-28 2015-12-16
2016-01-27 2016-03-16 2016-04-27 2016-06-15 2016-07-27 2016-09-21 2016-11-02 2016-12-14
2017-02-01 2017-03-15 2017-05-03 2017-06-14 2017-07-26 2017-09-20 2017-11-01 2017-12-13
2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19
2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11
2020-01-29 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16
2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
"""
FOMC_DECISION_DATES = sorted(pd.Timestamp(d) for d in FOMC_DECISION_DATES_STR.split())
assert len(FOMC_DECISION_DATES) == 8 * 18 - 1  # 2020年只有7次例行會議

WINDOW = 15


def sec(msg):
    print(f"\n{'='*10} {msg} {'='*10}")


def load_prices():
    sec("Loading price_volume.csv，過濾至150檔投資池")
    df = pd.read_csv(PRICE_PATH, dtype={TICKER: "string"})
    df[DATE] = pd.to_datetime(df[DATE].astype(str), format="%Y%m%d")
    df[TICKER] = df[TICKER].astype(int)
    uni = df[df[TICKER].isin(UNIVERSE)].copy()
    del df
    uni = uni.sort_values([TICKER, DATE]).reset_index(drop=True)
    uni["ret"] = uni.groupby(TICKER)[C].pct_change()
    uni["turnover"] = uni[C] * uni[V] * 1000.0
    print("shape:", uni.shape, "| 檔數:", uni[TICKER].nunique())
    return uni


def fomc_event_study(prices):
    sec("FOMC決策日(day0=決策後第一個台灣交易日)前後 市場反應")

    trading_days = np.array(sorted(prices[DATE].unique()))
    ret_wide = prices.pivot(index=DATE, columns=TICKER, values="ret").reindex(trading_days)
    to_wide = prices.pivot(index=DATE, columns=TICKER, values="turnover").reindex(trading_days)
    index_ret = ret_wide.mean(axis=1)  # 150檔等權重代理指數報酬

    panels = []
    used_events = 0
    for fomc_date in FOMC_DECISION_DATES:
        target = fomc_date + pd.Timedelta(days=1)  # 決策後第一個台灣交易日
        idx = np.searchsorted(trading_days, np.datetime64(target), side="left")
        if idx <= WINDOW or idx >= len(trading_days) - WINDOW:
            continue
        used_events += 1
        for o in range(-WINDOW, WINDOW + 1):
            d = trading_days[idx + o]
            row_ret = ret_wide.loc[d]
            panels.append({
                "fomc_date": fomc_date, "offset": o, "date": d,
                "cs_mean_abs_ret": row_ret.abs().mean(),
                "cs_std_ret": row_ret.std(),
                "index_abs_ret": abs(index_ret.loc[d]),
                "median_turnover": to_wide.loc[d].median(),
            })
    panel = pd.DataFrame(panels)
    print(f"可用事件數: {used_events} / {len(FOMC_DECISION_DATES)}")
    # price_volume.csv 已知有少數還原權值批次重算造成的單日跳動雜訊(見rev_prof_net_summary.md /
    # 09phase分析同樣遇過)，例如2014-07-15單日cross-sectional std飆到1204%，是單一個股的
    # 資料異常，不是真的FOMC效應；用median而非mean聚合可以避免這種離群值主導平均數。
    agg = panel.groupby("offset")[["cs_mean_abs_ret", "cs_std_ret", "index_abs_ret", "median_turnover"]].median()

    baseline_std = agg.loc[(agg.index < -5) | (agg.index > 5), "cs_std_ret"].median()
    event_std = agg.loc[(agg.index >= 0) & (agg.index <= 2), "cs_std_ret"].median()
    baseline_idx = agg.loc[(agg.index < -5) | (agg.index > 5), "index_abs_ret"].median()
    event_idx = agg.loc[(agg.index >= 0) & (agg.index <= 2), "index_abs_ret"].median()
    baseline_to = agg.loc[(agg.index < -5) | (agg.index > 5), "median_turnover"].median()
    event_to = agg.loc[(agg.index >= 0) & (agg.index <= 2), "median_turnover"].median()

    print(f"\n--- cross-sectional 分歧度 (個股間差異程度) ---")
    print(f"窗外(|offset|>5)平均std: {baseline_std:.4%}  |  day0~+2平均std: {event_std:.4%}  |  放大倍數: {event_std/baseline_std:.2f}x")
    print(f"(對照：Q3財報截止日 event study 量到的放大倍數約 1.02x)")

    print(f"\n--- 市場整體方向性幅度 (150檔等權重代理指數 |日報酬|) ---")
    print(f"窗外平均: {baseline_idx:.4%}  |  day0~+2平均: {event_idx:.4%}  |  放大倍數: {event_idx/baseline_idx:.2f}x")

    print(f"\n--- 成交金額 ---")
    print(f"窗外中位數: {baseline_to/1e8:.2f}億  |  day0~+2中位數: {event_to/1e8:.2f}億  |  放大倍數: {event_to/baseline_to:.2f}x")

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(agg.index, agg["index_abs_ret"] * 100, marker="o", ms=3, color="#4C72B0")
    axes[0].axvline(0, color="gray", ls="--", lw=1)
    axes[0].set_ylabel("代理指數 |日報酬| (%)")
    axes[0].set_title(f"FOMC決策日前後 台股150檔投資池反應 ({FOMC_DECISION_DATES[0].year}-{FOMC_DECISION_DATES[-1].year}年平均，{used_events}次會議)")

    axes[1].plot(agg.index, agg["cs_std_ret"] * 100, marker="o", ms=3, color="#C44E52")
    axes[1].axvline(0, color="gray", ls="--", lw=1)
    axes[1].set_ylabel("cross-sectional 報酬std (%)")

    axes[2].plot(agg.index, agg["median_turnover"] / 1e8, marker="o", ms=3, color="#55A868")
    axes[2].axvline(0, color="gray", ls="--", lw=1)
    axes[2].set_ylabel("中位數成交金額 (億元)")
    axes[2].set_xlabel("相對FOMC決策後首個台灣交易日的偏移 (0=day0)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/17_fomc_event_window.png", dpi=150)
    plt.close(fig)


def main():
    prices = load_prices()
    fomc_event_study(prices)
    sec("完成")


if __name__ == "__main__":
    main()
