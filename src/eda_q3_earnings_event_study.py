"""
AI CUP 2026 玉山挑戰賽 — Q3財報公布對股價的影響（event study）
純描述性分析，不含選股/策略邏輯。

背景：比賽視窗 2026/10/26-11/27 剛好覆蓋台股Q3(7-9月)財報法定申報截止日
(季末後45日 = 11/14)。我們手上沒有每家公司「實際申報日期」這種逐筆公告時間，
只有 rev_prof_net.csv 裡的「所屬季度」(年/月)。因此這裡採用市場慣例做法：
用法定截止日 11/14 當作事件錨點(day0)——因為台股慣例上大量公司集中在截止日
前幾天才申報(拖到最後一刻)，用截止日當錨點是業界常見的近似做法，
不是每家公司的精確公告日，解讀時要記得這個限制。

兩塊分析：
  A. 市場層級：截止日前後(-15~+15交易日) cross-sectional 報酬分歧度/成交量變化，
     多年(2005-2025)平均，驗證「財報公布真的會讓股價變動放大」。
  B. 個股層級：用 Q3 合併總損益 YoY 成長率做「盈餘驚奇」代理指標，分五分位，
     看截止日前後各分位的累積報酬走勢是否分化(post-earnings-announcement drift)，
     驗證財報好壞是否真的對後續股價有可辨識的影響方向。
"""

import warnings
warnings.filterwarnings("ignore")

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
FUND_PATH = "data/rev_prof_net.csv"
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

Q3_DEADLINE = (11, 14)  # 季末(9/30)後45日法定申報截止日
WINDOW = 15             # 事件窗：day0 前後各15個交易日
YEARS = range(2008, 2026)  # 2008年後Q1/Q3才普遍申報(見rev_prof_net_summary.md A4)；2026年競賽當年也涵蓋


def sec(msg):
    print(f"\n{'='*10} {msg} {'='*10}")


# ------------------------------------------------------------ load ----
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
    print("shape:", uni.shape, "| 檔數:", uni[TICKER].nunique(), "| 日期範圍:", uni[DATE].min().date(), "~", uni[DATE].max().date())
    return uni


def load_fundamentals():
    sec("Loading rev_prof_net.csv，取Q3(9月)合併總損益")
    df = pd.read_csv(FUND_PATH, low_memory=False)
    for c in ["營業收入淨額", "營業利益", "稅前淨利", "合併總損益"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    q3 = df[df["季別"] == 3].copy()
    q3["年"] = q3["年/月"] // 100
    q3 = q3[q3[TICKER].isin(UNIVERSE)][[TICKER, "年", "合併總損益"]].drop_duplicates()
    return q3


# ------------------------------------------------------------ A: 市場層級事件窗 ----
def market_event_study(prices):
    sec("A. 市場層級：Q3申報截止日(11/14)前後 cross-sectional 分歧度/成交量")

    trading_days = np.array(sorted(prices[DATE].unique()))
    ret_wide = prices.pivot(index=DATE, columns=TICKER, values="ret").reindex(trading_days)
    to_wide = prices.pivot(index=DATE, columns=TICKER, values="turnover").reindex(trading_days)

    panels = []
    for y in YEARS:
        target = pd.Timestamp(y, *Q3_DEADLINE)
        idx = np.searchsorted(trading_days, np.datetime64(target), side="left")
        if idx <= WINDOW or idx >= len(trading_days) - WINDOW:
            continue
        for o in range(-WINDOW, WINDOW + 1):
            d = trading_days[idx + o]
            row_ret = ret_wide.loc[d]
            row_to = to_wide.loc[d]
            panels.append({
                "year": y, "offset": o, "date": d,
                "cs_mean_abs_ret": row_ret.abs().mean(),
                "cs_std_ret": row_ret.std(),
                "n_tickers": row_ret.notna().sum(),
                "median_turnover": row_to.median(),
            })
    panel = pd.DataFrame(panels)
    agg = panel.groupby("offset")[["cs_mean_abs_ret", "cs_std_ret", "median_turnover"]].mean()

    baseline_std = agg.loc[(agg.index < -5) | (agg.index > 5), "cs_std_ret"].mean()
    event_std = agg.loc[(agg.index >= -1) & (agg.index <= 3), "cs_std_ret"].mean()
    print(f"截止日窗外(|offset|>5)平均 cross-sectional 報酬std: {baseline_std:.4%}")
    print(f"截止日附近(offset -1~+3)平均 cross-sectional 報酬std: {event_std:.4%}")
    print(f"放大倍數: {event_std/baseline_std:.2f}x")

    baseline_to = agg.loc[(agg.index < -5) | (agg.index > 5), "median_turnover"].mean()
    event_to = agg.loc[(agg.index >= -1) & (agg.index <= 3), "median_turnover"].mean()
    print(f"\n截止日窗外 中位數成交金額: {baseline_to/1e8:.2f}億")
    print(f"截止日附近 中位數成交金額: {event_to/1e8:.2f}億 (放大 {event_to/baseline_to:.2f}x)")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(agg.index, agg["cs_std_ret"] * 100, marker="o", ms=3, color="#C44E52")
    axes[0].axvline(0, color="gray", ls="--", lw=1)
    axes[0].set_ylabel("cross-sectional 報酬std (%)")
    axes[0].set_title(f"Q3財報申報截止日(day0=11/14)前後 個股報酬分歧度（{YEARS.start}-{YEARS.stop-1}年平均）")

    axes[1].plot(agg.index, agg["median_turnover"] / 1e8, marker="o", ms=3, color="#4C72B0")
    axes[1].axvline(0, color="gray", ls="--", lw=1)
    axes[1].set_ylabel("中位數成交金額 (億元)")
    axes[1].set_xlabel("相對截止日的交易日偏移 (0=法定截止日或次一交易日)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/15_q3_deadline_event_window.png", dpi=150)
    plt.close(fig)

    return ret_wide, trading_days


# ------------------------------------------------------------ B: 個股層級 PEAD ----
def surprise_drift_study(prices, ret_wide, trading_days, fund_q3):
    sec("B. 個股層級：Q3盈餘驚奇(YoY合併總損益成長) x 截止日後累積報酬")

    fund_q3 = fund_q3.sort_values([TICKER, "年"]).copy()
    fund_q3["prev"] = fund_q3.groupby(TICKER)["合併總損益"].shift(1)
    fund_q3["yoy_growth"] = np.where(
        fund_q3["prev"].abs() > 1e-6,
        (fund_q3["合併總損益"] - fund_q3["prev"]) / fund_q3["prev"].abs(),
        np.nan,
    )
    fund_q3 = fund_q3.dropna(subset=["yoy_growth"])

    def qbucket(s):
        try:
            return pd.qcut(s, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=s.index)

    fund_q3["quintile"] = fund_q3.groupby("年")["yoy_growth"].transform(qbucket)
    print(f"可算出YoY盈餘成長分位的 公司-年 組合數: {fund_q3['quintile'].notna().sum()}")
    print("各分位YoY成長中位數 (合併全部年度)：")
    print(fund_q3.groupby("quintile", observed=True)["yoy_growth"].median())

    records = []
    for _, r in fund_q3.dropna(subset=["quintile"]).iterrows():
        tid, y, q = r[TICKER], int(r["年"]), r["quintile"]
        target = pd.Timestamp(y, *Q3_DEADLINE)
        idx = np.searchsorted(trading_days, np.datetime64(target), side="left")
        if idx <= WINDOW or idx >= len(trading_days) - WINDOW or tid not in ret_wide.columns:
            continue
        window = ret_wide[tid].iloc[idx - WINDOW: idx + WINDOW + 1].values
        if np.isnan(window).sum() > WINDOW:  # 資料太不完整就跳過
            continue
        records.append((tid, y, q, window))

    if not records:
        print("沒有足夠資料做個股層級分析")
        return

    offsets = np.arange(-WINDOW, WINDOW + 1)
    df_panel = pd.DataFrame([r[3] for r in records], columns=offsets)
    df_panel["quintile"] = [r[2] for r in records]
    print(f"\n可配對到事件窗價量的 公司-年 樣本數: {len(df_panel)}")
    print(df_panel["quintile"].value_counts().sort_index())

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("RdYlGn", 5)
    for q in [1, 2, 3, 4, 5]:
        sub = df_panel[df_panel["quintile"] == q][offsets]
        if sub.empty:
            continue
        car = (1 + sub.fillna(0)).cumprod(axis=1).mean(axis=0) - 1
        ax.plot(offsets, car * 100, label=f"Q{q} (盈餘驚奇{'最差' if q==1 else '最佳' if q==5 else ''})",
                color=palette[q - 1], lw=2)
        print(f"Q{q}: day0 CAR={car.loc[0]*100:.2f}%, day+{WINDOW} CAR={car.loc[WINDOW]*100:.2f}%")

    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("Q3盈餘驚奇五分位 x 截止日前後累積報酬 (CAR, %)")
    ax.set_xlabel("相對申報截止日的交易日偏移")
    ax.set_ylabel("累積報酬 (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/16_q3_surprise_quintile_car.png", dpi=150)
    plt.close(fig)

    pre = df_panel[list(range(-WINDOW, 0))].fillna(0)
    post = df_panel[list(range(1, WINDOW + 1))].fillna(0)
    df_panel["pre_car"] = (1 + pre).prod(axis=1) - 1
    df_panel["post_car"] = (1 + post).prod(axis=1) - 1
    spread = df_panel.groupby("quintile", observed=True)[["pre_car", "post_car"]].mean()
    print("\n各分位 截止日前 vs 截止日後 平均累積報酬：")
    print(spread)
    q5_q1_post = spread.loc[5, "post_car"] - spread.loc[1, "post_car"]
    print(f"\nQ5(驚喜最佳) - Q1(驚喜最差) 截止日後累積報酬價差: {q5_q1_post*100:.2f}%")


def main():
    prices = load_prices()
    ret_wide, trading_days = market_event_study(prices)
    fund_q3 = load_fundamentals()
    surprise_drift_study(prices, ret_wide, trading_days, fund_q3)
    sec("完成")


if __name__ == "__main__":
    main()
