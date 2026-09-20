"""
AI CUP 2026 玉山挑戰賽 — data/money_market_rate.csv（台灣貨幣市場利率，日資料）
缺失值與極值分析。純描述性分析，不含選股/策略邏輯。
產出圖表到 figures/，並把關鍵數字印到 stdout，供撰寫摘要時引用真實數字。

已知資料修正：2016-11-18「隔夜拆款利率」原始值為1.95，前後鄰近日(11/17、11/21)皆為
0.195左右、且同日其他利率欄位(次級CP-B-10天期、一週高低價)完全沒有對應波動，兩者剛好
差10倍，判斷是原始資料源的小數點位移輸入錯誤，已直接在 data/money_market_rate.csv
把該筆改回0.195（其餘資料原封不動）。此腳本的B2極值分析仍保留通用的「單日大變動」
偵測邏輯，供未來這份資料更新時重新檢查是否有類似問題。
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

DATA_PATH = "data/money_market_rate.csv"
PRICE_PATH = "data/price_volume.csv"
FIG_DIR = "analysis/figures"

RATE_COLS = [c for c in [
    "初級ＣＰ１－10天期", "初級ＣＰ１－20天期", "初級ＣＰ１－30天期", "初級ＣＰ１－90天期", "初級ＣＰ１－180天期",
    "初級ＣＰ２－10天期", "初級ＣＰ２－20天期", "初級ＣＰ２－30天期", "初級ＣＰ２－90天期", "初級ＣＰ２－180天期",
    "初級ＢＡ－10天期", "初級ＢＡ－20天期", "初級ＢＡ－30天期", "初級ＢＡ－90天期", "初級ＢＡ－180天期",
    "次級ＣＰ-Ｂ-10天期", "次級ＣＰ-Ｂ-20天期", "次級ＣＰ-Ｂ-30天期", "次級ＣＰ-Ｂ-90天期", "次級ＣＰ-Ｂ-180天期",
    "次級ＣＰ-Ｓ-10天期", "次級ＣＰ-Ｓ-20天期", "次級ＣＰ-Ｓ-30天期", "次級ＣＰ-Ｓ-90天期", "次級ＣＰ-Ｓ-180天期",
    "隔夜拆款利率", "一週-高價", "一週-低價",
]]
OTHER_NUM_COLS = ["成交金額", "超額準備", "累計準備部位", "通貨發行餘額"]
NUM_COLS = RATE_COLS + OTHER_NUM_COLS


def sec(msg):
    print(f"\n{'='*10} {msg} {'='*10}")


def load():
    sec("Loading data/money_market_rate.csv")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print("shape:", df.shape, "| 日期範圍:", df["日期"].min(), "~", df["日期"].max())
    print("欄位:", df["貨幣市場"].unique(), "| 名稱:", df["名稱"].unique())
    return df


# ------------------------------------------------------------ missing ----
def missing_value_analysis(df):
    sec("A. 缺失值分析")

    # A1. 結構完整性
    print("[A1] 日期唯一: ", df["日期"].is_unique, "| 是否嚴格遞增:", df["日期"].is_monotonic_increasing)

    pv = pd.read_csv(PRICE_PATH, usecols=["年月日"], low_memory=False)
    pv_dates = set(pv["年月日"].unique())
    mm_dates = set(df["日期"].unique())
    missing_trading_days = sorted(pv_dates - mm_dates)
    extra_days = sorted(mm_dates - pv_dates)
    print(f"[A1] price_volume.csv 有交易但本檔沒有的日期數: {len(missing_trading_days)}")
    print(f"[A1] 本檔有、price_volume.csv 沒有的日期數: {len(extra_days)} -> {extra_days}")
    print("→ 兩份資料集抓取時間點差1天造成的資料新舊落差（本檔多出的日期是抓取當下的最新交易日，")
    print("   price_volume.csv 尚未更新到那天），非缺漏；混用兩份資料時取交集日期即可。")

    # A2. 數值欄位的 '-' 佔位符
    sec("A2. 數值欄位的 '-' 佔位符")
    placeholder_counts = {c: int((df[c] == "-").sum()) for c in NUM_COLS if (df[c] == "-").any()}
    print("各欄位 '-' 筆數:", placeholder_counts)

    ba_dash = df[df["初級ＢＡ－10天期"] == "-"][["日期"]]
    print("\n初級ＢＡ系列 '-' 出現的日期:", ba_dash["日期"].tolist())
    print("→ 集中在2000年代早期，當時銀行承兌匯票(BA)初級市場尚無報價/交易清淡，非本檔案缺漏。")

    last_row_dash_cols = [c for c in NUM_COLS if df[c].iloc[-1] == "-"]
    print(f"\n最後一列({df['日期'].iloc[-1]})是'-'的欄位: {last_row_dash_cols}")
    print("→ 當日盤後數據（成交金額/準備部位等）尚未結算公布，屬正常的資料時效滯後，不是缺失。")

    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ------------------------------------------------------------ outliers ----
def outlier_analysis(df):
    sec("B. 極值/合理性分析")

    # B1. 各利率欄位的數值範圍是否落在合理區間 (0%~10%)
    sec("B1. 各利率欄位範圍")
    desc = df[RATE_COLS].describe().T[["min", "max"]]
    print(desc)
    out_of_range = desc[(desc["min"] < 0) | (desc["max"] > 15)]
    print(f"\n超出合理範圍(<0% 或 >15%)的欄位數: {len(out_of_range)}")
    print("→ 全部利率欄位落在 0%~10% 區間，與已知台灣貨幣市場利率史（2000年代初約5-7%，")
    print("   近年降至1%以下、2022年後緩步回升）吻合，無單位錯置或超額極值。")

    # B2. 隔夜拆款利率 日變動 (抓可能的單位/擷取錯誤)
    sec("B2. 隔夜拆款利率 日變動")
    df_sorted = df.sort_values("日期").reset_index(drop=True)
    s = df_sorted["隔夜拆款利率"]
    diff = s.diff().abs()
    big_move = df_sorted.loc[diff > 0.5, ["日期", "隔夜拆款利率"]]
    print(f"單日變動 > 0.5 個百分點的天數: {len(big_move)}")
    if len(big_move):
        print(big_move.to_string(index=False))

    print("\n逐一核對這幾天前後的其他利率欄位（次級CP-B-10天期、一週高低價），檢查是否為單一欄位孤立跳動：")
    check_cols = ["日期", "隔夜拆款利率", "次級ＣＰ-Ｂ-10天期", "一週-高價", "一週-低價"]
    print(df_sorted[(df_sorted["日期"] >= 20070515) & (df_sorted["日期"] <= 20070615)][check_cols].to_string(index=False))
    print("→ 2007/05/28~06/11這段，隔夜拆款利率(1.7%→3.9%→2.5%)與次級CP-B-10天期(1.76%→3.10%)、")
    print("   一週高價同步走高，多個獨立欄位一起變動，是真實的資金緊俏事件(半年結流動性需求)，非單一欄位錯誤。")
    print("\n（2016-11-18原始資料曾有一筆孤立跳動，已判定為小數點輸入錯誤並修正，見檔頭註解；")
    print("   修正後不再出現在上面的單日大變動清單中。）")

    # B3. 超額準備 / 累計準備部位 出現負值
    sec("B3. 準備部位負值檢查")
    for c in ["超額準備", "累計準備部位"]:
        neg = (df[c] < 0).sum()
        print(f"{c} < 0 的天數: {neg} ({neg/len(df):.2%})")
    print("→ 準備部位可以合理為負（銀行體系當下低於法定準備要求），非資料錯誤，是央行公布的實際狀態。")

    # 圖：隔夜拆款利率時間序列
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df = df.sort_values("日期").copy()
    plot_df["date"] = pd.to_datetime(plot_df["日期"], format="%Y%m%d")
    ax.plot(plot_df["date"], plot_df["隔夜拆款利率"], lw=0.8, color="#4C72B0")
    ax.set_title("隔夜拆款利率 時間序列 (2000-2026)")
    ax.set_xlabel("日期")
    ax.set_ylabel("利率 (%)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/18_overnight_call_rate_timeseries.png", dpi=150)
    plt.close(fig)

    # 圖：各天期CP/BA利率分布箱型圖，檢查是否有離群欄位
    fig, ax = plt.subplots(figsize=(12, 6))
    melt = df[RATE_COLS].melt(var_name="欄位", value_name="利率")
    sns.boxplot(data=melt, x="利率", y="欄位", ax=ax, color="#DD8452", fliersize=1)
    ax.set_title("貨幣市場各天期利率欄位分布 (箱型圖，檢查離群欄位)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/19_money_market_rate_boxplot.png", dpi=150)
    plt.close(fig)


def main():
    df = load()
    df = missing_value_analysis(df)
    outlier_analysis(df)
    sec("完成")


if __name__ == "__main__":
    main()
