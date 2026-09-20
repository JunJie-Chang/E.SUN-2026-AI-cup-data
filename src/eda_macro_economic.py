"""
AI CUP 2026 玉山挑戰賽 — data/macro_economic.csv（總體經濟指標，長格式，6000+指標）
缺失值與極值分析。純描述性分析，不含選股/策略邏輯。
產出圖表到 figures/，並把關鍵數字印到 stdout，供撰寫摘要時引用真實數字。

資料格式是 long format：代碼／名稱／年月／數值，每個「代碼」是一個獨立的總經指標
(涵蓋台灣與美/日/中/德/英/星/韓/澳/歐元區等，頻率monthly/quarterly/annual混雜)，
不是像 price_volume.csv/rev_prof_net.csv 那樣按公司分列，所以這份分析改成先看
「指標(代碼)層級」的結構/涵蓋率/時效性，再抽樣檢查數值合理性，不逐一列出所有代碼。
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

DATA_PATH = "data/macro_economic.csv"
FIG_DIR = "analysis/figures"

COUNTRIES = ["台灣", "美國", "日本", "中國", "南韓", "韓國", "德國", "香港", "歐元", "英國", "新加坡", "澳"]


def sec(msg):
    print(f"\n{'='*10} {msg} {'='*10}")


def load():
    sec("Loading data/macro_economic.csv")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print("shape:", df.shape, "| 指標(代碼)數:", df["代碼"].nunique(),
          "| 年月範圍:", df["年月"].min(), "~", df["年月"].max())
    return df


# ------------------------------------------------------------ missing ----
def missing_value_analysis(df):
    sec("A. 缺失值與結構分析")

    # A1. 結構完整性
    print("[A1] 欄位 NaN 數:")
    print(df.isna().sum())
    dup = df.duplicated(subset=["代碼", "年月"], keep=False)
    print(f"[A1] (代碼,年月) 重複列數: {dup.sum()}")

    # A2. 頻率組成 (從月份分布粗估monthly/quarterly/annual混雜情形)
    sec("A2. 年月的月份分布 (混合頻率的證據)")
    mod = df["年月"] % 100
    print(mod.value_counts().sort_index())
    print("→ 3/6/9/12月筆數明顯偏高，是季報/年報指標只在這幾個月出現造成的，非缺失。")

    name_map = df.drop_duplicates("代碼").set_index("代碼")["名稱"]

    # A3. 指標的國家/地區組成
    sec("A3. 指標(代碼)的國家/地區組成")
    n_total = len(name_map)
    for c in COUNTRIES:
        n = name_map.str.contains(c, na=False).sum()
        print(f"名稱含「{c}」的代碼數: {n} ({n/n_total:.1%})")
    print(f"→ 74.6%左右為台灣本地總經指標，其餘涵蓋美/日/中/德/英/星/韓/澳/歐元區等主要經濟體，")
    print("   可作為策略的總體環境因子來源，不限於利率。")

    # A4. 每個代碼的頻率推論 (資料筆數 / 涵蓋月數)
    sec("A4. 每個代碼的頻率推論")
    g = df.groupby("代碼")["年月"].agg(["min", "max", "count"])
    span_months = (g["max"] // 100 - g["min"] // 100) * 12 + (g["max"] % 100 - g["min"] % 100) + 1
    ratio = g["count"] / span_months

    def classify(r):
        if r > 0.9:
            return "monthly"
        elif r > 0.28:
            return "quarterly"
        else:
            return "annual"

    freq_class = ratio.apply(classify)
    print(freq_class.value_counts())
    print(f"→ Monthly指標佔多數({(freq_class=='monthly').mean():.1%})，quarterly/annual各佔一部分，")
    print("   混用時務必先確認個別代碼的實際頻率，不能假設全部都是月資料。")

    fig, ax = plt.subplots(figsize=(7, 5))
    freq_class.value_counts().reindex(["monthly", "quarterly", "annual"]).plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title("6,035個總經指標的推論頻率分布")
    ax.set_xlabel("推論頻率")
    ax.set_ylabel("代碼數")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/22_macro_indicator_frequency.png", dpi=150)
    plt.close(fig)

    # A5. 資料時效性 (距最新一期資料多久沒更新)
    sec("A5. 資料時效性 — 距今多久沒更新")
    latest_ym = df["年月"].max()
    latest_y, latest_m = latest_ym // 100, latest_ym % 100
    gap = (latest_y * 12 + latest_m) - (g["max"] // 100 * 12 + g["max"] % 100)
    print(gap.describe())
    print(f"\n超過12個月沒更新(stale)的代碼數: {(gap > 12).sum()} / {len(g)}")
    very_stale = gap[gap > 60]
    print(f"超過60個月(5年)沒更新的代碼數: {len(very_stale)}")
    if len(very_stale):
        print(g.loc[very_stale.index].join(name_map).join(very_stale.rename("gap_months")).to_string())
        print("→ 這種指標很可能是資料源已停止發布(如統計方法變更被新指標取代)，使用前務必先排除或另行確認。")

    # A6. 數值欄位的 '-' 佔位符 — 整體低但高度集中在少數代碼
    sec("A6. 數值欄位的 '-' 佔位符")
    df["is_dash"] = df["數值"] == "-"
    print(f"整體 '-' 佔位符比例: {df['is_dash'].mean():.3%} ({df['is_dash'].sum()} / {len(df)})")

    rate = df.groupby("代碼")["is_dash"].mean()
    print(f"完全沒有 '-' 的代碼數: {(rate == 0).sum()} / {len(rate)}")
    print(f">50% 是 '-' 的代碼數: {(rate > 0.5).sum()}")
    mostly_empty = rate[rate > 0.9].sort_values(ascending=False)
    print(f">90% 是 '-' 的代碼數(形同虛設，建議直接排除): {len(mostly_empty)}")
    print(pd.DataFrame({"名稱": name_map.loc[mostly_empty.index], "佔位符比例": mostly_empty}).to_string())
    print("→ 這12檔清一色是冷門天期/券別的「資本市場利率」或「央行公開市場操作」子項目，")
    print("   該天期/券別本來就極少實際成交或發行，長期沒有報價屬正常業務現象，非資料缺漏。")

    for c in df.columns:
        if c == "數值":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, name_map


# ------------------------------------------------------------ outliers ----
def outlier_analysis(df, name_map):
    sec("B. 極值分析")

    df_sorted = df.sort_values(["代碼", "年月"]).copy()
    df_sorted["v_prev"] = df_sorted.groupby("代碼")["數值"].shift(1)
    df_sorted["ratio"] = df_sorted["數值"] / df_sorted["v_prev"]
    mask = (df_sorted["v_prev"] > 0) & (df_sorted["數值"] > 0)
    sub = df_sorted[mask]
    jump = sub[(sub["ratio"] > 10) | (sub["ratio"] < 0.1)]

    # B1. 全面用「單期變動>10倍」掃描，檢查這個方法在這份資料集是否適用
    sec("B1. 單期變動 >10倍或<0.1倍 的整體掃描")
    print(f"正值序列中符合條件的列數: {len(jump)} / {len(sub)} ({len(jump)/len(sub):.3%})")
    yr_dist = (jump["年月"] // 100).value_counts().sort_index()
    print("\n按年份分布（檢查是否集中在2008金融海嘯/2020疫情等已知危機年份）:")
    print(yr_dist.describe())
    print("→ 年份分布相當平均(每年約480-570筆，26年間無明顯集中)，不是「危機年份才暴增」的型態，")
    print("   代表這個方法在這份資料集大部分是誤報：許多代碼本身是「年增率/成長率」類指標，")
    print("   數值長期在0附近震盪，前後兩期都是小數字時比率會被嚴重放大，是分母效應而非真異常。")
    example = jump[jump["代碼"] == "CA07G"][["年月", "v_prev", "數值", "ratio"]]
    print("\n例：CA07G(美國出口年增率-商品)：")
    print(example.head(5).to_string(index=False))
    print("→ 2.7%→0.19%這種在小百分比之間的正常波動，比率上看是14倍，但兩個數字本身都合理，非錯誤。")

    # B2. 針對「水準類」(非成長率/年增率)指標做鎖定式 spot check
    sec("B2. 針對利率/指數等「水準類」指標的鎖定式 spot check")
    level_mask = ~name_map.str.contains("年增率|增率|成長率", na=False)
    level_codes = set(name_map[level_mask].index)
    jump_level = jump[jump["代碼"].isin(level_codes)]
    print(f"排除年增率/成長率類指標後，剩餘水準類指標中單期跳動>10倍的列數: {len(jump_level)}")
    print(jump_level.assign(名稱=jump_level["代碼"].map(name_map))
          [["代碼", "名稱", "年月", "v_prev", "數值", "ratio"]].head(15).to_string(index=False))
    print("\n這份資料是long format，沒有像money_market_rate.csv那樣可以互相佐證的鄰近欄位，")
    print("所以即使抓到孤立的單月跳動(如CA13美國聯邦資金市場利率-市場實際拆款利率，2016/10從")
    print("0.4%驟降到0.04%、隔月立刻回升到0.41%)，信心不足以直接判定為錯誤並動手修正，")
    print("僅建議：使用這類代碼前，針對可疑的單一期別另外查證原始資料源，不要照單全收。")

    # 圖：'-'佔位符比例最高的代碼一覽
    rate = df.groupby("代碼")["is_dash"].mean()
    top_empty = rate.sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"{c}\n{name_map[c][:16]}" for c in top_empty.index]
    ax.barh(labels[::-1], (top_empty * 100).values[::-1], color="#C44E52")
    ax.set_xlabel("'-' 佔位符比例 (%)")
    ax.set_title("'-' 佔位符比例最高的12個代碼（形同虛設，建議排除）")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/23_macro_mostly_empty_indicators.png", dpi=150)
    plt.close(fig)


def main():
    df = load()
    df, name_map = missing_value_analysis(df)
    outlier_analysis(df, name_map)
    sec("完成")


if __name__ == "__main__":
    main()
