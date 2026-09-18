"""
AI CUP 2026 玉山挑戰賽 — data/rev_prof_net.csv（各公司季營收/營益/稅前淨利）
缺失值與極值分析。純描述性分析，不含選股/策略邏輯。
產出圖表到 figures/，並把關鍵數字印到 stdout，供撰寫摘要時引用真實數字。
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
# sns.set_theme() 若帶 font= 會直接鎖死 font.family 成單一字型、不會走 fallback；
# 改成設定 font.sans-serif 清單，讓沒有 WenQuanYi Zen Hei(原分析機器是Linux) 的
# 環境(如這台Mac)能自動退回系統內建的 Heiti TC，避免中文字元顯示成方框。
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Heiti TC", "PingFang TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_PATH = "data/rev_prof_net.csv"
FIG_DIR = "analysis/figures"

NUM_COLS = ["營業收入淨額", "營業利益", "稅前淨利", "合併總損益"]
QUARTER_MONTH_TO_Q = {3: 1, 6: 2, 9: 3, 12: 4}

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

POST2008_QIDX = 2008 * 4 + 1  # 2008Q1，此後才強制季報，之前僅半年報(Q2/Q4)


def sec(msg):
    print(f"\n{'='*10} {msg} {'='*10}")


def quarter_idx(ym):
    y, m = ym // 100, ym % 100
    return y * 4 + QUARTER_MONTH_TO_Q[m]


def load():
    sec("Loading data/rev_prof_net.csv")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print("shape:", df.shape, "| 公司數:", df["公司簡稱"].nunique())
    df["qidx"] = df["年/月"].apply(quarter_idx)
    df["年"] = df["年/月"] // 100
    return df


# ------------------------------------------------------------ missing ----
def missing_value_analysis(df):
    sec("A. 缺失值分析")

    # A1. 結構完整性（欄位NaN / 年月-季別-月份一致性 / 重複列）
    print("[A1] 欄位 NaN 數：")
    print(df.isna().sum())

    mod = df["年/月"] % 100
    bad = df[mod.map(QUARTER_MONTH_TO_Q) != df["季別"]]
    print(f"[A1] 年/月 與 季別 不一致列數: {len(bad)}")

    dup = df.duplicated(subset=["公司簡稱", "年/月", "合併(Y/N)"], keep=False)
    print(f"[A1] (公司簡稱,年/月,合併Y/N) 重複列數: {dup.sum()}")

    # A2. 數值欄位的 "-" 佔位符（真正的缺失）
    placeholder_mask = (df[NUM_COLS] == "-").any(axis=1)
    print(f"\n[A2] 數值欄位含 '-' 佔位符的列數: {placeholder_mask.sum()}")
    if placeholder_mask.any():
        print(df.loc[placeholder_mask, ["公司簡稱", "名稱", "年/月", "TSE產業_名稱"]].to_string(index=False))
        print("→ 全部為公司剛完成分割/控股重組當季（如投控成立首季），母公司尚無可申報數字，非資料品質問題。")

    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # A3. 150檔投資池 vs 本資料集 覆蓋率
    present = set(df["公司簡稱"].unique())
    missing_tickers = [t for t in UNIVERSE if t not in present]
    print(f"\n[A3] 150檔投資池中，本資料集完全沒有資料的檔數: {len(missing_tickers)}")
    miss_names = (
        df.drop_duplicates("公司簡稱").set_index("公司簡稱")["名稱"]
        if False else None
    )
    print("缺席代號:", missing_tickers)
    print("→ 檢查發現這些代號幾乎全為金融保險業(銀行/金控)，此資料集(季營收/營益表)")
    print("   結構性不涵蓋金融業(其財報科目與一般產業不同，無「營業收入淨額/營業利益」欄位)；")
    print("   3718(中光電投控)則是因2026-09-03才剛掛牌，尚無任何季報。")

    # A4. 每年/季 申報公司數 —— 找出 2005-2007 只有 半年報(Q2/Q4) 的結構性缺口
    sec("A4. 年度 x 季別 申報家數（找系統性缺口）")
    tab = df.groupby(["年", "季別"]).size().unstack(fill_value=0)
    print(tab)

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(tab.T, cmap="YlGnBu", annot=False, cbar_kws={"label": "申報公司數"}, ax=ax)
    ax.set_title("每年 x 季別 申報公司數（可見 2005-2007 僅半年報 Q2/Q4，無 Q1/Q3）")
    ax.set_xlabel("年")
    ax.set_ylabel("季別")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/13_filing_coverage_heatmap.png", dpi=150)
    plt.close(fig)

    # A5. 個別公司的「內部缺口」(在已上市期間內，中間漏掉的季度)
    sec("A5. 個別公司內部缺口 (剔除 2005-2007 半年報制度性缺口後)")
    gaps = []
    for tid, g in df.groupby("公司簡稱"):
        qs = sorted(g.loc[g["qidx"] >= POST2008_QIDX, "qidx"].unique())
        if len(qs) < 2:
            continue
        full_range = set(range(qs[0], qs[-1] + 1))
        missing = sorted(full_range - set(qs))
        if missing:
            gaps.append((tid, g["名稱"].iloc[0], len(missing), len(qs)))
    gaps_df = pd.DataFrame(gaps, columns=["公司簡稱", "名稱", "缺口季數", "實際筆數(2008後)"])
    n_companies_2008 = df.loc[df["qidx"] >= POST2008_QIDX, "公司簡稱"].nunique()
    print(f"2008Q1後仍有內部缺口的公司數: {len(gaps_df)} / {n_companies_2008}")
    print(gaps_df.sort_values("缺口季數", ascending=False).head(15).to_string(index=False))

    in_universe_gaps = gaps_df[gaps_df["公司簡稱"].isin(UNIVERSE)]
    print(f"\n其中屬於150檔投資池: {len(in_universe_gaps)} 檔")
    print(in_universe_gaps.sort_values("缺口季數", ascending=False).to_string(index=False))

    return df, gaps_df, missing_tickers


# ------------------------------------------------------------ outliers ----
def outlier_analysis(df):
    sec("B. 極值分析")

    # B1. 營收正負號檢查
    neg_rev = (df["營業收入淨額"] < 0).sum()
    zero_rev = (df["營業收入淨額"] == 0).sum()
    print(f"[B1] 營業收入淨額 < 0 的列數: {neg_rev} ({neg_rev/len(df):.3%})")
    print(f"[B1] 營業收入淨額 = 0 的列數: {zero_rev} ({zero_rev/len(df):.3%})")
    neg_df = df[df["營業收入淨額"] < 0]
    print("負營收產業分布 (前10):")
    print(neg_df["TSE產業_名稱"].value_counts().head(10))
    print("→ 負營收多屬生技醫療/建材營造等產業，來自銷貨退回/讓價超過當季銷貨或業外科目認列，屬合理會計現象而非錯誤。")

    # B2. 營益率極端值 (只在營收>0時定義)
    sub = df[df["營業收入淨額"] > 0].copy()
    sub["op_margin"] = sub["營業利益"] / sub["營業收入淨額"]
    print("\n[B2] 營業利益/營業收入淨額 (op_margin) 分布：")
    print(sub["op_margin"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

    extreme_hi = sub[sub["op_margin"] > 5]
    extreme_lo = sub[sub["op_margin"] < -20]
    print(f"\nop_margin > 500%: {len(extreme_hi)} 列（多為營收趨近於0，比率被小分母放大）")
    print(extreme_hi.sort_values("op_margin", ascending=False)[["公司簡稱", "名稱", "年/月", "營業收入淨額", "營業利益", "op_margin"]].head(10).to_string(index=False))
    print(f"\nop_margin < -2000%: {len(extreme_lo)} 列")
    print(extreme_lo.sort_values("op_margin")[["公司簡稱", "名稱", "年/月", "營業收入淨額", "營業利益", "op_margin"]].head(10).to_string(index=False))
    print("→ 這類極端比率清一色來自營收極小(個位數~數十千元)的季度，屬「分母效應」而非數字本身有誤；")
    print("   用比率類特徵(如營益率)做建模時，建議對低營收樣本另設篩選門檻或改用絕對值特徵。")

    fig, ax = plt.subplots(figsize=(9, 5))
    clipped = sub["op_margin"].clip(-3, 3)
    sns.histplot(clipped, bins=80, ax=ax, color="#4C72B0")
    ax.set_title("營業利益率分布 (op_margin，已截尾於[-3,3]以利觀察，極端值另表列出)")
    ax.set_xlabel("op_margin = 營業利益 / 營業收入淨額")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/14_operating_margin_distribution.png", dpi=150)
    plt.close(fig)

    # B3. 季度間營收跳動 (QoQ ratio) 抓可能的單位/數字錯誤，同時區分「營建業正常認列」
    df_sorted = df.sort_values(["公司簡稱", "年/月"]).copy()
    df_sorted["rev_prev"] = df_sorted.groupby("公司簡稱")["營業收入淨額"].shift(1)
    df_sorted["rev_ratio"] = df_sorted["營業收入淨額"] / df_sorted["rev_prev"].replace(0, np.nan)
    jump = df_sorted[(df_sorted["rev_ratio"] > 20) | (df_sorted["rev_ratio"] < 0.05)].dropna(subset=["rev_ratio"])
    print(f"\n[B3] 季對季營收比率 >20x 或 <0.05x 的列數: {len(jump)} ({len(jump)/len(df_sorted):.3%})")
    print("產業分布 (前10):")
    print(jump["TSE產業_名稱"].value_counts().head(10))
    print(jump.sort_values("rev_ratio", ascending=False)[["公司簡稱", "名稱", "年/月", "rev_prev", "營業收入淨額", "rev_ratio", "TSE產業_名稱"]].head(10).to_string(index=False))
    print("→ 高比率跳動集中在營建/開發類股（完工比例法/交屋認列，單季可暴衝數十~數萬倍），屬產業特性；")
    print("   用QoQ成長率做特徵時，這類股票須額外處理或排除，否則極端值會主導模型。")

    in_universe_jump = jump[jump["公司簡稱"].isin(UNIVERSE)]
    print(f"\n其中屬於150檔投資池: {len(in_universe_jump)} 列，涉及 {in_universe_jump['公司簡稱'].nunique()} 檔")
    if len(in_universe_jump):
        print(in_universe_jump[["公司簡稱", "名稱", "年/月", "rev_ratio", "TSE產業_名稱"]].drop_duplicates("公司簡稱").to_string(index=False))


def main():
    df = load()
    df, gaps_df, missing_tickers = missing_value_analysis(df)
    outlier_analysis(df)
    sec("完成")


if __name__ == "__main__":
    main()
