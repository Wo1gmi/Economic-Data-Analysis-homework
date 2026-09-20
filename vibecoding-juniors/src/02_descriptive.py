import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed" / "analysis.parquet"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"


def table_t1_sample(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("year")
    t1 = pd.DataFrame(
        {
            "n_vacancies": g.size(),
            "n_employers": g["employer_id"].nunique(),
            "n_roles": g["professional_roles_clean"].nunique(),
            "n_regions": g["region"].nunique(),
            "share_moscow": g.apply(lambda x: (x["region"] == "Москва").mean(), include_groups=False),
        }
    )
    return t1


def table_t2_jun_definitions(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"definition": "J1 (noExperience + 1-3 года)", "n": int(df["jun_j1"].sum()),
         "share_of_sample": df["jun_j1"].mean()},
        {"definition": "J2 (только noExperience)", "n": int(df["jun_j2"].sum()),
         "share_of_sample": df["jun_j2"].mean()},
        {"definition": "J3 (junior/младший/стажёр в названии)", "n": int(df["jun_j3"].sum()),
         "share_of_sample": df["jun_j3"].mean()},
    ]
    t2 = pd.DataFrame(rows)
    return t2


def table_t2b_overlap(df: pd.DataFrame) -> pd.DataFrame:
    j2 = df["jun_j2"]
    between = df["experience"] == "between1And3"
    return pd.DataFrame(
        {
            "group": ["J2 (noExperience)", "between1And3 (не входит в J2)"],
            "n": [int(j2.sum()), int(between.sum())],
            "share_with_jun_title_j3": [df.loc[j2, "jun_j3"].mean(), df.loc[between, "jun_j3"].mean()],
        }
    )


def figure_r1(df: pd.DataFrame) -> None:
    m = df.groupby("published_month")[["jun_j1", "jun_j2", "jun_j3"]].mean().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(m.index, m["jun_j1"], label="J1: без опыта + 1–3 года (основное)", linewidth=2)
    ax.plot(m.index, m["jun_j2"], label="J2: только без опыта", linewidth=1.5, linestyle="--")
    ax.plot(m.index, m["jun_j3"], label="J3: junior/младший в названии", linewidth=1.5, linestyle=":")
    ax.axvline(cfg.CUTOFF_MONTH, color="gray", linestyle="-", alpha=0.5, linewidth=1)
    ax.text(cfg.CUTOFF_MONTH, ax.get_ylim()[1] * 0.02, " отсечка 2025-02", fontsize=8, color="gray")
    ax.set_ylabel("Доля джуновских вакансий среди всех вакансий месяца")
    ax.set_xlabel("Месяц публикации")
    ax.set_title("Р1. Доля джуновских вакансий по месяцам, три определения")
    ax.set_xticks(m.index[::3])
    ax.set_xticklabels(m.index[::3], rotation=90, fontsize=7)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "R1_jun_share_by_month.png", dpi=150)
    plt.close(fig)


def table_t3_shift_share(df: pd.DataFrame) -> dict:
    role = df["professional_roles_clean"]
    period = df["post"].map({False: "before", True: "after"})

    counts = pd.crosstab(role, period)
    jun_counts = pd.crosstab(role, period, values=df["jun_j1"], aggfunc="sum")

    w = counts.div(counts.sum(axis=0), axis=1)
    y = jun_counts / counts

    w_bar = (w["before"] + w["after"]) / 2
    y_bar = (y["before"] + y["after"]) / 2
    dY = y["after"] - y["before"]
    dW = w["after"] - w["before"]

    within = (w_bar * dY).sum()
    between = (y_bar * dW).sum()

    y_total_before = df.loc[~df["post"], "jun_j1"].mean()
    y_total_after = df.loc[df["post"], "jun_j1"].mean()
    total_change = y_total_after - y_total_before

    result = {
        "y_before": y_total_before,
        "y_after": y_total_after,
        "total_change": total_change,
        "within_role_component": within,
        "between_role_component": between,
        "sum_components": within + between,
        "residual": total_change - (within + between),
    }
    return result


def table_t4_by_role_year(df: pd.DataFrame) -> pd.DataFrame:
    share = df.groupby(["professional_roles_clean", "year"])["jun_j1"].mean().unstack("year")
    n = df.groupby(["professional_roles_clean", "year"]).size().unstack("year")
    share.columns = [f"share_{c}" for c in share.columns]
    n.columns = [f"n_{c}" for c in n.columns]
    return pd.concat([share, n], axis=1).sort_index()


def figure_r2(df: pd.DataFrame) -> pd.Series:
    m = df.groupby("published_month")["ai_mention"].mean().sort_index()
    m.rename("ai_mention_share").to_csv(TABLES / "R2_ai_mention_by_month.csv")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(m.index, m.values, color="darkred", linewidth=2)
    ax.axvline(cfg.CUTOFF_MONTH, color="gray", linestyle="-", alpha=0.5, linewidth=1)
    ax.text(cfg.CUTOFF_MONTH, ax.get_ylim()[1] * 0.9, " отсечка 2025-02", fontsize=8, color="gray")
    ax.set_ylabel("Доля вакансий с упоминанием ИИ-инструментов")
    ax.set_xlabel("Месяц публикации")
    ax.set_title("Р2. Доля вакансий с упоминанием ИИ-инструментов по месяцам")
    ax.set_xticks(m.index[::3])
    ax.set_xticklabels(m.index[::3], rotation=90, fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "R2_ai_mention_by_month.png", dpi=150)
    plt.close(fig)
    return m


def main() -> None:
    t0 = time.time()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED)

    t1 = table_t1_sample(df)
    t1.to_csv(TABLES / "T1_sample.csv")
    print("Т1 готова:\n", t1, "\n")

    t2 = table_t2_jun_definitions(df)
    t2.to_csv(TABLES / "T2_jun_definitions.csv", index=False)
    t2b = table_t2b_overlap(df)
    t2b.to_csv(TABLES / "T2b_jun_overlap.csv", index=False)
    print("Т2 готова:\n", t2, "\n")
    print("Т2b (пересечения):\n", t2b, "\n")

    figure_r1(df)
    print("Р1 сохранён\n")

    ss = table_t3_shift_share(df)
    pd.Series(ss).to_csv(TABLES / "T3_shift_share.csv", header=["value"])
    print("Т3 (shift-share):")
    for k, v in ss.items():
        print(f"  {k}: {v:.5f}")
    assert abs(ss["residual"]) < 1e-9, "Shift-share не сходится!"
    print("  -> сумма компонент сходится с наблюдаемым изменением (проверено).\n")

    t4 = table_t4_by_role_year(df)
    t4.to_csv(TABLES / "T4_jun_by_role_year.csv")
    print("Т4 сохранена, roles x years:", t4.shape, "\n")

    figure_r2(df)
    ai_by_year = df.groupby(df["published_month"].str[:4])["ai_mention"].mean()
    print("Р2 сохранён (график + output/tables/R2_ai_mention_by_month.csv)")
    print("Доля упоминаний ИИ по годам (сверка с разделом 1.4 записки):")
    print((ai_by_year * 100).round(2).astype(str) + "%")
    print()

    print(f"Описательный слой (Т1-Т4, Р1-Р2) завершён за {time.time() - t0:.1f} с.")


if __name__ == "__main__":
    main()
