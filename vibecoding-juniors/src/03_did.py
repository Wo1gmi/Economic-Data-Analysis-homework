import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed" / "analysis.parquet"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"


def manual_did(df: pd.DataFrame, jun_col: str) -> dict:
    m = df.groupby(["role_group", "post"])[jun_col].mean().unstack("post")
    m.columns = ["before", "after"]
    code_before, code_after = m.loc["code", "before"], m.loc["code", "after"]
    noncode_before, noncode_after = m.loc["noncode", "before"], m.loc["noncode", "after"]
    code_change = code_after - code_before
    noncode_change = noncode_after - noncode_before
    did = code_change - noncode_change
    return {
        "code_before": code_before,
        "code_after": code_after,
        "code_change": code_change,
        "noncode_before": noncode_before,
        "noncode_after": noncode_after,
        "noncode_change": noncode_change,
        "manual_did": did,
    }


def regression_no_fe(df: pd.DataFrame, jun_col: str):
    d = df.copy()
    d["code"] = (d["role_group"] == "code").astype(int)
    d["post_i"] = d["post"].astype(int)
    d["y"] = d[jun_col].astype(int)
    model = smf.ols("y ~ code * post_i", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["employer_id"]}
    )
    return model


def regression_fe(df: pd.DataFrame, jun_col: str):
    d = df.copy()
    d["code"] = (d["role_group"] == "code").astype(int)
    d["post_i"] = d["post"].astype(int)
    d["y"] = d[jun_col].astype(int)
    model = smf.ols(
        "y ~ code:post_i + C(professional_roles_clean) + C(published_month)",
        data=d,
    ).fit(cov_type="cluster", cov_kwds={"groups": d["employer_id"]})
    return model


def regression_fe_continuous(df: pd.DataFrame, jun_col: str):
    d = df.copy()
    d["post_i"] = d["post"].astype(int)
    d["y"] = d[jun_col].astype(int)
    d["exposure_pp"] = d["role_ai_exposure"] * 100
    model = smf.ols(
        "y ~ exposure_pp:post_i + C(professional_roles_clean) + C(published_month)",
        data=d,
    ).fit(cov_type="cluster", cov_kwds={"groups": d["employer_id"]})
    return model


def extract_coef(model, name: str) -> dict:
    if name not in model.params.index:
        raise KeyError(f"Не найден коэффициент '{name}' среди {list(model.params.index)}")
    p = name
    ci = model.conf_int().loc[p]
    return {
        "term": p,
        "coef": model.params[p],
        "se": model.bse[p],
        "p_value": model.pvalues[p],
        "ci_low": ci[0],
        "ci_high": ci[1],
        "n_obs": int(model.nobs),
    }


def run_all_specs(df: pd.DataFrame, jun_col: str, label: str) -> list:
    sub = df[df["role_group"].isin(["code", "noncode"])].copy()
    sub = sub[sub["employer_id_valid"]]

    rows = []

    md = manual_did(sub, jun_col)
    rows.append({
        "definition": label, "spec": "manual_4means",
        "coef": md["manual_did"], "se": np.nan, "p_value": np.nan,
        "ci_low": np.nan, "ci_high": np.nan, "n_obs": len(sub),
        "note": f"code: {md['code_before']:.3f}->{md['code_after']:.3f} "
                f"({md['code_change']:+.3f}); noncode: {md['noncode_before']:.3f}->"
                f"{md['noncode_after']:.3f} ({md['noncode_change']:+.3f})",
    })

    m_no_fe = regression_no_fe(sub, jun_col)
    c = extract_coef(m_no_fe, "code:post_i")
    rows.append({"definition": label, "spec": "ols_no_fe", **c,
                  "note": "должен совпасть с manual_4means"})

    match_ok = abs(rows[-1]["coef"] - md["manual_did"]) < 1e-9
    rows[-1]["note"] += f"; совпадение: {match_ok}"

    m_fe = regression_fe(sub, jun_col)
    c = extract_coef(m_fe, "code:post_i")
    rows.append({"definition": label, "spec": "ols_fe_role_month", **c,
                  "note": "FE роли+месяца, кластеры по работодателю"})

    m_cont = regression_fe_continuous(sub, jun_col)
    c = extract_coef(m_cont, "exposure_pp:post_i")
    rows.append({"definition": label, "spec": "ols_fe_continuous_exposure", **c,
                  "note": "коэфф. на 1 п.п. экспозиции роли к ИИ"})

    return rows


def figure_r3(df: pd.DataFrame) -> None:
    sub = df[df["role_group"].isin(["code", "noncode", "analytics"])]
    q = sub.groupby(["quarter", "role_group"])["jun_j1"].mean().unstack("role_group").sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(q.index, q["code"], label="Код (разраб., тестировщик, DevOps, тимлид разработки)",
            linewidth=2, marker="o", markersize=3)
    ax.plot(q.index, q["noncode"], label="Не-код (техподдержка, сисадмин, дизайн, ИБ и др.)",
            linewidth=2, marker="s", markersize=3)
    ax.plot(q.index, q["analytics"], label="Аналитика/DS (вне основной спецификации)",
            linewidth=1.5, linestyle="--", color="gray")
    ax.set_ylabel("Доля джуновских вакансий (J1) в квартале")
    ax.set_xlabel("Квартал")
    ax.set_title("Р3. Доля джуновских вакансий: код-роли против не-код, по кварталам")
    ax.set_xticks(range(len(q.index)))
    ax.set_xticklabels(q.index, rotation=90, fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "R3_jun_share_code_vs_noncode.png", dpi=150)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    df = pd.read_parquet(PROCESSED)

    all_rows = []
    for jun_col, label in [("jun_j1", "J1"), ("jun_j2", "J2"), ("jun_j3", "J3")]:
        print(f"=== {label} ({jun_col}) ===")
        rows = run_all_specs(df, jun_col, label)
        for r in rows:
            se_str = "n/a" if pd.isna(r["se"]) else f"{r['se']:.4f}"
            print(f"  {r['spec']:28s} coef={r['coef']:+.4f}  se={se_str}  n={r['n_obs']}")
        all_rows.extend(rows)

    t5 = pd.DataFrame(all_rows)
    t5.to_csv(TABLES / "T5_did_summary.csv", index=False)
    print(f"\nТ5 сохранена: output/tables/T5_did_summary.csv")

    figure_r3(df)
    print("Р3 сохранён")

    print(f"\nОсновная DiD-модель (Т5, Р3) завершена за {time.time() - t0:.1f} с.")


if __name__ == "__main__":
    main()
