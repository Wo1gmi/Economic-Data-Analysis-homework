import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed" / "analysis.parquet"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"

JUN_COL = "jun_j1"


def base_sample(df: pd.DataFrame, code_roles=None, noncode_roles=None) -> pd.DataFrame:
    if code_roles is None:
        code_roles = cfg.CODE_ROLES
    if noncode_roles is None:
        noncode_roles = cfg.NONCODE_ROLES
    d = df[df["professional_roles_clean"].isin(code_roles | noncode_roles)].copy()
    d = d[d["employer_id_valid"]]
    d["code"] = d["professional_roles_clean"].isin(code_roles).astype(int)
    return d


def fit_did_fe(d: pd.DataFrame, jun_col: str, post_col: str = "post") -> dict:
    dd = d.copy()
    dd["y"] = dd[jun_col].astype(int)
    dd["post_i"] = dd[post_col].astype(int)
    model = smf.ols(
        "y ~ code:post_i + C(professional_roles_clean) + C(published_month)",
        data=dd,
    ).fit(cov_type="cluster", cov_kwds={"groups": dd["employer_id"]})
    p = "code:post_i"
    ci = model.conf_int().loc[p]
    return {
        "coef": model.params[p], "se": model.bse[p], "p_value": model.pvalues[p],
        "ci_low": ci[0], "ci_high": ci[1], "n_obs": int(model.nobs),
    }


def event_study(df: pd.DataFrame) -> pd.DataFrame:
    import re as _re

    d = base_sample(df)
    d["y"] = d[JUN_COL].astype(int)
    base = cfg.EVENT_STUDY_BASE_MONTH
    d["month_cat"] = pd.Categorical(d["published_month"])
    model = smf.ols(
        "y ~ code:C(month_cat) + C(professional_roles_clean) + C(published_month)",
        data=d,
    ).fit(cov_type="cluster", cov_kwds={"groups": d["employer_id"]})

    month_params = {}
    for p in model.params.index:
        if p.startswith("code:C(month_cat"):
            month = _re.search(r"\[(?:T\.)?([^\]]+)\]$", p).group(1)
            month_params[month] = p

    cov = model.cov_params()
    base_term = month_params[base]
    beta_base = model.params[base_term]
    var_base = cov.loc[base_term, base_term]

    rows = []
    for month, term in sorted(month_params.items()):
        beta = model.params[term] - beta_base
        var = cov.loc[term, term] + var_base - 2 * cov.loc[term, base_term]
        se = np.sqrt(max(var, 0.0))
        rows.append({
            "month": month, "coef": beta, "se": se,
            "ci_low": beta - 1.96 * se, "ci_high": beta + 1.96 * se,
        })
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def figure_r4(event_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.errorbar(
        event_df["month"], event_df["coef"],
        yerr=[event_df["coef"] - event_df["ci_low"], event_df["ci_high"] - event_df["coef"]],
        fmt="o", markersize=3, capsize=2, linewidth=1, color="steelblue",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(cfg.CUTOFF_MONTH, color="gray", linestyle="--", alpha=0.6, linewidth=1)
    ax.text(cfg.CUTOFF_MONTH, ax.get_ylim()[1] * 0.9, " отсечка 2025-02", fontsize=8, color="gray")
    ax.set_ylabel("Коэффициент code x month (база 2025-01), п.п. доли джунов")
    ax.set_xlabel("Месяц")
    ax.set_title("Р4. Событийное исследование: код-роли относительно не-код, по месяцам")
    ax.set_xticks(event_df["month"][::3])
    ax.set_xticklabels(event_df["month"][::3], rotation=90, fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "R4_event_study.png", dpi=150)
    plt.close(fig)


def band_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    d = base_sample(df)
    d["between_1_3"] = (d["experience"] == "between1And3").astype(int)
    rows = []
    for col, label in [("jun_j2", "только noExperience (J2)"),
                        ("between_1_3", "только between1And3 (J1 минус J2)")]:
        r = fit_did_fe(d, col)
        rows.append({"check": f"декомпозиция J1 по сегменту опыта: {label}", **r})
    return pd.DataFrame(rows)


def pretrend_checks(df: pd.DataFrame) -> pd.DataFrame:
    d = base_sample(df)
    first_month = sorted(d["published_month"].unique())[0]
    d["t"] = (
        pd.PeriodIndex(d["published_month"], freq="M")
        - pd.Period(first_month, freq="M")
    ).map(lambda x: x.n)
    d["y"] = d[JUN_COL].astype(int)
    d["post_i"] = d["post"].astype(int)

    rows = []

    pre = d[d["published_month"] < cfg.CUTOFF_MONTH]
    m1 = smf.ols(
        "y ~ code:t + C(professional_roles_clean) + C(published_month)",
        data=pre,
    ).fit(cov_type="cluster", cov_kwds={"groups": pre["employer_id"]})
    p1 = "code:t"
    ci1 = m1.conf_int().loc[p1]
    rows.append({
        "check": f"наклон разрыва код/не-код в допериоде (до {cfg.CUTOFF_MONTH}), п.п./мес",
        "coef": m1.params[p1], "se": m1.bse[p1], "p_value": m1.pvalues[p1],
        "ci_low": ci1[0], "ci_high": ci1[1], "n_obs": int(m1.nobs),
    })

    m2 = smf.ols(
        "y ~ code:post_i + code:t + C(professional_roles_clean) + C(published_month)",
        data=d,
    ).fit(cov_type="cluster", cov_kwds={"groups": d["employer_id"]})
    p2 = "code:post_i"
    ci2 = m2.conf_int().loc[p2]
    rows.append({
        "check": "основная спецификация + групповой линейный тренд code:t",
        "coef": m2.params[p2], "se": m2.bse[p2], "p_value": m2.pvalues[p2],
        "ci_low": ci2[0], "ci_high": ci2[1], "n_obs": int(m2.nobs),
    })

    for start, label in [("2024-01", "13 мес."), ("2024-07", "7 мес.")]:
        s = d[d["published_month"] >= start]

        m_notrend = smf.ols(
            "y ~ code:post_i + C(professional_roles_clean) + C(published_month)",
            data=s,
        ).fit(cov_type="cluster", cov_kwds={"groups": s["employer_id"]})
        p0 = "code:post_i"
        ci0 = m_notrend.conf_int().loc[p0]
        rows.append({
            "check": f"без тренда, допериод от {start} ({label})",
            "coef": m_notrend.params[p0], "se": m_notrend.bse[p0],
            "p_value": m_notrend.pvalues[p0],
            "ci_low": ci0[0], "ci_high": ci0[1], "n_obs": int(m_notrend.nobs),
        })

        m_trend = smf.ols(
            "y ~ code:post_i + code:t + C(professional_roles_clean) + C(published_month)",
            data=s,
        ).fit(cov_type="cluster", cov_kwds={"groups": s["employer_id"]})
        p1 = "code:post_i"
        ci1 = m_trend.conf_int().loc[p1]
        rows.append({
            "check": f"+ групповой тренд code:t, допериод от {start} ({label})",
            "coef": m_trend.params[p1], "se": m_trend.bse[p1],
            "p_value": m_trend.pvalues[p1],
            "ci_low": ci1[0], "ci_high": ci1[1], "n_obs": int(m_trend.nobs),
        })

    d["t2"] = d["t"] ** 2
    m_quad = smf.ols(
        "y ~ code:post_i + code:t + code:t2 + C(professional_roles_clean) + C(published_month)",
        data=d,
    ).fit(cov_type="cluster", cov_kwds={"groups": d["employer_id"]})
    p2 = "code:post_i"
    ci2b = m_quad.conf_int().loc[p2]
    rows.append({
        "check": "+ квадратичный тренд code:t + code:t^2, полная выборка",
        "coef": m_quad.params[p2], "se": m_quad.bse[p2], "p_value": m_quad.pvalues[p2],
        "ci_low": ci2b[0], "ci_high": ci2b[1], "n_obs": int(m_quad.nobs),
    })

    return pd.DataFrame(rows)


def robustness_checks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    d = base_sample(df)
    r = fit_did_fe(d, JUN_COL)
    rows.append({"check": "основная спецификация (2025-02, код vs не-код, вся выборка)", **r})

    d_pl = base_sample(df)
    d_pl = d_pl[d_pl["published_month"] < cfg.PLACEBO_SAMPLE_END]
    d_pl["post_placebo"] = d_pl["published_month"] >= cfg.PLACEBO_CUTOFF
    r = fit_did_fe(d_pl, JUN_COL, post_col="post_placebo")
    rows.append({"check": f"плацебо: дата {cfg.PLACEBO_CUTOFF}, выборка < {cfg.PLACEBO_SAMPLE_END}", **r})

    for cutoff in cfg.CUTOFF_SENSITIVITY:
        d_c = base_sample(df)
        d_c["post_alt"] = d_c["published_month"] >= cutoff
        r = fit_did_fe(d_c, JUN_COL, post_col="post_alt")
        rows.append({"check": f"дата отсечки = {cutoff}", **r})

    code_alt1 = cfg.CODE_ROLES - {"Тестировщик"}
    noncode_alt1 = cfg.NONCODE_ROLES | {"Тестировщик"}
    d1 = base_sample(df, code_alt1, noncode_alt1)
    r = fit_did_fe(d1, JUN_COL)
    rows.append({"check": "состав: тестировщики -> в группу не-код", **r})

    code_alt2 = cfg.CODE_ROLES | cfg.ANALYTICS_ROLES
    d2 = base_sample(df, code_alt2, cfg.NONCODE_ROLES)
    r = fit_did_fe(d2, JUN_COL)
    rows.append({"check": "состав: аналитики/DS -> в группу код", **r})

    cis_non_rf = {"Алматы", "Минск", "Ташкент", "Астана"}
    d_rf = base_sample(df)
    d_rf = d_rf[~d_rf["region"].isin(cis_non_rf)]
    r = fit_did_fe(d_rf, JUN_COL)
    rows.append({"check": "регионы: без Алматы/Минска/Ташкента/Астаны", **r})

    d_msk = base_sample(df)
    d_msk = d_msk[d_msk["region"].isin({"Москва", "Санкт-Петербург"})]
    r = fit_did_fe(d_msk, JUN_COL)
    rows.append({"check": "только Москва и Санкт-Петербург", **r})

    all_roles = sorted(cfg.CODE_ROLES | cfg.NONCODE_ROLES)
    for role in all_roles:
        code_ex = cfg.CODE_ROLES - {role}
        noncode_ex = cfg.NONCODE_ROLES - {role}
        d_ex = base_sample(df, code_ex, noncode_ex)
        r = fit_did_fe(d_ex, JUN_COL)
        rows.append({"check": f"без роли: {role}", **r})

    return pd.concat(
        [pd.DataFrame(rows), band_decomposition(df), pretrend_checks(df)],
        ignore_index=True,
    )


def table_t7_employer_fe(df: pd.DataFrame) -> dict:
    from patsy import dmatrices
    from scipy import stats as _stats

    d = base_sample(df)
    before_emp = set(d.loc[~d["post"], "employer_id"])
    after_emp = set(d.loc[d["post"], "employer_id"])
    panel_emp = before_emp & after_emp
    d_panel = d[d["employer_id"].isin(panel_emp)].copy()
    d_panel["y"] = d_panel[JUN_COL].astype(int)
    d_panel["post_i"] = d_panel["post"].astype(int)

    y_mat, x_mat = dmatrices(
        "y ~ code:post_i + C(professional_roles_clean) + C(published_month)",
        data=d_panel, return_type="dataframe",
    )
    emp = d_panel["employer_id"].reset_index(drop=True)
    y_mat = y_mat.reset_index(drop=True)
    x_mat = x_mat.reset_index(drop=True)

    y_demeaned = y_mat["y"] - y_mat["y"].groupby(emp).transform("mean")
    x_demeaned = x_mat.drop(columns=["Intercept"]).apply(
        lambda col: col - col.groupby(emp).transform("mean")
    )

    import statsmodels.api as sm
    model = sm.OLS(y_demeaned, x_demeaned).fit(
        cov_type="cluster", cov_kwds={"groups": emp}
    )
    matches = [p for p in model.params.index if p == "code:post_i"]
    p = matches[0]

    n_obs = int(model.nobs)
    k_explicit = x_demeaned.shape[1]
    n_employers = len(panel_emp)
    df_correction = np.sqrt(
        (n_obs - k_explicit) / (n_obs - k_explicit - n_employers)
    )

    coef = model.params[p]
    se = model.bse[p] * df_correction
    z = coef / se
    p_value = 2 * (1 - _stats.norm.cdf(abs(z)))
    ci_low, ci_high = coef - 1.96 * se, coef + 1.96 * se

    return {
        "coef": coef, "se": se, "p_value": p_value,
        "ci_low": ci_low, "ci_high": ci_high, "n_obs": n_obs,
        "n_employers": n_employers,
        "se_uncorrected": model.bse[p],
    }


def table_t7b_ai_adopters(df: pd.DataFrame) -> pd.DataFrame:
    d = base_sample(df)
    before = d[~d["post"]].groupby("employer_id")["ai_mention"].mean()
    after = d[d["post"]].groupby("employer_id")["ai_mention"].mean()
    both = pd.DataFrame({"ai_before": before, "ai_after": after}).dropna()
    adopters = both[(both["ai_before"] == 0) & (both["ai_after"] > 0)].index
    non_adopters = both[(both["ai_before"] == 0) & (both["ai_after"] == 0)].index

    jun_before_after = d.groupby(["employer_id", "post"])[JUN_COL].mean().unstack("post")
    jun_before_after.columns = ["jun_before", "jun_after"]

    def summarize(idx, label):
        sub = jun_before_after.loc[jun_before_after.index.intersection(idx)]
        return {
            "group": label, "n_employers": len(sub),
            "jun_share_before": sub["jun_before"].mean(),
            "jun_share_after": sub["jun_after"].mean(),
            "change": sub["jun_after"].mean() - sub["jun_before"].mean(),
        }

    return pd.DataFrame([
        summarize(adopters, "начали упоминать ИИ после отсечки (не упоминали до)"),
        summarize(non_adopters, "никогда не упоминали ИИ"),
    ])


def main() -> None:
    t0 = time.time()
    df = pd.read_parquet(PROCESSED)

    print("Событийное исследование (Р4)...")
    ev = event_study(df)
    ev.to_csv(TABLES / "R4_event_study_coefficients.csv", index=False)
    figure_r4(ev)
    print("  Р4 сохранён\n")

    print("Проверки устойчивости (Т6)...")
    t6 = robustness_checks(df)
    t6.to_csv(TABLES / "T6_robustness.csv", index=False)
    print(t6[["check", "coef", "se", "p_value", "n_obs"]].to_string(index=False))
    placebo_row = t6[t6["check"].str.startswith("плацебо")].iloc[0]
    print(f"\n  Плацебо-оценка: {placebo_row['coef']:+.4f} (p={placebo_row['p_value']:.3f}) "
          f"{'-> неотличима от нуля' if placebo_row['p_value'] > 0.1 else '-> ВНИМАНИЕ: значима!'}")
    excl_rows = t6[t6["check"].str.startswith("без роли")]
    signs = np.sign(excl_rows["coef"])
    print(f"  Знак эффекта при исключении любой роли: "
          f"{'устойчив' if signs.nunique() == 1 else 'МЕНЯЕТСЯ, см. таблицу'}\n")

    trend_row = t6[t6["check"].str.startswith("основная спецификация + групповой")].iloc[0]
    print(f"  ВАЖНО: проверка на допериодный тренд: с групповым трендом code:t "
          f"коэффициент = {trend_row['coef']:+.4f} (p={trend_row['p_value']:.3f}). "
          f"{'Эффект НЕ выдерживает контроль на предтренд, см. записку, раздел 4.' if trend_row['p_value'] > 0.1 else 'Эффект выдерживает контроль на предтренд.'}\n")

    print("Т7: модель с FE работодателя...")
    t7 = table_t7_employer_fe(df)
    pd.Series(t7).to_csv(TABLES / "T7_employer_fe.csv", header=["value"])
    print(f"  coef={t7['coef']:+.4f} se={t7['se']:.4f} n_employers={t7['n_employers']} n_obs={t7['n_obs']}\n")

    print("Т7b: наблюдение по фирмам-'усыновителям' ИИ (описательное, не эффект)...")
    t7b = table_t7b_ai_adopters(df)
    t7b.to_csv(TABLES / "T7b_ai_adopters_observation.csv", index=False)
    print(t7b.to_string(index=False))

    print(f"\nПроверки идентификации и слой работодателя (Р4, Т6, Т7) завершены за {time.time() - t0:.1f} с.")


if __name__ == "__main__":
    main()
