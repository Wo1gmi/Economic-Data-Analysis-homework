import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "vacancies.parquet"
OUT_FILE = ROOT / "data" / "processed" / "analysis.parquet"


def normalize_roles(df: pd.DataFrame) -> pd.Series:
    roles = df["professional_roles"].copy()
    roles = roles.replace(
        {
            cfg.ORG_PLACEHOLDER_ROLE: cfg.ORG_PLACEHOLDER_ROLE_TARGET,
            cfg.ORG_PLACEHOLDER_ROLE_SUPPORT: cfg.ORG_PLACEHOLDER_ROLE_SUPPORT_TARGET,
        }
    )
    return roles


def build_jun_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["jun_j1"] = df["experience"].isin(cfg.JUN_J1_EXPERIENCE)
    out["jun_j2"] = df["experience"].isin(cfg.JUN_J2_EXPERIENCE)
    out["jun_j3"] = df["title"].str.contains(cfg.JUN_J3_TITLE_PATTERN, na=False)
    return out


def build_role_group(roles_clean: pd.Series) -> pd.Series:
    group = pd.Series(index=roles_clean.index, dtype="object")
    group[roles_clean.isin(cfg.CODE_ROLES)] = "code"
    group[roles_clean.isin(cfg.NONCODE_ROLES)] = "noncode"
    group[roles_clean.isin(cfg.ANALYTICS_ROLES)] = "analytics"
    group[roles_clean.isin(cfg.LEADERSHIP_ROLES)] = "leadership"
    if group.isnull().any():
        unmapped = roles_clean[group.isnull()].unique()
        raise ValueError(f"Роли без группы (обновите config.py): {unmapped}")
    return group


def build_ai_mention(df: pd.DataFrame) -> pd.Series:
    text = (
        df["title"].fillna("")
        + " "
        + df["description"].fillna("")
        + " "
        + df["skills"].fillna("")
    )
    return text.str.contains(cfg.AI_MENTION_PATTERN, na=False)


def build_role_exposure(df: pd.DataFrame, roles_clean: pd.Series) -> pd.Series:
    mask_period = df["published_month"] >= cfg.ROLE_EXPOSURE_PERIOD_START
    ai = build_ai_mention(df)
    exposure_by_role = (
        ai[mask_period].groupby(roles_clean[mask_period]).mean()
    )
    return roles_clean.map(exposure_by_role)


def main() -> None:
    t0 = time.time()
    df = pd.read_parquet(RAW_FILE)
    n_raw = len(df)
    if n_raw != cfg.RAW_N_ROWS:
        print(
            f"ВНИМАНИЕ: число строк в сыром файле ({n_raw}) не совпадает "
            f"с ожидаемым ({cfg.RAW_N_ROWS}). Возможно, источник обновился."
        )

    roles_clean = normalize_roles(df)
    df["professional_roles_clean"] = roles_clean
    df["role_group"] = build_role_group(roles_clean)

    jun = build_jun_flags(df)
    df = pd.concat([df, jun], axis=1)

    df["ai_mention"] = build_ai_mention(df)
    df["role_ai_exposure"] = build_role_exposure(df, roles_clean)

    df["post"] = df["published_month"] >= cfg.CUTOFF_MONTH
    df["year"] = df["published_month"].str.slice(0, 4)
    df["quarter"] = (
        df["year"]
        + "Q"
        + (((df["published_month"].str.slice(5, 7).astype(int) - 1) // 3) + 1).astype(str)
    )

    df["employer_id_valid"] = df["employer_id"].str.strip() != ""

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)

    elapsed = time.time() - t0
    print(f"Готово за {elapsed:.1f} с. Строк: {len(df)} (ожидалось {cfg.RAW_N_ROWS}).")
    rel_out = OUT_FILE.relative_to(ROOT)
    print(f"Сохранено: {rel_out} ({OUT_FILE.stat().st_size / 1e6:.1f} МБ)")

    assert len(df) == n_raw, "Число строк изменилось при обработке"
    assert not df["jun_j1"].isnull().any()
    assert not df["jun_j2"].isnull().any()
    assert not df["jun_j3"].isnull().any()
    assert not df["role_group"].isnull().any()
    assert not df["ai_mention"].isnull().any()
    print("Флаг jun_j1, N =", int(df["jun_j1"].sum()))
    print("Флаг jun_j2, N =", int(df["jun_j2"].sum()))
    print("Флаг jun_j3, N =", int(df["jun_j3"].sum()))
    print("\nКросс-таблица J1 x J3 (доли от J1):")
    print(pd.crosstab(df["jun_j1"], df["jun_j3"], normalize="index"))
    print("\nРоли без группы: (должно быть пусто)")
    print(df.loc[df["role_group"].isnull(), "professional_roles_clean"].unique())
    print("\nРаспределение по группам ролей:")
    print(df["role_group"].value_counts())

    code_noncode = df[df["role_group"].isin(["code", "noncode"])]
    n_excluded = (~code_noncode["employer_id_valid"]).sum()
    print(
        f"\nВ основной выборке (код+не-код) {len(code_noncode)} вакансий, из них "
        f"{n_excluded} с пустым employer_id; они исключаются из всех моделей с "
        f"кластеризацией/FE по работодателю (Т5-Т7), рабочая выборка моделей: "
        f"{len(code_noncode) - n_excluded}."
    )


if __name__ == "__main__":
    main()
