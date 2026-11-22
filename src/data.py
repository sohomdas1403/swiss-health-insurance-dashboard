"""
Data loading and cleaning for the Swiss health insurance deductible tool.

Sources (FOPH):
- Premiums: "Health insurance premiums", opendata.swiss, file Prämien_CH.xlsx (premium year 2027).
  Premiums in the file are MONTHLY, in CHF.
- Expenses: "Financial year 2025 - Gross costs per insured person by age, sex,
  hospitalisation and PCGs" (Table 3.6 Brutto, EFIND 2025).
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Cost brackets of the FOPH tables (gross costs per insured person, CHF per year)
# Each bracket is (lower bound, upper bound). The last bracket is open (> 50'000).
# ---------------------------------------------------------------------------
BRACKETS = [
    (0, 0),
    (1, 500),
    (501, 1_000),
    (1_001, 2_000),
    (2_001, 5_000),
    (5_001, 10_000),
    (10_001, 20_000),
    (20_001, 50_000),
    (50_001, np.inf),
]
BRACKET_LABELS = ["=0", "1-500", "501-1'000", "1'001-2'000", "2'001-5'000",
                  "5'001-10'000", "10'001-20'000", "20'001-50'000", ">50'000"]
P_COLS = [f"p{i}" for i in range(len(BRACKETS))]   # share of insured in each bracket

DEDUCTIBLES_ADULT = [300, 500, 1000, 1500, 2000, 2500]

AGE_CLASS_PREMIUM = {"AKA_01_KIN": "child", "AKA_02_JUG": "young_adult", "AKA_03_ERW": "adult"}
ACCIDENT = {"MIT_UNF": True, "OHN_UNF": False}

# The 26 Swiss cantons. The premium file also contains two other codes (ZE, ZR) with
# unusually low premiums (about CHF 80-105 per month), which are not cantons and are
# not documented in the FOPH explanation file: they are excluded.
CANTONS = ["AG", "AI", "AR", "BE", "BL", "BS", "FR", "GE", "GL", "GR", "JU", "LU", "NE",
           "NW", "OW", "SG", "SH", "SO", "SZ", "TG", "TI", "UR", "VD", "VS", "ZG", "ZH"]


# ---------------------------------------------------------------------------
# Premiums
# ---------------------------------------------------------------------------
def load_premiums(path, adults_only=True):
    """Read Prämien_CH.xlsx and return a tidy table of ANNUAL premiums.

    One row = one insurer x canton x region x age class x accident cover x tariff x deductible.
    """
    raw = pd.read_excel(path, sheet_name=0)
    df = pd.DataFrame({
        "year": raw["Geschäftsjahr"],
        "insurer_id": raw["Versicherer"],
        "canton": raw["Kanton"],
        "region": raw["Region"],
        "age_class": raw["Altersklasse"].map(AGE_CLASS_PREMIUM),
        "accident": raw["Unfalleinschluss"].map(ACCIDENT),
        "model": raw["Tariftyp"],               # BASE, PRAXIS (family doctor / HMO), TEL_DIG, ...
        "tariff_id": raw["Tarif"],
        "tariff_name": raw["Tarifbezeichnung"],
        "deductible": raw["Franchise"].str[-4:].astype(int),   # e.g. FRA_01_E_0300 -> 300
        "premium_month": raw["Prämie"].astype(float),
    })
    df["premium_year"] = (12 * df["premium_month"]).round(2)
    df = df[df["canton"].isin(CANTONS)]          # drop the non-canton codes ZE and ZR
    if adults_only:
        df = df[df["age_class"].isin(["adult", "young_adult"])]
    return df.reset_index(drop=True)


def premium_schedule(premiums, canton, region, age_class, accident, model="BASE"):
    """Return a table (one row per insurer/tariff, one column per deductible) of annual premiums.

    Only complete schedules (all 6 adult deductibles available) are kept.
    """
    sel = premiums[(premiums.canton == canton) & (premiums.region == region)
                   & (premiums.age_class == age_class) & (premiums.accident == accident)
                   & (premiums.model == model)]
    table = sel.pivot_table(index=["insurer_id", "tariff_id", "tariff_name"],
                            columns="deductible", values="premium_year", aggfunc="first")
    table = table.reindex(columns=DEDUCTIBLES_ADULT)
    return table.dropna()   # drops e.g. an insurer that does not offer the 2'000 deductible


# ---------------------------------------------------------------------------
# Expenses (FOPH table 3.6, "Brutto")
# ---------------------------------------------------------------------------
def load_expenses(path, sheet="Français", small_n=1_000):
    """Read the FOPH gross-cost table 3.6 and return a tidy table.

    Columns: age, sex, hospitalised, pcg (if present), n, mean_cost, sd, p0..p8, plus
    quality flags. The shares p0..p8 are fractions (0-1).
    Masked cells ("NA") are set to 0 and the shares are rescaled to sum to 1.
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None)

    # The header row is the one containing "Classe d'âge" (layout: 2 header rows, empty first column)
    header_row = raw.index[raw.iloc[:, 1].map(lambda v: "Classe d" in str(v))][0]
    headers = [str(h) for h in raw.iloc[header_row].tolist()]
    has_pcg = any("PCG" in h for h in headers)

    # Data rows: from header_row + 2 until the first footnote / empty row
    body = raw.iloc[header_row + 2:, 1:]
    body = body[body.iloc[:, 1].isin(["femmes", "hommes"])]   # keep real data rows only

    id_cols = ["age", "sex", "hospitalised"] + (["pcg"] if has_pcg else [])
    body.columns = id_cols + ["n", "mean_cost", "sd"] + P_COLS
    df = body.copy()

    for c in ["hospitalised"] + (["pcg"] if has_pcg else []):
        df[c] = df[c].map({"oui": True, "non": False})
    for c in ["n", "mean_cost", "sd"] + P_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")   # "NA" -> NaN

    df["n_masked"] = df[P_COLS].isna().sum(axis=1)       # number of masked brackets
    shares = df[P_COLS].fillna(0) / 100
    df["share_sum_raw"] = shares.sum(axis=1).round(4)    # before rescaling (should be ~1)
    df[P_COLS] = shares.div(shares.sum(axis=1), axis=0).round(6)  # rescale to sum to 1
    df["small_sample"] = df["n"] < small_n
    return df.reset_index(drop=True)


def expense_distribution(expenses, age, sex, hospitalised, pcg=None):
    """Return the bracket probabilities (array of 9) and the row for one profile."""
    sel = expenses[(expenses.age == age) & (expenses.sex == sex)
                   & (expenses.hospitalised == hospitalised)]
    if pcg is not None and "pcg" in expenses:
        sel = sel[sel.pcg == pcg]
    if len(sel) != 1:
        raise ValueError(f"Expected one row for this profile, found {len(sel)}")
    row = sel.iloc[0]
    return row[P_COLS].to_numpy(dtype=float), row


# ---------------------------------------------------------------------------
# Export clean CSV files
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    out_dir = Path("data/clean")
    out_dir.mkdir(parents=True, exist_ok=True)

    premiums = load_premiums(raw_dir / "Praemien_CH.xlsx")
    premiums.to_csv(out_dir / "premiums_2027.csv", index=False)
    print("premiums:", premiums.shape)

    expenses = load_expenses(raw_dir / "Table_3.6_Brutto_EFIND2025.xlsx")
    expenses.to_csv(out_dir / "expenses_2025.csv", index=False)
    print("expenses:", expenses.shape,
          "| rows with masked cells:", int((expenses.n_masked > 0).sum()),
          "| small samples:", int(expenses.small_sample.sum()))