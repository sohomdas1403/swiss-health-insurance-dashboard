"""
Calculation engine for the Swiss health insurance deductible tool.

Part 1 - expected cost:
- Cost function C(F, M)
- Expected cost E[C(F, M)] and its bounds from the cost brackets
- Linear-premium benchmark P(F) = P(300) - 0.7 (F - 300) and check on real premiums
- Slope of the expected cost (concavity check) and bounds on the difference between two deductibles

Part 2 - risk aversion:
- CARA certainty equivalent CE_a(F) (for a = 0 it equals the expected cost)
- Coin-flip calibration of the risk-aversion coefficient a (bisection)
- Critical coefficients a* where the recommended deductible changes

The expected-cost part (bounds per cost bracket) will be added from Adrian's formulas.
All amounts are in CHF per year. Probabilities p are the 9 bracket shares from data.py.
"""

import numpy as np

from data import BRACKETS, DEDUCTIBLES_ADULT

COPAY_RATE = 0.10                          # co-payment above the deductible
COPAY_CAP = 700                            # maximum co-payment per year (adults)
COPAY_CAP_EXPENSE = COPAY_CAP / COPAY_RATE  # = 7'000: expenses above F + 7'000 add nothing
COIN_FLIP_LOSS = 2000                      # the gamble: pay 0 or 2'000 with prob. 1/2
MAX_DISCOUNT_RATE = 0.70                   # legal maximum: premium discount <= 70% of the extra risk
F_MIN = 300                                # lowest adult deductible

LOWER = np.array([lo for lo, hi in BRACKETS], dtype=float)
UPPER = np.array([hi for lo, hi in BRACKETS], dtype=float)          # last one = inf
MID = np.array([0, 250, 750, 1500, 3500, 7500, 15000, 35000, 75000], dtype=float)
# Note: for every adult deductible (F <= 2'500) the cost is constant once M >= F + 7'000 <= 9'500,
# so the value chosen inside the brackets above 10'000 (including the open one) does not matter.

VALUES = {"lower": LOWER, "mid": MID, "upper": UPPER}


# ---------------------------------------------------------------------------
# Cost function (common base of the model, needed for the risk-aversion part)
# ---------------------------------------------------------------------------
def out_of_pocket(F, M):
    """Deductible + co-payment paid by the insured for expenses M (works with arrays)."""
    M = np.asarray(M, dtype=float)
    return np.minimum(M, F) + COPAY_RATE * np.minimum(np.maximum(M - F, 0), COPAY_CAP_EXPENSE)


def cost(F, M, premium):
    """Total annual cost C(F, M) = premium + deductible paid + co-payment."""
    return premium + out_of_pocket(F, M)


# ---------------------------------------------------------------------------
# Part 1: expected cost 
# ---------------------------------------------------------------------------
def expected_cost(F, premium, p, values="mid"):
    """E[C(F, M)] = sum_i p_i C(F, M_i), with M_i the lower bound, midpoint or upper bound
    of each cost bracket."""
    return float(np.sum(p * cost(F, VALUES[values], premium)))


def expected_cost_bounds(F, premium, p):
    """Lower and upper bound of E[C(F, M)]. C is increasing in M, so taking the lowest
    (highest) expense of every bracket gives a lower (upper) bound."""
    return expected_cost(F, premium, p, "lower"), expected_cost(F, premium, p, "upper")


def linear_premium(F, premium_300):
    """Theoretical benchmark: insurers give the maximum legal discount at every level,
    P(F) = P(300) - 0.7 (F - 300)."""
    return premium_300 - MAX_DISCOUNT_RATE * (F - F_MIN)


def premium_discount_check(premiums):
    """Compare the real premiums with the legal maximum discount.
    Returns, for each F, the actual discount P(300) - P(F), the maximum 0.7 (F - 300)
    and the share of the maximum that is granted (1.0 = linear benchmark holds)."""
    p300 = premiums[F_MIN]
    out = {}
    for F in sorted(premiums):
        if F == F_MIN:
            continue
        actual = p300 - premiums[F]
        maximum = MAX_DISCOUNT_RATE * (F - F_MIN)
        out[F] = {"discount": actual, "max_discount": maximum, "share_of_max": actual / maximum}
    return out


def survival(m, p, values="mid"):
    """S(m) = Pr(M > m), using one representative value per bracket."""
    return float(np.sum(p[VALUES[values] > m]))


def expected_cost_slope(F, p, values="mid"):
    """Slope of the expected cost under the linear-premium benchmark:
    dE[C]/dF = -0.7 + 0.9 S(F) + 0.1 S(F + 7000)."""
    return -MAX_DISCOUNT_RATE + (1 - COPAY_RATE) * survival(F, p, values) \
        + COPAY_RATE * survival(F + COPAY_CAP_EXPENSE, p, values)


def is_concave(premiums, p, values="mid", tol=1e-9):
    """Discrete concavity check of F -> E[C(F, M)] on the available deductibles:
    the slopes between consecutive deductibles must never increase."""
    Fs = sorted(premiums)
    ec = [expected_cost(F, premiums[F], p, values) for F in Fs]
    slopes = [(ec[k + 1] - ec[k]) / (Fs[k + 1] - Fs[k]) for k in range(len(Fs) - 1)]
    return all(slopes[k + 1] <= slopes[k] + tol for k in range(len(slopes) - 1)), slopes


def difference_bounds(F1, F2, premiums, p):
    """Bounds on E[C(F1, M)] - E[C(F2, M)] that use only the bracket shares.
    d(M) = C(F1, M) - C(F2, M) is piecewise linear in M, so inside each bracket its minimum
    and maximum are reached at the bracket ends or at the kinks F1, F2, F1 + 7000, F2 + 7000.
    If the upper bound is < 0, F1 is cheaper on average whatever the costs inside the brackets."""
    kinks = np.array([F1, F2, F1 + COPAY_CAP_EXPENSE, F2 + COPAY_CAP_EXPENSE])
    d = lambda M: cost(F1, M, premiums[F1]) - cost(F2, M, premiums[F2])
    low = high = 0.0
    for pi, lo, hi in zip(p, LOWER, UPPER):
        if pi == 0:
            continue
        top = hi if np.isfinite(hi) else max(lo, kinks.max()) + 1.0   # d is constant after the last kink
        points = np.concatenate(([lo, top], kinks[(kinks > lo) & (kinks < top)]))
        values = d(points)
        low += pi * values.min()
        high += pi * values.max()
    return float(low), float(high)


# ---------------------------------------------------------------------------
# Part 2: risk aversion (CARA utility u(x) = -exp(-a x))
# ---------------------------------------------------------------------------
def certainty_equivalent(F, premium, p, a, values="mid"):
    """CE_a(F) = (1/a) ln( sum_i p_i exp(a C_i) ). For a = 0 it is the expected cost.
    Numerically stable ("log-sum-exp"): the largest exponent a*C_i is factored out, so that
    every exponential is <= 1 and nothing overflows, even for very large |a|."""
    c = cost(F, VALUES[values], premium)
    if abs(a) < 1e-12:
        return float(np.sum(p * c))
    keep = p > 0                                  # brackets with zero probability play no role
    x = a * c[keep]
    x_max = x.max()
    return float((x_max + np.log(np.sum(p[keep] * np.exp(x - x_max)))) / a)


def risk_premium(F, premium, p, a, values="mid"):
    """Risk premium pi_a(F) = CE_a(F) - E[C(F, M)] >= 0 ."""
    return certainty_equivalent(F, premium, p, a, values) - certainty_equivalent(F, premium, p, 0.0, values)


def certainty_equivalent_taylor(F, premium, p, a, values="mid"):
    """Second-order (Arrow-Pratt) approximation: CE ~ E[C] + (a/2) Var(C) (section 2.4).
    Used only to compare with the exact certainty equivalent."""
    c = cost(F, VALUES[values], premium)
    mean = float(np.sum(p * c))
    var = float(np.sum(p * (c - mean) ** 2))
    return mean + a / 2 * var


def coin_flip_answer(a, loss=COIN_FLIP_LOSS):
    """Certainty equivalent w(a) of the gamble 'pay 0 or `loss` with probability 1/2'."""
    if abs(a) < 1e-12:
        return loss / 2
    return float((np.log(0.5) + np.logaddexp(0.0, a * loss)) / a)


def a_from_answer(w, loss=COIN_FLIP_LOSS, tol=1e-10, max_iter=200):
    """Solve w(a) = w for a by bisection. w(a) is continuous and increasing, from 0 (a -> -inf)
    through loss/2 (a = 0) to loss (a -> +inf), so the solution exists and is unique."""
    if not 0 < w < loss:
        raise ValueError(f"The answer must be strictly between 0 and {loss} CHF.")
    if abs(w - loss / 2) < 1e-9:
        return 0.0
    lo, hi = (0.0, 1.0) if w > loss / 2 else (-1.0, 0.0)
    while (coin_flip_answer(hi) < w) if w > loss / 2 else (coin_flip_answer(lo) > w):
        if w > loss / 2:
            hi *= 2                      # widen the interval until it contains the root
        else:
            lo *= 2
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if coin_flip_answer(mid) < w:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def best_deductible(premiums, p, a=0.0, values="mid"):
    """Optimal deductible for risk aversion a (a = 0: expected cost).
    premiums: dict {deductible: annual premium}. Returns (best F, {F: CE_a(F)})."""
    ce = {F: certainty_equivalent(F, premiums[F], p, a, values) for F in premiums}
    return min(ce, key=ce.get), ce


def critical_thresholds(premiums, p, a_max=0.01, n_grid=4001, values="mid", tol=1e-12):
    """Values a* where the optimal deductible changes, found on a grid then refined by
    bisection on D(a) = CE_a(F1) - CE_a(F2). Each a* is also translated into the
    coin-flip answer w* = w(a*)."""
    grid = np.linspace(0.0, a_max, n_grid)
    best = [best_deductible(premiums, p, a, values)[0] for a in grid]
    out = []
    for k in range(1, n_grid):
        F1, F2 = best[k - 1], best[k]
        if F1 == F2:
            continue
        lo, hi = grid[k - 1], grid[k]
        D = lambda a: certainty_equivalent(F1, premiums[F1], p, a, values) - \
            certainty_equivalent(F2, premiums[F2], p, a, values)
        while hi - lo > tol:
            mid = (lo + hi) / 2
            if D(mid) < 0:   # F1 still preferred
                lo = mid
            else:
                hi = mid
        a_star = (lo + hi) / 2
        out.append({"from": F1, "to": F2, "a_star": a_star, "w_star": coin_flip_answer(a_star)})
    return out


def robustness(w, thresholds):
    """Where the user's coin-flip answer w falls relative to the thresholds w*.
    Returns the nearest threshold and the distance to it in CHF (a large distance means a
    robust recommendation, a small one means that both deductibles are defensible)."""
    if not thresholds:
        return None
    nearest = min(thresholds, key=lambda t: abs(t["w_star"] - w))
    return {"nearest": nearest, "distance_chf": abs(nearest["w_star"] - w)}


# ---------------------------------------------------------------------------
# Quick check with the real data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    from data import premium_schedule, expense_distribution

    prem = pd.read_csv("data/clean/premiums_2027.csv")
    exp = pd.read_csv("data/clean/expenses_2025.csv")

    schedule = premium_schedule(prem, "VD", "PR_REG_1", "adult", True, "BASE")
    premiums = schedule.median().to_dict()          # median insurer, Lausanne, standard model
    p, _ = expense_distribution(exp, "26-30", "hommes", False, pcg=False)

    print("Premiums (median insurer):", {F: round(v) for F, v in premiums.items()})

    print("\n--- Part 1: expected cost ---")
    check = premium_discount_check(premiums)
    print("Share of the maximum legal discount granted:",
          {F: round(v["share_of_max"], 3) for F, v in check.items()})
    for F in DEDUCTIBLES_ADULT:
        lo, hi = expected_cost_bounds(F, premiums[F], p)
        print(f"F = {F}: E[C] = {expected_cost(F, premiums[F], p):.0f}  (bounds {lo:.0f} - {hi:.0f}),"
              f" slope {expected_cost_slope(F, p):+.3f}")
    concave, slopes = is_concave(premiums, p)
    print("Concave on the six deductibles:", concave, "| slopes:", [round(s, 3) for s in slopes])
    lo, hi = difference_bounds(2500, 300, premiums, p)
    print(f"E[C(2500)] - E[C(300)] is between {lo:.0f} and {hi:.0f} CHF")

    print("\n--- Part 2: risk aversion ---")
    a = a_from_answer(1400)
    print(f"a for an answer of 1400: {a:.6f}  (check w(a) = {coin_flip_answer(a):.1f})")
    print("F | expected cost (a = 0) | certainty equivalent (a for 1400)")
    for F in DEDUCTIBLES_ADULT:
        print(F, round(certainty_equivalent(F, premiums[F], p, 0.0)),
              round(certainty_equivalent(F, premiums[F], p, a)))
    print("Best (a = 0):", best_deductible(premiums, p)[0],
          "| best (a for 1400):", best_deductible(premiums, p, a)[0])
    thresholds = critical_thresholds(premiums, p)
    for t in thresholds:
        print(f"switch {t['from']} -> {t['to']}: a* = {t['a_star']:.6f}, w* = {t['w_star']:.0f}")
    print("Exact CE vs Taylor approximation (F = 2500):")
    for w in [1200, 1400, 1600, 1800]:
        aw = a_from_answer(w)
        print(f"  answer {w}: exact {certainty_equivalent(2500, premiums[2500], p, aw):.0f}, "
              f"Taylor {certainty_equivalent_taylor(2500, premiums[2500], p, aw):.0f}")
    r = robustness(1400, thresholds)
    print(f"Answer 1400: nearest switch {r['nearest']['from']} -> {r['nearest']['to']} "
          f"at {r['nearest']['w_star']:.0f}, distance {r['distance_chf']:.0f} CHF")