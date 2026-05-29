"""
Homework solution: One-Way ANOVA + Tukey/Scheffe post-hoc tests,
and Friedman test + post-hoc rank-sum comparisons.

Required packages:
    pip install numpy scipy pandas statsmodels
"""

import itertools
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import studentized_range
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ALPHA_ANOVA = 0.05
ALPHA_FRIEDMAN = 0.01


def anova_tukey_scheffe():
    # Reaction time scores from the Montelpare chapter example.
    drug_a = np.array([12, 15, 16, 13, 14, 15, 17, 11, 18, 14], dtype=float)
    drug_b = np.array([45, 54, 39, 65, 34, 63, 55, 51, 53, 60], dtype=float)
    placebo = np.array([25, 26, 28, 22, 26, 27, 23, 26, 24, 25], dtype=float)

    groups = [drug_a, drug_b, placebo]
    names = ["Drug A", "Drug B", "Placebo"]

    all_values = np.concatenate(groups)
    labels = np.repeat(names, [len(g) for g in groups])

    k = len(groups)
    N = len(all_values)
    grand_mean = all_values.mean()
    means = np.array([g.mean() for g in groups])

    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ss_total = ((all_values - grand_mean) ** 2).sum()

    df_between = k - 1
    df_within = N - k
    df_total = N - 1
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    F_manual = ms_between / ms_within
    F_scipy, p_scipy = stats.f_oneway(*groups)
    F_crit = stats.f.ppf(1 - ALPHA_ANOVA, df_between, df_within)

    print("=== 1. ONE-WAY ANOVA ===")
    print(pd.DataFrame({"Group": names, "n": [len(g) for g in groups], "Mean": means}))
    print(f"Grand mean = {grand_mean:.6f}\n")

    anova_table = pd.DataFrame({
        "Source": ["Between groups", "Within groups", "Total"],
        "SS": [ss_between, ss_within, ss_total],
        "df": [df_between, df_within, df_total],
        "MS": [ms_between, ms_within, np.nan],
        "F": [F_manual, np.nan, np.nan],
        "p-value": [p_scipy, np.nan, np.nan],
    })
    print(anova_table.to_string(index=False))
    print(f"\nSciPy f_oneway: F = {F_scipy:.6f}, p = {p_scipy:.6e}")
    print(f"F critical at alpha = 0.05: {F_crit:.6f}")
    print("Decision: reject H0 at alpha = 0.05.\n")

    print("=== Tukey HSD post-hoc test ===")
    tukey = pairwise_tukeyhsd(all_values, labels, alpha=ALPHA_ANOVA)
    print(tukey)

    qcrit = studentized_range.ppf(1 - ALPHA_ANOVA, k, df_within)
    tukey_hsd = qcrit * np.sqrt(ms_within / len(drug_a))
    print(f"\nTukey q critical = {qcrit:.6f}")
    print(f"Tukey minimum significant difference = {tukey_hsd:.6f}")

    print("\n=== Scheffe post-hoc check ===")
    scheffe_cd = np.sqrt((k - 1) * F_crit * ms_within * (1 / len(drug_a) + 1 / len(drug_b)))
    print(f"Scheffe critical difference for equal n = {scheffe_cd:.6f}")
    for (i, (name_i, group_i)), (j, (name_j, group_j)) in itertools.combinations(enumerate(zip(names, groups)), 2):
        diff = abs(group_i.mean() - group_j.mean())
        F_pair = diff ** 2 / (ms_within * (1 / len(group_i) + 1 / len(group_j))) / (k - 1)
        p_pair = stats.f.sf(F_pair, k - 1, df_within)
        print(f"{name_i} vs {name_j}: difference = {diff:.6f}, "
              f"Scheffe F = {F_pair:.6f}, p = {p_pair:.6e}, "
              f"significant = {diff > scheffe_cd}")


def friedman_metaheuristics():
    # Objective values for 4 metaheuristics over 21 instances.
    # Lower values are better, so rank 1 is assigned to the smallest value in each row.
    X = np.array([
        [565.02, 567.98, 568.86, 565.99],
        [662.84, 619.35, 617.48, 614.23],
        [664.73, 629.59, 620.50, 618.04],
        [857.84, 809.13, 817.71, 803.51],
        [949.98, 858.98, 858.95, 841.63],
        [1084.82, 949.89, 942.60, 961.47],
        [837.80, 832.91, 838.50, 830.48],
        [906.16, 881.26, 882.70, 876.21],
        [1000.27, 955.95, 921.97, 918.45],
        [1076.88, 1052.65, 1074.38, 1050.11],
        [1170.17, 1107.47, 1108.88, 1100.95],
        [1217.01, 1184.58, 1166.59, 1158.88],
        [1364.50, 1296.33, 1340.98, 1305.83],
        [1464.20, 1384.13, 1367.91, 1354.04],
        [1544.21, 1488.71, 1454.91, 1437.52],
        [1064.89, 1003.00, 1007.26, 1003.07],
        [1104.67, 1042.79, 1035.23, 1042.61],
        [1202.00, 1141.94, 1110.13, 1118.63],
        [887.22, 813.98, 823.01, 819.81],
        [963.06, 852.89, 859.06, 860.12],
        [952.29, 914.04, 915.38, 909.06],
    ], dtype=float)

    b, k = X.shape
    names = [f"M{j}" for j in range(1, k + 1)]
    ranks = np.array([stats.rankdata(row, method="average") for row in X])

    rank_sums = ranks.sum(axis=0)
    avg_ranks = ranks.mean(axis=0)
    squared_rank_sums = (ranks ** 2).sum(axis=0)

    A2 = (ranks ** 2).sum()
    B2 = (rank_sums ** 2).sum() / b
    T2 = ((b - 1) * (B2 - b * k * (k + 1) ** 2 / 4)) / (A2 - B2)
    Fcrit = stats.f.ppf(1 - ALPHA_FRIEDMAN, k - 1, (b - 1) * (k - 1))

    chi2_stat, chi2_p = stats.friedmanchisquare(*[X[:, j] for j in range(k)])

    print("\n\n=== 2. FRIEDMAN TEST ===")
    print(pd.DataFrame({
        "Metaheuristic": names,
        "Rank sum": rank_sums,
        "Average rank": avg_ranks,
        "Sum of squared ranks": squared_rank_sums,
    }).to_string(index=False))

    print(f"\nA2 = {A2:.6f}")
    print(f"B2 = {B2:.6f}")
    print(f"Paper F-approximation T2 = {T2:.6f}")
    print(f"F critical at alpha = 0.01 with df = (3, 60): {Fcrit:.6f}")
    print("Decision by paper method: reject H0.\n")

    print(f"SciPy friedmanchisquare: chi-square = {chi2_stat:.6f}, p = {chi2_p:.6e}")
    print("Decision by SciPy method: reject H0.\n")

    df = (b - 1) * (k - 1)
    tcrit = stats.t.ppf(1 - ALPHA_FRIEDMAN / 2, df)
    critical_difference = tcrit * np.sqrt((2 * b * (A2 - B2)) / ((b - 1) * (k - 1)))
    print("=== Post-hoc paired comparisons after Friedman ===")
    print(f"t critical = {tcrit:.6f}")
    print(f"Critical difference for rank sums = {critical_difference:.6f}")

    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(rank_sums[i] - rank_sums[j])
            rows.append({
                "Comparison": f"{names[i]} vs {names[j]}",
                "|Rank sum difference|": diff,
                "Different?": diff > critical_difference,
            })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nConclusion: M4 is best; M1 is worse than all the others; M2 and M3 are not significantly different.")


if __name__ == "__main__":
    anova_tukey_scheffe()
    friedman_metaheuristics()
