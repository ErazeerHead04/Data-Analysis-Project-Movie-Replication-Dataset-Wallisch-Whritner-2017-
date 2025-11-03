# ============================================================
# Data Analysis Project – Movie Replication Dataset (Wallisch & Whritner, 2017)
# Author: Ahmmed Razee
# ============================================================
# This script performs hypothesis testing for multiple research questions (Q1–Q10)
# taught under IDS lectures 1–7 (sampling, hypothesis testing, significance testing,
# parametric vs. nonparametric inference).
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# Load Data
# ============================================================

df = pd.read_csv("movieReplicationSet.csv", encoding="ISO-8859-1")

# Identify movie columns — first 400 columns correspond to movie ratings
movie_cols = df.columns[:400]

# ============================================================
# Q1: Popular vs. Unpopular Movies
# ============================================================
# Compare mean ratings between popular and unpopular movies
# Using two-sample t-test (parametric) or Mann–Whitney U (nonparametric)
# H0: μ_popular = μ_unpopular
# H1: μ_popular ≠ μ_unpopular

popular_movies = df[movie_cols].mean().nlargest(50)
unpopular_movies = df[movie_cols].mean().nsmallest(50)

popular_mean = popular_movies.values
unpopular_mean = unpopular_movies.values

# Normality check (Shapiro–Wilk test)
p_norm1 = stats.shapiro(popular_mean)[1]
p_norm2 = stats.shapiro(unpopular_mean)[1]

if p_norm1 < 0.05 or p_norm2 < 0.05:
    # Nonparametric alternative if not normal
    stat, p = stats.mannwhitneyu(popular_mean, unpopular_mean)
    test_type = "Mann–Whitney U"
else:
    # Parametric test (Welch’s t-test)
    stat, p = stats.ttest_ind(popular_mean, unpopular_mean, equal_var=False)
    test_type = "Welch’s t-test"

print(f"Q1: Popular vs. Unpopular Movies — {test_type}, t={stat:.3f}, p={p:.10e}, "
      f"mean(popular)={popular_mean.mean():.2f}, mean(unpopular)={unpopular_mean.mean():.2f}")

# Visualization
plt.figure(figsize=(6, 4))
plt.bar(["Popular", "Unpopular"], [popular_mean.mean(), unpopular_mean.mean()],
        color=["gold", "gray"], edgecolor="black")
plt.ylabel("Average Rating")
plt.title("Q1: Popular vs. Unpopular Movies")
plt.tight_layout()
plt.savefig("Q1_PopularityComparison.png", dpi=300)
plt.close()

# ============================================================
# Q2: New vs. Old Movies
# ============================================================
# H0: μ_new = μ_old
# H1: μ_new ≠ μ_old
# Use Welch’s t-test (safe for unequal variances)

# Assume "new" means release year ≥ 2000 (approximate)
new_movies = [m for m in movie_cols if "(20" in m]
old_movies = [m for m in movie_cols if "(19" in m]

new_ratings = df[new_movies].mean()
old_ratings = df[old_movies].mean()

t_stat, p_val = stats.ttest_ind(new_ratings, old_ratings, equal_var=False)
print(f"Q2: New vs. Old Movies — t={t_stat:.3f}, p={p_val:.5f}, "
      f"mean(new)={new_ratings.mean():.2f}, mean(old)={old_ratings.mean():.2f}")

plt.figure(figsize=(6, 4))
plt.bar(["New (2000+)", "Old (<2000)"], [new_ratings.mean(), old_ratings.mean()],
        color=["lightgreen", "lightgray"], edgecolor="black")
plt.ylabel("Average Rating")
plt.title("Q2: New vs. Old Movies")
plt.tight_layout()
plt.savefig("Q2_NewOldMovies.png", dpi=300)
plt.close()

# ============================================================
<<<<<<< HEAD
# Q3 — Gender Differences
# ============================================================

df = df[df[gender_col].isin([1, 2])]  # clean gender column
p_values_gender = []

movie_shrek = 'Shrek (2001)'
male_ratings = df.loc[df[gender_col] == 2, movie_shrek].dropna()
female_ratings = df.loc[df[gender_col] == 1, movie_shrek].dropna()

if len(male_ratings) > 5 and len(female_ratings) > 5:
    t_shrek, p_shrek = stats.ttest_ind(male_ratings, female_ratings, equal_var=False)
    mean_male = male_ratings.mean()
    mean_female = female_ratings.mean()
    print(f"Q3: 'Shrek (2001)' — t={t_shrek:.3f}, p={p_shrek:.5f}, "
      f"mean(male)={mean_male:.2f}, mean(female)={mean_female:.2f}")
    plt.figure(figsize=(5, 5))
    plt.bar(['Male', 'Female'], [mean_male, mean_female],
            color=['green', 'red'], edgecolor='black')
    plt.title("Mean Ratings of 'Shrek (2001)' (Q3)")
    plt.ylabel("Mean Rating (0–4)")
    plt.tight_layout()
    plt.savefig("Q3_Shrek_Gender.png", dpi=300)
    plt.close()

# ============================================================
# Q4 — Gender Differences
=======
# Q4: Gender Differences Across Movies
>>>>>>> af11f46 (Final update: test.py (Q1–Q10))
# ============================================================
# H0: mean_rating_male = mean_rating_female
# H1: mean_rating_male ≠ mean_rating_female
# Apply Bonferroni correction for multiple testing

gender_col = [c for c in df.columns if "Gender identity" in c][0]
alpha = 0.005 / len(movie_cols)  # Bonferroni correction

gender_df = df[df[gender_col].isin([1, 2])]  # 1=female, 2=male
p_values = []

for m in movie_cols:
    female = gender_df[gender_df[gender_col] == 1][m].dropna()
    male = gender_df[gender_df[gender_col] == 2][m].dropna()
    _, p = stats.ttest_ind(female, male, equal_var=False)
    p_values.append(p)

sig_movies = np.sum(np.array(p_values) < alpha)
print(f"Q4: {sig_movies} movies ({sig_movies/len(movie_cols)*100:.2f}%) differ significantly by gender (α={alpha:.5f}).")

plt.figure(figsize=(7,4))
plt.hist(p_values, bins=50, color="lightblue", edgecolor="black")
plt.title("Q4: Gender Differences — p-value Distribution")
plt.xlabel("p-value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("Q4_GenderDifferences_Histogram.png", dpi=300)
plt.close()

# ============================================================
# Q5: Only Child Effect (Example: The Lion King)
# ============================================================
# H0: mean_only = mean_sibling
# H1: mean_only ≠ mean_sibling

only_col = [c for c in df.columns if "Are you an only child" in c][0]
movie = "The Lion King (1994)"

only = df[df[only_col] == 1][movie].dropna()
sibling = df[df[only_col] == 0][movie].dropna()

t, p = stats.ttest_ind(only, sibling, equal_var=False)
print(f"Q5: '{movie}' — t={t:.3f}, p={p:.5f}, mean(only)={only.mean():.2f}, mean(sibling)={sibling.mean():.2f}")

plt.figure(figsize=(6,4))
plt.bar(["Only Child", "Has Siblings"], [only.mean(), sibling.mean()],
        color=["violet", "lightgray"], edgecolor="black")
plt.title(f"Q5: '{movie}' — Only Child Effect")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.savefig("Q5_LionKing_OnlyChild.png", dpi=300)
plt.close()

# ============================================================
# Q6: Number of Movies Showing 'Only Child' Effect
# ============================================================

alpha_corr = 0.005 / len(movie_cols)  # Bonferroni
p_values = []

for m in movie_cols:
    only = df[df[only_col] == 1][m].dropna()
    sibling = df[df[only_col] == 0][m].dropna()
    if len(only) > 2 and len(sibling) > 2:
        _, p = stats.ttest_ind(only, sibling, equal_var=False)
        p_values.append(p)

sig_movies = np.sum(np.array(p_values) < alpha_corr)
print(f"Q6: {sig_movies} movies ({sig_movies/len(movie_cols)*100:.2f}%) show an 'only child' effect (α={alpha_corr:.5f}).")

plt.figure(figsize=(7,4))
plt.hist(p_values, bins=50, color="pink", edgecolor="black")
plt.title("Q6: Only Child Effect — p-value Distribution")
plt.xlabel("p-value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("Q6_OnlyChild_Histogram.png", dpi=300)
plt.close()

# ============================================================
# Q7: Viewing Preference — Social vs. Alone (Example Movie)
# ============================================================

social_col = [c for c in df.columns if "Movies are best enjoyed alone" in c][0]
movie = "The Wolf of Wall Street (2013)"

<<<<<<< HEAD
    plt.figure(figsize=(5, 5))
    plt.bar(['Prefers Socially', 'Prefers Alone'],
            [mean_social, mean_solo],
            color=['mediumseagreen', 'gold'], edgecolor='black')
    plt.title("Mean Ratings of 'The Wolf of Wall Street (2013)' (Q7)")
    plt.ylabel("Mean Rating (0–4)")
    plt.tight_layout()
    plt.savefig("Q7_WolfOfWallStreet_SocialPref.png", dpi=300)
    plt.close()
    
# ============================================================
# Q8 — What proportion of movies exhibit such a “social watching” effect?
# ============================================================

p_values_social = []

for movie in movie_cols:
    solo_ratings = df.loc[df[social_col] == 1, movie].dropna()
    social_ratings = df.loc[df[social_col] == 0, movie].dropna()
    
    if len(solo_ratings) > 10 and len(social_ratings) > 10:
        t_social, p_social = stats.ttest_ind(solo_ratings, social_ratings, equal_var=False, nan_policy='omit')
        p_values_social.append(p_social)

significant_social = sum(p < 0.005 for p in p_values_social)
prop_social_diff = significant_social / len(p_values_social)

print(f"Q8: {significant_social} movies ({prop_social_diff:.2%}) show a 'social' effect (α=0.005).")

plt.figure(figsize=(7, 5))
plt.hist(p_values_social, bins=30, color='lightgreen', edgecolor='black')
plt.axvline(0.005, color='red', linestyle='--', label='α = 0.005')
plt.title("Distribution of p-values across Social Comparisons (Q8)")
plt.xlabel("p-value")
plt.ylabel("Number of Movies")
plt.legend()
plt.tight_layout()
plt.savefig("Q8_Social_Histogram.png", dpi=300)
plt.close()

# ============================================================
# Q9 - Is the ratings distribution of ‘Home Alone (1990)’ different than that of ‘Finding Nemo (2003)’? 
# ============================================================

from statsmodels.distributions.empirical_distribution import ECDF

col1 = "Home Alone (1990)"
col2 = "Finding Nemo (2003)"

d1_full = df[col1]
d2_full = df[col2]

# Element-wise cleaning for independent-sample style tests
d1 = d1_full.dropna().values
d2 = d2_full.dropna().values
n1, n2 = len(d1), len(d2)
n1, n2
e1 = ECDF(d1)
e2 = ECDF(d2)


stats.ks_2samp(d1, d2)

print(f"Q9: The KS test for the two movies gives us a p-value {1.8193141593127503e-09} (α=0.005).")

plt.step(e1.x, e1.y, color='red', label = 'Home Alone (1990)')
plt.step(e2.x, e2.y, color='blue', label = 'Finding Nemo (2003)')
plt.xlabel('rating')
plt.ylabel('cumulative probability')
plt.title('Distribution Comparison')
plt.grid()
plt.savefig("Q9_Dist_Comp.png", dpi=300)
plt.close()

# ============================================================
# QBonus - Does life stress affect how people feel about 'The Silence of the Lambs (1991)'? 
# ============================================================

luck_col = "My life is very stressful"

p_values_luck = []

# Define movie columns (first 400)
movie_lamb = 'The Silence of the Lambs (1991)'
unluckiest_watchers = df.loc[df[luck_col] == 1, movie_lamb].dropna()
unlucky_watchers = df.loc[df[luck_col] == 2, movie_lamb].dropna()
moderate_watchers = df.loc[df[luck_col] == 3, movie_lamb].dropna()
lucky_watchers = df.loc[df[luck_col] == 4, movie_lamb].dropna()
luckiest_watchers = df.loc[df[luck_col] == 5, movie_lamb].dropna()

if len(unluckiest_watchers) > 5 and len(unlucky_watchers) > 5 and len(moderate_watchers) > 5 and len(lucky_watchers) > 5 and len(luckiest_watchers) > 5:
    t_luck, p_luck = stats.f_oneway(unluckiest_watchers, unlucky_watchers, moderate_watchers, lucky_watchers, luckiest_watchers)
    mean_unluckiest = unluckiest_watchers.mean()
    mean_unlucky = unlucky_watchers.mean()
    mean_moderate = moderate_watchers.mean()
    mean_lucky = lucky_watchers.mean()
    mean_luckiest = luckiest_watchers.mean()
    print(f"Bonus: 'The Silence of the Lambs (1991)' — t={t_luck:.3f}, p={p_luck:.5f}, "
          f"mean(most stressed)={mean_unluckiest:.2f}, mean(stressed)={mean_unlucky:.2f}, mean(moderate stress) = {mean_moderate:.2f}, mean(little stress) = {mean_lucky:.2f}, mean(barely stressed) = {mean_luckiest:.2f} ")
        
    plt.figure(figsize=(5, 5))
    plt.bar(['Most Stressed', 'Stressed', 'Moderate', 'Not Stressed', 'Least Stressed'],
           [mean_unluckiest, mean_unlucky, mean_moderate, mean_lucky, mean_luckiest],
           color=['mediumseagreen', 'gold', 'blue', 'yellow', 'red'], edgecolor='black')
    plt.title("Mean Ratings of 'The Silence of the Lambs (1991)' (Bonus)")
    plt.ylabel("Mean Rating (0–4)")
    plt.tight_layout()
    plt.savefig("QBonus_SilenceLambs_Stress.png", dpi=300)
    plt.close()

print("\n✅ All plots saved for Q1–Q9 and Bonus")





=======
social = df[df[social_col] == 0][movie].dropna()  # 0 = social
alone = df[df[social_col] == 1][movie].dropna()   # 1 = alone

t, p = stats.ttest_ind(social, alone, equal_var=False)
print(f"Q7: '{movie}' — t={t:.3f}, p={p:.5f}, mean(social)={social.mean():.2f}, mean(alone)={alone.mean():.2f}")

plt.figure(figsize=(6,4))
plt.bar(["Social Viewers", "Solo Viewers"], [social.mean(), alone.mean()],
        color=["lightcoral", "lightgray"], edgecolor="black")
plt.ylabel("Average Rating")
plt.title(f"Q7: '{movie}' — Social vs. Alone Viewing")
plt.tight_layout()
plt.savefig("Q7_WolfOfWallStreet_SocialPref.png", dpi=300)
plt.close()

# ============================================================
# Q10: Franchise Consistency — ANOVA & Kruskal–Wallis
# ============================================================
# H0: all movies in franchise have same mean rating
# H1: at least one differs
# Parametric (ANOVA) and Nonparametric (Kruskal–Wallis)

print("\n================ Q10: Franchise Consistency Analysis ================\n")

franchises = [
    'Star Wars', 'Harry Potter', 'The Matrix',
    'Indiana Jones', 'Jurassic Park',
    'Pirates of the Caribbean', 'Toy Story', 'Batman'
]

anova_results = []

for series in franchises:
    matched_movies = [m for m in movie_cols if series.lower() in m.lower()]
    if len(matched_movies) < 2:
        continue

    samples = [df[m].dropna() for m in matched_movies]
    f_stat, p_val_anova = stats.f_oneway(*samples)
    h_stat, p_val_kw = stats.kruskal(*samples)

    anova_results.append({
        "Franchise": series,
        "NumMovies": len(matched_movies),
        "p_ANOVA": p_val_anova,
        "p_Kruskal": p_val_kw
    })

anova_df = pd.DataFrame(anova_results)
anova_df["Inconsistent_ANOVA"] = anova_df["p_ANOVA"] < 0.05
anova_df["Inconsistent_KW"] = anova_df["p_Kruskal"] < 0.05

print(anova_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

n_inconsistent_anova = anova_df["Inconsistent_ANOVA"].sum()
n_inconsistent_kw = anova_df["Inconsistent_KW"].sum()

print(f"\nQ10 (ANOVA): {n_inconsistent_anova} franchises show inconsistent quality (p < 0.05).")
print(f"Q10 (Kruskal–Wallis): {n_inconsistent_kw} franchises show inconsistent quality (p < 0.05).")

plt.figure(figsize=(9,5))
width = 0.35
x = np.arange(len(anova_df))

plt.bar(x - width/2, anova_df["p_ANOVA"], width, label="ANOVA p-value", color="steelblue", edgecolor="black")
plt.bar(x + width/2, anova_df["p_Kruskal"], width, label="Kruskal–Wallis p-value", color="seagreen", edgecolor="black")
plt.axhline(0.05, color="red", linestyle="--", label="α = 0.05")
plt.xticks(x, anova_df["Franchise"], rotation=45, ha="right")
plt.ylabel("p-value")
plt.title("Q10: Consistency of Ratings Across Movie Franchises")
plt.legend()
plt.tight_layout()
plt.savefig("Q10_FranchiseConsistency_Tests.png", dpi=300)
plt.close()

print("\n All analyses complete. Plots saved for Q1–Q10.\n")
>>>>>>> af11f46 (Final update: test.py (Q1–Q10))
