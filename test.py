# ============================================================
# Data Analysis Project – Movie Replication Dataset (Wallisch & Whritner, 2017)
# ============================================================
# This script performs hypothesis testing for multiple research questions (Q1–Q10 + Bonus)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("movieReplicationSet.csv", encoding="ISO-8859-1")
movie_cols = df.columns[:400]  # first 400 columns are movie ratings

# ============================================================
# Q1: Popular vs. Unpopular Movies
# ============================================================

popular_movies = df[movie_cols].mean().nlargest(50)
unpopular_movies = df[movie_cols].mean().nsmallest(50)

popular_mean = popular_movies.values
unpopular_mean = unpopular_movies.values

# Normality check
p_norm1 = stats.shapiro(popular_mean)[1]
p_norm2 = stats.shapiro(unpopular_mean)[1]

if p_norm1 < 0.05 or p_norm2 < 0.05:
    stat, p = stats.mannwhitneyu(popular_mean, unpopular_mean)
    test_type = "Mann–Whitney U"
else:
    stat, p = stats.ttest_ind(popular_mean, unpopular_mean, equal_var=False)
    test_type = "Welch’s t-test"

print(f"Q1: Popular vs. Unpopular Movies — {test_type}, t={stat:.3f}, p={p:.10e}, "
      f"mean(popular)={popular_mean.mean():.2f}, mean(unpopular)={unpopular_mean.mean():.2f}")

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

new_movies = [m for m in movie_cols if "(20" in m]
old_movies = [m for m in movie_cols if "(19" in m]

new_ratings = df[new_movies].mean()
old_ratings = df[old_movies].mean()

t_stat, p_val = stats.ttest_ind(new_ratings, old_ratings, equal_var=False)
print(f"Q2: New vs. Old Movies — t={t_stat:.3f}, p={p_val:.5f}, "
      f"mean(new)={new_ratings.mean():.2f}, mean(old)={old_ratings.mean():.2f}")

plt.bar(["New (2000+)", "Old (<2000)"], [new_ratings.mean(), old_ratings.mean()],
        color=["lightgreen", "lightgray"], edgecolor="black")
plt.ylabel("Average Rating")
plt.title("Q2: New vs. Old Movies")
plt.tight_layout()
plt.savefig("Q2_NewOldMovies.png", dpi=300)
plt.close()

# ============================================================
# Q3: Gender Difference Example (Shrek)
# ============================================================

gender_col = [c for c in df.columns if "Gender identity" in c][0]
df_gender = df[df[gender_col].isin([1, 2])]  # clean gender data
movie_shrek = 'Shrek (2001)'

male = df_gender[df_gender[gender_col] == 2][movie_shrek].dropna()
female = df_gender[df_gender[gender_col] == 1][movie_shrek].dropna()

if len(male) > 5 and len(female) > 5:
    t_shrek, p_shrek = stats.ttest_ind(male, female, equal_var=False)
    print(f"Q3: 'Shrek (2001)' — t={t_shrek:.3f}, p={p_shrek:.5f}, mean(male)={male.mean():.2f}, mean(female)={female.mean():.2f}")
    plt.bar(['Male', 'Female'], [male.mean(), female.mean()],
            color=['green', 'red'], edgecolor='black')
    plt.title("Mean Ratings: 'Shrek (2001)' (Q3)")
    plt.ylabel("Mean Rating (0–4)")
    plt.tight_layout()
    plt.savefig("Q3_Shrek_Gender.png", dpi=300)
    plt.close()

# ============================================================
# Q4: Gender Differences Across All Movies
# ============================================================

alpha = 0.005 / len(movie_cols)
p_values = []

for m in movie_cols:
    female = df_gender[df_gender[gender_col] == 1][m].dropna()
    male = df_gender[df_gender[gender_col] == 2][m].dropna()
    if len(female) > 5 and len(male) > 5:
        _, p = stats.ttest_ind(female, male, equal_var=False)
        p_values.append(p)

sig_movies = np.sum(np.array(p_values) < alpha)
print(f"Q4: {sig_movies} movies ({sig_movies/len(movie_cols)*100:.2f}%) differ significantly by gender (α={alpha:.5f}).")

plt.hist(p_values, bins=50, color="lightblue", edgecolor="black")
plt.title("Q4: Gender Differences — p-value Distribution")
plt.xlabel("p-value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("Q4_GenderDifferences_Histogram.png", dpi=300)
plt.close()

# ============================================================
# Q5–Q7: Only Child & Viewing Preferences
# ============================================================

only_col = [c for c in df.columns if "Are you an only child" in c][0]
social_col = [c for c in df.columns if "Movies are best enjoyed alone" in c][0]

# Q5 Example: The Lion King (1994)
movie = "The Lion King (1994)"
only = df[df[only_col] == 1][movie].dropna()
sibling = df[df[only_col] == 0][movie].dropna()
t, p = stats.ttest_ind(only, sibling, equal_var=False)
print(f"Q5: '{movie}' — t={t:.3f}, p={p:.5f}, mean(only)={only.mean():.2f}, mean(sibling)={sibling.mean():.2f}")

# Q6: Count of Only-Child Effects
alpha_corr = 0.005 / len(movie_cols)
p_values = []
for m in movie_cols:
    only = df[df[only_col] == 1][m].dropna()
    sibling = df[df[only_col] == 0][m].dropna()
    if len(only) > 2 and len(sibling) > 2:
        _, p = stats.ttest_ind(only, sibling, equal_var=False)
        p_values.append(p)
sig_movies = np.sum(np.array(p_values) < alpha_corr)
print(f"Q6: {sig_movies} movies ({sig_movies/len(movie_cols)*100:.2f}%) show 'only child' effect (α={alpha_corr:.5f}).")

# Q7: Viewing Preferences — The Wolf of Wall Street
movie = "The Wolf of Wall Street (2013)"
social = df[df[social_col] == 0][movie].dropna()
alone = df[df[social_col] == 1][movie].dropna()
t, p = stats.ttest_ind(social, alone, equal_var=False)
print(f"Q7: '{movie}' — t={t:.3f}, p={p:.5f}, mean(social)={social.mean():.2f}, mean(alone)={alone.mean():.2f}")

# ============================================================
# Q8: Proportion of Movies with “Social Watching” Effect
# ============================================================

p_values_social = []
for movie in movie_cols:
    solo = df[df[social_col] == 1][movie].dropna()
    group = df[df[social_col] == 0][movie].dropna()
    if len(solo) > 10 and len(group) > 10:
        _, p = stats.ttest_ind(solo, group, equal_var=False)
        p_values_social.append(p)

sig_social = sum(p < 0.005 for p in p_values_social)
print(f"Q8: {sig_social} movies ({sig_social/len(p_values_social)*100:.2f}%) show 'social watching' effect (α=0.005).")

# ============================================================
# Q9: Distribution Difference — Home Alone vs. Finding Nemo
# ============================================================

col1, col2 = "Home Alone (1990)", "Finding Nemo (2003)"
d1, d2 = df[col1].dropna(), df[col2].dropna()
stat, p_ks = stats.ks_2samp(d1, d2)
print(f"Q9: 'Home Alone' vs 'Finding Nemo' — KS test, p={p_ks:.2e}")

e1, e2 = ECDF(d1), ECDF(d2)
plt.step(e1.x, e1.y, color='red', label='Home Alone (1990)')
plt.step(e2.x, e2.y, color='blue', label='Finding Nemo (2003)')
plt.xlabel('Rating')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.title("Q9: Distribution Comparison")
plt.tight_layout()
plt.savefig("Q9_Dist_Comp.png", dpi=300)
plt.close()

# ============================================================
# Q10: Franchise Consistency (ANOVA & Kruskal–Wallis)
# ============================================================

franchises = [
    'Star Wars', 'Harry Potter', 'The Matrix',
    'Indiana Jones', 'Jurassic Park',
    'Pirates of the Caribbean', 'Toy Story', 'Batman'
]

results = []
for f in franchises:
    matched = [m for m in movie_cols if f.lower() in m.lower()]
    if len(matched) >= 2:
        samples = [df[m].dropna() for m in matched]
        f_stat, p_anova = stats.f_oneway(*samples)
        h_stat, p_kw = stats.kruskal(*samples)
        results.append([f, len(matched), p_anova, p_kw])

anova_df = pd.DataFrame(results, columns=["Franchise", "NumMovies", "p_ANOVA", "p_KW"])
anova_df["Inconsistent_ANOVA"] = anova_df["p_ANOVA"] < 0.05
anova_df["Inconsistent_KW"] = anova_df["p_KW"] < 0.05
print("\nQ10 Franchise Consistency Results:\n", anova_df)

# ============================================================
# Bonus: Stress vs. Ratings for 'The Silence of the Lambs'
# ============================================================

stress_col = "My life is very stressful"
movie = 'The Silence of the Lambs (1991)'

groups = [df[df[stress_col] == i][movie].dropna() for i in range(1, 6)]
f_stat, p_val = stats.f_oneway(*groups)
means = [g.mean() for g in groups]
print(f"\nBonus: '{movie}' — ANOVA, p={p_val:.5f}, means={means}")

plt.bar(['Most', 'High', 'Moderate', 'Low', 'Least'], means, color='orange', edgecolor='black')
plt.title("Bonus: Stress vs. Movie Rating ('Silence of the Lambs')")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.savefig("QBonus_SilenceLambs_Stress.png", dpi=300)
plt.close()

print("\n✅ All analyses complete. Plots saved for Q1–Q10 + Bonus.\n")

