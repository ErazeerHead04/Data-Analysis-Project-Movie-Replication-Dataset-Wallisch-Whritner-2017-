import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load and inspect the dataset
# -------------------------------
df = pd.read_csv("movieReplicationSet.csv")

# Define correct demographic column names based on your file
gender_col = 'Gender identity (1 = female; 2 = male; 3 = self-described)'
only_child_col = 'Are you an only child? (1: Yes; 0: No; -1: Did not respond)'
social_col = 'Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)'

# Define movie columns (first 400)
movie_cols = df.columns[:400]

# -------------------------------
# Data cleaning
# -------------------------------
# Keep only rows with valid gender entries (1=female, 2=male)
df = df[df[gender_col].isin([1, 2])]

# -------------------------------
# Q4: Gender differences across all movies
# -------------------------------
p_values_gender = []

for movie in movie_cols:
    male_ratings = df.loc[df[gender_col] == 2, movie].dropna()
    female_ratings = df.loc[df[gender_col] == 1, movie].dropna()

    # Ensure enough data points for both groups
    if len(male_ratings) > 10 and len(female_ratings) > 10:
        t_stat, p_val = stats.ttest_ind(male_ratings, female_ratings, equal_var=False, nan_policy='omit')
        p_values_gender.append(p_val)

# Calculate proportion of significant gender differences
significant_gender = sum(p < 0.005 for p in p_values_gender)
prop_gender_diff = significant_gender / len(p_values_gender)

print(f"Q4: {significant_gender} movies ({prop_gender_diff:.2%}) differ significantly by gender (α=0.005).")

plt.figure(figsize=(7, 5))
plt.hist(p_values_gender, bins=30, color='skyblue', edgecolor='black')
plt.axvline(0.005, color='red', linestyle='--', label='α = 0.005')
plt.title("Distribution of p-values across gender comparisons (Q4)")
plt.xlabel("p-value")
plt.ylabel("Number of Movies")
plt.legend()
plt.tight_layout()
plt.savefig("Q4_GenderDifferences_Histogram.png", dpi=300)
plt.close()

# -------------------------------
# Q5: Only child effect – 'The Lion King (1994)'
# -------------------------------
movie_lion_king = 'The Lion King (1994)'

only_ratings = df.loc[df[only_child_col] == 1, movie_lion_king].dropna()
sibling_ratings = df.loc[df[only_child_col] == 0, movie_lion_king].dropna()

if len(only_ratings) > 5 and len(sibling_ratings) > 5:
    t_lk, p_lk = stats.ttest_ind(only_ratings, sibling_ratings, equal_var=False, nan_policy='omit')
    mean_only = only_ratings.mean()
    mean_sibling = sibling_ratings.mean()
    print(f"Q5: 'The Lion King (1994)' — t={t_lk:.3f}, p={p_lk:.5f}, "
          f"mean(only)={mean_only:.2f}, mean(sibling)={mean_sibling:.2f}")
else:
    print("Q5: Not enough data for 'The Lion King (1994)' comparison.")

plt.figure(figsize=(5, 5))
plt.bar(['Only Child', 'Has Siblings'], [mean_only, mean_sibling],
        color=['lightcoral', 'lightblue'], edgecolor='black')
plt.title("Mean Ratings of 'The Lion King (1994)' by Only-Child Status (Q5)")
plt.ylabel("Mean Rating (0–4)")
plt.tight_layout()
plt.savefig("Q5_LionKing_OnlyChild.png", dpi=300)
plt.close()

# -------------------------------
# Q6: Only child effect across all movies
# -------------------------------
p_values_onlychild = []

for movie in movie_cols:
    only_ratings = df.loc[df[only_child_col] == 1, movie].dropna()
    sibling_ratings = df.loc[df[only_child_col] == 0, movie].dropna()

    if len(only_ratings) > 10 and len(sibling_ratings) > 10:
        t_stat, p_val = stats.ttest_ind(only_ratings, sibling_ratings, equal_var=False, nan_policy='omit')
        p_values_onlychild.append(p_val)

significant_onlychild = sum(p < 0.005 for p in p_values_onlychild)
prop_onlychild_diff = significant_onlychild / len(p_values_onlychild)

print(f"Q6: {significant_onlychild} movies ({prop_onlychild_diff:.2%}) show an 'only child' effect (α=0.005).")

plt.figure(figsize=(7, 5))
plt.hist(p_values_onlychild, bins=30, color='lightgreen', edgecolor='black')
plt.axvline(0.005, color='red', linestyle='--', label='α = 0.005')
plt.title("Distribution of p-values across Only-Child Comparisons (Q6)")
plt.xlabel("p-value")
plt.ylabel("Number of Movies")
plt.legend()
plt.tight_layout()
plt.savefig("Q6_OnlyChild_Histogram.png", dpi=300)
plt.close()

# -------------------------------
# Q7: Social watching preference – 'The Wolf of Wall Street (2013)'
# -------------------------------
movie_wolf = 'The Wolf of Wall Street (2013)'

# In the dataset, 1 = prefers alone, 0 = prefers socially
social_watchers = df.loc[df[social_col] == 0, movie_wolf].dropna()
solo_watchers = df.loc[df[social_col] == 1, movie_wolf].dropna()

if len(social_watchers) > 5 and len(solo_watchers) > 5:
    t_wolf, p_wolf = stats.ttest_ind(social_watchers, solo_watchers, equal_var=False, nan_policy='omit')
    mean_social = social_watchers.mean()
    mean_solo = solo_watchers.mean()
    print(f"Q7: 'The Wolf of Wall Street (2013)' — t={t_wolf:.3f}, p={p_wolf:.5f}, "
          f"mean(social)={mean_social:.2f}, mean(alone)={mean_solo:.2f}")
else:
    print("Q7: Not enough data for 'The Wolf of Wall Street (2013)' comparison.")

plt.figure(figsize=(5, 5))
plt.bar(['Prefers Socially', 'Prefers Alone'],
        [mean_social, mean_solo],
        color=['mediumseagreen', 'gold'], edgecolor='black')
plt.title("Mean Ratings of 'The Wolf of Wall Street (2013)' by Viewing Preference (Q7)")
plt.ylabel("Mean Rating (0–4)")
plt.tight_layout()
plt.savefig("Q7_WolfOfWallStreet_SocialPref.png", dpi=300)
plt.close()

print("\n✅ All plots saved:")
print("  - Q4_GenderDifferences_Histogram.png")
print("  - Q5_LionKing_OnlyChild.png")
print("  - Q6_OnlyChild_Histogram.png")
print("  - Q7_WolfOfWallStreet_SocialPref.png")