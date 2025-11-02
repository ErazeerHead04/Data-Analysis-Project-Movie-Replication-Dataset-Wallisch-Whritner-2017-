import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Load Dataset
# ============================================================
df = pd.read_csv("movieReplicationSet.csv")

# Define demographic column names
gender_col = 'Gender identity (1 = female; 2 = male; 3 = self-described)'
only_child_col = 'Are you an only child? (1: Yes; 0: No; -1: Did not respond)'
social_col = 'Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)'

# Define movie columns (first 400)
movie_cols = df.columns[:400]

# ============================================================
# Q1 — Popular vs. Unpopular Movies
# ============================================================

# Compute number of ratings (non-NaN) for each movie
popularity = df[movie_cols].notna().sum()

# Compute mean rating per movie
mean_ratings = df[movie_cols].mean()

# Split movies into high vs low popularity using median split
median_popularity = popularity.median()
popular_movies = mean_ratings[popularity >= median_popularity]
unpopular_movies = mean_ratings[popularity < median_popularity]

# Welch’s t-test: are popular movies rated higher?
t_pop, p_pop = stats.ttest_ind(popular_movies, unpopular_movies, equal_var=False)
mean_popular = popular_movies.mean()
mean_unpopular = unpopular_movies.mean()

print(f"Q1: Popular vs. Unpopular Movies — t={t_pop:.3f}, p={p_pop:.5f}, "
      f"mean(popular)={mean_popular:.2f}, mean(unpopular)={mean_unpopular:.2f}")

# Plot
plt.figure(figsize=(5, 5))
plt.bar(['Popular', 'Unpopular'], [mean_popular, mean_unpopular],
        color=['mediumseagreen', 'lightcoral'], edgecolor='black')
plt.title("Mean Ratings: Popular vs. Unpopular Movies (Q1)")
plt.ylabel("Mean Rating (0–4)")
plt.tight_layout()
plt.savefig("Q1_PopularityComparison.png", dpi=300)
plt.close()

# ============================================================
# Q2 — New vs. Old Movies
# ============================================================

# Extract years from column names (if present)
import re
years = []
for col in movie_cols:
    match = re.search(r'\((\d{4})\)', col)
    if match:
        years.append(int(match.group(1)))
    else:
        years.append(np.nan)

years = pd.Series(years, index=movie_cols)
mean_ratings_by_movie = df[movie_cols].mean()

# Remove movies without year info
valid_movies = years.dropna().index
valid_years = years.dropna()
valid_means = mean_ratings_by_movie[valid_movies]

# Median split by year
median_year = valid_years.median()
new_movies = valid_means[valid_years >= median_year]
old_movies = valid_means[valid_years < median_year]

# Welch’s t-test
t_year, p_year = stats.ttest_ind(new_movies, old_movies, equal_var=False)
mean_new = new_movies.mean()
mean_old = old_movies.mean()

print(f"Q2: New vs. Old Movies — t={t_year:.3f}, p={p_year:.5f}, "
      f"mean(new)={mean_new:.2f}, mean(old)={mean_old:.2f}")

# Plot
plt.figure(figsize=(5, 5))
plt.bar(['New', 'Old'], [mean_new, mean_old],
        color=['steelblue', 'lightgray'], edgecolor='black')
plt.title("Mean Ratings: New vs. Old Movies (Q2)")
plt.ylabel("Mean Rating (0–4)")
plt.tight_layout()
plt.savefig("Q2_NewOldMovies.png", dpi=300)
plt.close()

# ============================================================
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
# ============================================================
df = df[df[gender_col].isin([1, 2])]  # clean gender column
p_values_gender = []

for movie in movie_cols:
    male_ratings = df.loc[df[gender_col] == 2, movie].dropna()
    female_ratings = df.loc[df[gender_col] == 1, movie].dropna()
    if len(male_ratings) > 10 and len(female_ratings) > 10:
        t_stat, p_val = stats.ttest_ind(male_ratings, female_ratings, equal_var=False)
        p_values_gender.append(p_val)

significant_gender = sum(p < 0.005 for p in p_values_gender)
prop_gender_diff = significant_gender / len(p_values_gender)
print(f"Q4: {significant_gender} movies ({prop_gender_diff:.2%}) differ significantly by gender (α=0.005).")

plt.figure(figsize=(7, 5))
plt.hist(p_values_gender, bins=30, color='skyblue', edgecolor='black')
plt.axvline(0.005, color='red', linestyle='--', label='α = 0.005')
plt.title("Distribution of p-values across Gender Comparisons (Q4)")
plt.xlabel("p-value")
plt.ylabel("Number of Movies")
plt.legend()
plt.tight_layout()
plt.savefig("Q4_GenderDifferences_Histogram.png", dpi=300)
plt.close()

# ============================================================
# Q5 — Only Child vs. Has Siblings for 'The Lion King (1994)'
# ============================================================
movie_lion_king = 'The Lion King (1994)'
only_ratings = df.loc[df[only_child_col] == 1, movie_lion_king].dropna()
sibling_ratings = df.loc[df[only_child_col] == 0, movie_lion_king].dropna()

if len(only_ratings) > 5 and len(sibling_ratings) > 5:
    t_lk, p_lk = stats.ttest_ind(only_ratings, sibling_ratings, equal_var=False)
    mean_only = only_ratings.mean()
    mean_sibling = sibling_ratings.mean()
    print(f"Q5: 'The Lion King (1994)' — t={t_lk:.3f}, p={p_lk:.5f}, "
          f"mean(only)={mean_only:.2f}, mean(sibling)={mean_sibling:.2f}")

    plt.figure(figsize=(5, 5))
    plt.bar(['Only Child', 'Has Siblings'], [mean_only, mean_sibling],
            color=['lightcoral', 'lightblue'], edgecolor='black')
    plt.title("Mean Ratings of 'The Lion King (1994)' (Q5)")
    plt.ylabel("Mean Rating (0–4)")
    plt.tight_layout()
    plt.savefig("Q5_LionKing_OnlyChild.png", dpi=300)
    plt.close()

# ============================================================
# Q6 — Only Child Effects Across All Movies
# ============================================================
p_values_onlychild = []
for movie in movie_cols:
    only_ratings = df.loc[df[only_child_col] == 1, movie].dropna()
    sibling_ratings = df.loc[df[only_child_col] == 0, movie].dropna()
    if len(only_ratings) > 10 and len(sibling_ratings) > 10:
        t_stat, p_val = stats.ttest_ind(only_ratings, sibling_ratings, equal_var=False)
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

# ============================================================
# Q7 — Social Watching Preference for 'The Wolf of Wall Street (2013)'
# ============================================================
movie_wolf = 'The Wolf of Wall Street (2013)'
social_watchers = df.loc[df[social_col] == 0, movie_wolf].dropna()
solo_watchers = df.loc[df[social_col] == 1, movie_wolf].dropna()

if len(social_watchers) > 5 and len(solo_watchers) > 5:
    t_wolf, p_wolf = stats.ttest_ind(social_watchers, solo_watchers, equal_var=False)
    mean_social = social_watchers.mean()
    mean_solo = solo_watchers.mean()
    print(f"Q7: 'The Wolf of Wall Street (2013)' — t={t_wolf:.3f}, p={p_wolf:.5f}, "
          f"mean(social)={mean_social:.2f}, mean(alone)={mean_solo:.2f}")

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





