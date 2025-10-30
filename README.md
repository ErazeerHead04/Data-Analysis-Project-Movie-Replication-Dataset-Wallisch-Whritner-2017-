# Data Analysis Project — Movie Replication Dataset (Wallisch & Whritner, 2017)
 

---

## 📘 Project Overview

This project explores patterns in movie ratings using the **Movie Replication Dataset** (Wallisch & Whritner, 2017).  
The dataset contains ratings for 400 popular films from several hundred participants, along with demographic and behavioral survey responses.  

Our goal is to identify **factors that influence movie enjoyment** and explore whether individual differences (e.g., gender, personality traits, viewing habits, or social preferences) predict variation in movie ratings.

Each team member is responsible for a subset of research questions, but all analyses follow the same data-driven workflow and statistical framework.

---

## Data Description

The dataset includes:

- **Movie Ratings:** Ratings for 400 films (columns 1–400) on a 0–4 scale  
- **Demographic Variables:**  
  - Gender identity  
  - Only-child status  
  - Viewing preferences (alone vs. socially)  
  - Other personality and behavioral measures  

Each **row** represents one participant, and each **column** represents either a movie rating or a demographic/survey variable.

---

## Methodology

All analyses use the **Welch’s independent-samples t-test** or related inferential methods, depending on the comparison being made.  
This test was chosen because it:

- Compares the means of two independent groups  
- Does **not assume equal variances or equal sample sizes**  
- Is robust for real-world, unbalanced datasets  
- Is appropriate for continuous or approximately continuous rating data  

The significance threshold was set to **α = 0.005**, following the project’s guidelines.

---

## Project Workflow

1. **Data Import:**  
   The dataset (`movieReplicationSet.csv`) is loaded using pandas.

2. **Data Cleaning:**  
   Missing values and invalid demographic responses are excluded to ensure valid comparisons.

3. **Exploratory Analysis:**  
   Group-based descriptive statistics and visualizations (histograms, bar charts) are used to explore distributions and trends.

4. **Inferential Analysis:**  
   Statistical tests are conducted to assess group differences in movie ratings.

5. **Visualization:**  
   Results are visualized using Matplotlib (e.g., bar charts, histograms, significance markers).

6. **Reporting:**  
   Findings are reported using the AFYD framework:
   - **D:** Do — What was done  
   - **Y:** Why — Why the method was chosen  
   - **F:** Find — Key results  
   - **A:** Answer — Interpretation of findings

---

## Visual Outputs

Each analysis script generates visual summaries such as:

- **Histograms** of p-value distributions across comparisons  
- **Bar charts** comparing mean ratings between participant groups  
- **Saved PNG images** for inclusion in the written report or presentation  

These figures are automatically saved in the project directory.

---

## Interpretation Guidelines

The statistical results should be interpreted as follows:

- **p < 0.005:** Significant difference between groups → potential systematic effect  
- **p ≥ 0.005:** No significant difference → likely random variation  

When interpreting findings, focus on **effect direction** (which group rated higher) and **practical significance**, not just statistical significance.

---

## Dependencies

To run the analyses, install the following Python libraries:

```bash
pip install pandas scipy numpy matplotlib
```

---

## How to Run

1. Clone or download this repository.  
2. Ensure the dataset file `movieReplicationSet.csv` is in the same folder as the script (e.g., `test.py`).  
3. Run the analysis script:
   ```bash
   python test.py
   ```
4. Check the console for printed results and the working directory for generated plot images.

---


## 📚 Reference

Wallisch, P., & Whritner, J. (2017). *Movie Preferences and Personality: A Large-Scale Replication Study.*  
New York University, Department of Psychology.

---

## Notes

- All team members contribute to different subsets of questions but use the **same statistical pipeline** for consistency.  
- Analyses emphasize reproducibility, transparency, and correct statistical interpretation.  
- Figures and tables are incorporated into the AFYD-style final report.

---

## Summary

This repository provides a **reproducible workflow** for analyzing group differences in movie preferences using the Wallisch & Whritner (2017) dataset.  
By combining statistical rigor with clear visualizations, the project demonstrates how demographic and behavioral traits relate to patterns of media enjoyment.
