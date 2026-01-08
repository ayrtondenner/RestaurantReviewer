# RestaurantReviewer

## Suggestions for future work

Behavior- and engagement-focused analyses that can help understand how people interact with restaurants and how that relates to review outcomes:

- [x] Rating vs text length: compare `nota` against `review_len` and `title_len` (scatter/box) to see if longer reviews skew more positive/negative.
- [x] Ratings by “who with”: compare `nota` distributions across `em_companhia_de` (box/violin; optionally ANOVA/Kruskal).
- [x] Sponsorship effect: compare `nota` and `review_len` for `is_parceria_patrocinada=True` vs `False` (include effect size, not just mean).
- [ ] Image behavior: compare `nota` and `review_len` for `has_image=True` vs `False`, and for `images_count` buckets (0, 1–2, 3–5, 6+).
- [x] Contribution tiers: bucket `contribuicoes` into quantiles (low/med/high) and compare `nota`, `review_len`, and image posting rate.
- [ ] State-level differences: for `state`, compare average `nota`, % with images, and mean `review_len` (only for states with enough samples).
- [x] Time drift / seasonality in rating: track average `nota` over time (e.g., monthly) with confidence intervals.
- [x] Time vs engagement: compare `review_len` and `has_image` by `day_of_week` / `day_of_week_label` (weekday vs weekend effects).
- [x] Sub-score alignment: for rows with `nota_*`, compute gaps (e.g., `nota_comida - nota_custo`) and test which gaps best explain `nota`.
- [ ] Predicting `nota` (baseline model): train a simple model using `review_len`, `title_len`, `contribuicoes`, `images_count`, time fields, and available `nota_*`; inspect feature importances.
- [ ] Outlier review inspection: identify extreme cases (very long reviews, many images, very low ratings) and sample them for qualitative insight.
- [x] Topic exploration: TF‑IDF + clustering on `review` to surface recurring themes, then compare average `nota` per cluster.
- [x] Keyword ↔ rating association: for common tokens/bigrams, measure lift/association with low vs high ratings.
- [x] Sentiment proxy: lexicon-based sentiment on `review` (Portuguese lexicon if available) and correlate with `nota`.
- [ ] Missingness analysis: check if missing fields (e.g., `cidade_e_estado`, `nota_*`) correlate with `nota`, `review_len`, or engagement.
- [ ] Data quality checks: duplicates/near-duplicates, repeated titles, and date anomalies to validate conclusions.