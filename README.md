# RestaurantReviewer

End-to-end mini-pipeline to **collect Brazilian restaurant reviews from TripAdvisor (PT-BR)**, **normalize + enrich** them into a structured dataset, and then **analyze** the dataset with statistical plots and text-mining.

At a high level:

1) We collect review cards from locally saved TripAdvisor pages.
2) We parse/normalize fields (dates, ratings, categories, locations) and enrich with AI-powered representations (embeddings + 2D projections).
3) We explore the dataset in a notebook: distributions, correlations, seasonality, topics/keywords, and outliers.

## Install

This repo uses Conda via [environment.yml](environment.yml).

```bash
conda env create -f environment.yml
conda activate restaurantreviewer
```

If you will run embeddings / LLM calls, create a `.env` file with:

```env
OPENAI_API_KEY=...your-key...
```

Check [.env.example](.env.example) for example.

## Key files (pipeline)

- [1-extract-data.py](1-extract-data.py) (main)
	- Purpose: load each HTML page under `full_page/tripadvisor/` in a browser (Selenium) and extract each **review card** (`data-automation="reviewCard"`).
	- Outcome: saves individual cards to `raw_data/tripadvisor/card_<page>_<idx>.html` (pretty-indented for debugging).

- [2-normalize-and-enrich.py](2-normalize-and-enrich.py) (main)
	- Purpose: parse the raw card HTML files, normalize to a consistent schema, and enrich with AI features (text embeddings + dimensionality reduction columns).
	- Outcome: writes the final dataset CSV.

- [dataframes/tripadvisor.csv](dataframes/tripadvisor.csv) (main)
	- Purpose: the **canonical dataset** produced by the pipeline.
	- Outcome: 1 row per review with normalized fields (ratings, text, dates, location, etc.) plus enrichment columns.

- [analysis.ipynb](analysis.ipynb) (main)
	- Purpose: statistical + textual analysis of the dataset.
	- Outcome: plots and tables for review behavior, seasonality, sponsorship effects, topic discovery, keyword associations, sentiment proxy, dimensionality reduction visualizations, and outlier detection.

## Data flow

1) Download full page content to `full_page/tripadvisor/*.html`
2) [1-extract-data.py](1-extract-data.py): extract HTML cards
3) [2-normalize-and-enrich.py](2-normalize-and-enrich.py): `raw_data/tripadvisor/card_*.html`: normalize and enrich reviews
3) [dataframes/tripadvisor.csv](dataframes/tripadvisor.csv): dataframe created
4) [analysis.ipynb](analysis.ipynb): present data analysis

## Run the pipeline

1) Put the TripAdvisor pages you saved locally into `full_page/tripadvisor/`.

2) Extract raw review-card HTML:

```bash
python 1-extract-data.py
```

3) Normalize + enrich into a dataset:

```bash
python 2-normalize-and-enrich.py
```

4) Open [analysis.ipynb](analysis.ipynb) and run all cells.

## Libraries used

### Core scraping/parsing:

- `selenium`, `webdriver-manager` for browser automation and stable local runs.
- `beautifulsoup4` + `lxml` for HTML parsing.
- `tqdm` for progress bars.

### Data + visualization:

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `wordcloud`

### Machine learning / text analytics:

- `scikit-learn` for:
	- dimensionality reduction (`PCA`, `TruncatedSVD`, `t-SNE`, `NMF`)
	- topic discovery & clustering (`TfidfVectorizer`, `KMeans`)
	- outlier detection (e.g., `LocalOutlierFactor`, `IsolationForest`, etc. in the notebook)

### AI / LLM integration:

- Embeddings: the pipeline builds **text embeddings** for titles and reviews, then projects them into 2D with PCA/SVD/t-SNE to visualize structure and detect outliers.
	- The current implementation in [`services/embeddings_service.py`](services/embeddings_service.py) calls OpenAI embeddings (defaults to `text-embedding-3-large`).
- LLM summaries: [`services/chatgpt_service.py`](services/chatgpt_service.py) provides a helper to generate a PT-BR sentiment summary of a list of reviews.

## Results (current dataset)

Numbers below refer to the dataset currently saved at [dataframes/tripadvisor.csv](dataframes/tripadvisor.csv):

- Total reviews: **75** rows / **29** columns
- Date range: **2022-01-17** → **2025-12-29**
- Ratings distribution:
	- 1★: 7
	- 2★: 9
	- 3★: 10
	- 4★: 14
	- 5★: 35
- Sponsored reviews (`is_parceria_patrocinada=True`): **5** (**6.67%**)
- Reviews with at least one image: **33** (**44%**)
	- Mean images per review: **1.867** (max **13**)
- Top states by volume (UF parsed from `cidade_e_estado`): **SP (28)**, **MG (9)**, **RJ (7)**, **RS (4)**, **PE (3)**, **BA (2)**

## Notes

- This project assumes the TripAdvisor pages are already saved locally (under `full_page/tripadvisor/`).
- Please respect TripAdvisor’s terms and applicable laws when collecting data.

<!-- ## Suggestions for future work

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
- [ ] Data quality checks: duplicates/near-duplicates, repeated titles, and date anomalies to validate conclusions. -->