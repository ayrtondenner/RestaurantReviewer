## `generate_reports.py` vs `analysis.ipynb`

Both work from the same canonical dataset ([dataframes/tripadvisor.csv](dataframes/tripadvisor.csv)), but they serve different goals.

### What each one is for

- **[analysis.ipynb](analysis.ipynb)** is an **exploratory analysis notebook**.
	- Designed for interactive work: plots, tables, correlations, and deeper text/embedding exploration.
	- Includes AI-assisted summaries (via `services/chatgpt_service.py`) and more advanced analyses (TF‑IDF topics, token association, sentiment proxy, embedding maps, outlier detection).

- **[generate_reports.py](generate_reports.py)** is a **batch report generator**.
	- Designed to produce a consistent, shareable narrative output.
	- Generates **two Markdown reports** (EN + PT-BR) into `reports/`.
	- Focuses on a fixed set of high-level metrics and “executive summary”-style insights.

### Key differences (important)

- **Outputs:**
	- Notebook → interactive outputs (tables/plots) inside Jupyter.
	- Script → saved files: [reports/en-us_report.md](reports/en-us_report.md) and [reports/pt-br_report.md](reports/pt-br_report.md).

- **Sentiment:**
	- Script uses a **rating-bucket proxy**: positive (4–5★), neutral (3★), negative (1–2★) and derives a simple **consensus** label.
	- Notebook uses **LLM summaries** + a transparent **lexicon-based sentiment proxy** (and also supports token/topic exploration).

- **Keywords/themes:**
	- Script extracts **top-N keywords** by raw frequency (regex tokenization + basic PT stopwords).
	- Notebook uses **word clouds**, **token association (high vs low rating)**, and **topic clustering (TF‑IDF + KMeans)**.

- **Reproducibility & convenience:**
	- Script is easier to run end-to-end and produces the same report structure every time.
	- Notebook is richer for investigation, but the “final story” is spread across many cells.

### Metric mapping checklist (report → notebook)

This is a practical mapping of what the report script computes and where the notebook covers the same idea.
Notebook references below are **cell numbers** in [analysis.ipynb](analysis.ipynb).

#### Dataset loading & derived columns

- Load CSV into `df` → Cell 2
- Create/normalize derived columns (`title_len`, `review_len`, `images_count`, `has_image`, `state`, date parts, etc.) → Cell 3

#### Dataset overview (script: `get_basic_stats`)

- Total reviews (`len(df)`) → Cell 6
- Average rating (`df['nota'].mean()`) → Cell 6
- Reviews with images + % with images → Cell 6 and Cell 40
- Average title/review length → Cell 32–34 (distributions + mean/median; not a single “basic stats” block)
- Date range (min/max `data_postagem`) → not explicitly computed in the notebook
- Total columns (`len(df.columns)`) → not explicitly shown as a metric

#### Rating distributions

- `nota` distribution (counts + %) → Cell 18
- Sub-score availability (% coverage of `nota_custo/atendimento/comida/ambiente`) → Cell 19
- Sub-score grade distributions → Cell 20

#### Geography

- Reviews per state (script also truncates to top 10) → Cell 16

#### Categories / “who with” (`em_companhia_de`)

- Category distribution (counts + %) → Cell 22
- Rating by category (avg/median + boxplot) → Cell 44

#### Sponsorship (`is_parceria_patrocinada`)

- Sponsored vs non-sponsored counts + % → Cell 23
- Sponsored vs non-sponsored comparisons (rating, review length, % with images) → Cell 45

#### Contributions (`contribuicoes`)

- Distribution + summary stats → Cell 25
- Tiers vs ratings/behavior:
	- Notebook uses quantiles (`qcut`) → Cell 46
	- Script uses fixed bins `[0,10,50,200,inf]` → not present with the same binning

#### Temporal patterns

- Reviews by year/month/day-of-week/day-of-month → Cells 27–30
- Average rating by year:
	- Script computes per-year mean → not directly
	- Notebook shows monthly average rating over time → Cell 47
- Weekday vs weekend patterns:
	- Script computes weekday vs weekend avg rating (via an `is_weekday` flag)
	- Notebook computes weekend vs weekday behavior (including avg rating) → Cell 48

#### Rating relationships (script: `get_rating_analysis`)

- Avg review length by rating → closest: Cell 43 (regplots + binning)
- % reviews with images by rating → not explicitly present
- Avg rating by company → Cell 44
- Avg rating by contribution tier → Cell 46 (but tiering differs)

#### Keywords & themes

- Script: top 30 tokens with counts → not present in that exact format
- Notebook equivalents:
	- Word clouds → Cells 35–36
	- Topic clustering (TF‑IDF + KMeans) → Cell 52
	- Token association high vs low rating (log-odds + counts) → Cell 53

#### Sentiment patterns

- Script: rating-bucket sentiment counts + consensus label → not explicitly present
- Notebook alternatives:
	- LLM summaries per-year + whole dataset → Cells 8–14
	- Lexicon sentiment proxy correlated with `nota` → Cell 54

#### Extra analyses in the notebook (not in the report script)

- Correlation matrix + ranked correlations vs `nota` → Cell 38
- Image count distribution (`images_count`) → Cell 40
- Sub-score alignment vs overall rating (`subscore_mean`, gaps) → Cell 49
- Embedding maps (PCA/SVD/t‑SNE/NMF) + outlier detection workflows → Cells 56–70

### When to use which

- Use **[analysis.ipynb](analysis.ipynb)** when you want to explore hypotheses, inspect distributions visually, debug data issues, or go deeper on text/embeddings.
- Use **[generate_reports.py](generate_reports.py)** when you want a consistent, shareable summary in Markdown (EN + PT-BR) with the same sections every run.