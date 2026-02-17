# CompNeuro — Neural & Behavioral Analysis of Visual Decision-Making in ASD vs. WT Mice

A three-script Python pipeline for extracting and analyzing LFP and behavioral data from the **International Brain Laboratory (IBL)** visual decision-making task, comparing **ASD model mice** against **wildtype (WT)** controls. The pipeline covers raw data extraction from the IBL database, sliding-window GLM-based strategy clustering, and psychometric function (PMF) analysis.

---

## Method Overview

### CompNeuro-DataExtract.py — LFP & Behavioral Feature Extraction

Connects to the IBL Alyx database via `ONE` and streams raw data for all probe insertions targeting **visual cortex (VIS)** channels. For each session and trial, it extracts:

- **Spike counts** around stimulus onset
- **ROI motion energy** and **pupil diameter** from the left camera (DeepLabCut)
- **Behavioral trial variables**: reaction time, contrast difference, prior probability, feedback type, choice
- **LFP band power** (delta, theta, alpha, beta, gamma) in three time windows — baseline (−1 to 0 s pre-stimulus), decision (stimulus onset to response), and feedback (0 to +1 s post-feedback) — computed via Welch's method on VIS-channel LFP

All output is saved as a timestamped `.csv` with one row per trial across all sessions.

---

### LocalLinearGLM.py — Sliding-Window Strategy Clustering

Takes the behavioral trial data and models moment-to-moment decision strategy using a **sliding-window logistic regression (GLM)** with window size = 100 trials and step size = 20 trials. Each window produces three key coefficients:

- **β^contrast** — sensitivity to visual contrast
- **β^prior** — use of an exponential history-weighted prior
- **β^WSLS** — win-stay/lose-switch tendency (choice × feedback interaction)

Windows are then **median-split clustered** across all three coefficients simultaneously into **8 behavioral states (C1–C8)**, where C1 (high prior, high contrast, high WSLS) is the most optimal and C8 (low on all three) is the least. The pipeline then tracks:

- State occupancy and transitions over 20 quantile bins (each = 5% of session progress)
- Self-transition probabilities and group-level differences
- **Coefficient drift** at experimental block switches, using three complementary methods: incremental (expanding window), non-overlapping windows, and fixed-baseline comparison — each with both mean and total aggregation

---

### PMF&States.py — Psychometric Function Analysis

Uses the window-level state classifications from NeuroFn_GLM to assign behavioral states to individual trials and compute a **psychometric function (PMF)** for each of the 8 states. For each state, P(choose right) is plotted as a function of signed contrast level (−1 to +1), with empirical 95% CIs.

States are then **ranked by a combined distance score** from the perfect PMF:

```
S = d_Euclidean + d_MAE + 3 × |Acc@0 − 1.0|
```

where `Acc@0` is zero-contrast accuracy. Additional analyses include performance score trajectories across 200-trial epochs, state transition performance deltas (e.g. ΔS for C8→C1), and late-training regression events.

---

## File Structure

```
CompNeuro/
│
├── CompNeuro-DataExtract.py   # IBL data extraction: LFP band power + behavioral features → CSV
├── LocalLinearGLM.py          # Sliding-window GLM, 8-state clustering, drift analysis
├── PMF&States.py              # PMF by state, distance ranking, trajectory statistics
│
├── data/
│   ├── 82ASD_P2_BlockAdded.csv     # ASD mouse behavioral data (input)
│   └── 78BWM_P2_BlockAdded.csv     # WT mouse behavioral data (input)
│
├── outputs/             # All generated CSVs, PDFs, and .txt statistics files
│
└── README.md
```

> **Note:** Raw IBL LFP data is streamed directly from the IBL server and is not stored locally. Behavioral `.csv` files are not included in the repository — contact the authors for access.

---

## Dependencies

```bash
pip install one-api brainbox numpy scipy pandas matplotlib seaborn scikit-learn
```

- `one-api` — IBL ONE interface for remote data access
- `brainbox` — IBL spike sorting loader
- `scikit-learn` — logistic regression
- `scipy` — Welch PSD, statistical tests
- Tested on **Python ≥ 3.9**

---

## How to Run

### Step 1 — Extract LFP and behavioral features (`CompNeuro-DataExtract.py`)

```bash
python CompNeuro-DataExtract.py
```

This script connects to the public IBL server (no credentials needed beyond the defaults), searches for all probe insertions in VIS regions, and processes each session. Configure at the top of the script:

```python
# Remote IBL public database (default)
one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          username='intbrainlab',
          password='international')

# Filter to VIS insertions
pids = one.search_insertions(atlas_acronym=['VIS'], query_type='remote')
```

**Output:** `VIS_WT_LFP_data_YYYYMMDD_HHMMSS.csv` — one row per trial, 27 columns covering behavioral variables and LFP band powers across three time windows.

> Memory note: LFP data is loaded and deleted per-session using `gc.collect()`. For large PID lists, expect long runtimes. Limit with `pids = pids[:N]` for testing.

---

### Step 2 — GLM clustering and drift analysis (`LocalLinearGLM.py`)

```bash
python LocalLinearGLM.py
```

Point to your behavioral data files at the bottom of the script:

```python
df1 = pd.read_csv("path/to/82ASD_P2_BlockAdded.csv")   # ASD group
df2 = pd.read_csv("path/to/78BWM_P2_BlockAdded.csv")   # WT group

results = run_full_analysis_part1(df1, df2, proportion=1, output_dir='path/to/outputs')
```

Set `proportion < 1.0` to subsample subjects for faster testing (e.g. `proportion=0.3`).

**Outputs saved to `output_dir/`:**

| File | Contents |
|---|---|
| `all_windows_data.csv` | All sliding-window results with cluster assignments |
| `key_statistics.txt` | Clustering validation, group comparisons, temporal evolution |
| `drift_comparison_full_data.csv` | Per-switch drift metrics across all 3 methods |
| `drift_comparison_statistics.txt` | Effect sizes, correlations, Cohen's d |
| `state_transitions.pdf` | Transition matrices (ASD, WT, difference) |
| `strategy_evolution.pdf` | Strategy and accuracy evolution over 20 quantiles |
| `correlation_analysis.pdf` | High-prior usage vs. zero-contrast accuracy |
| `drift_method_comparison.pdf` | Drift-accuracy scatter across 6 method variants |
| `drift_group_comparison.pdf` | Group boxplots for all drift methods |
| `drift_coefficient_comparison.pdf` | Per-coefficient (contrast / prior / WSLS) drift |
| `drift_distributions.pdf` | Drift distribution histograms by group |

---

### Step 3 — PMF analysis (`PMF&States.py`)

```bash
python "PMF&States.py"
```

If running interactively after Step 2, pass the results object directly:

```python
pmf_results = run_pmf_analysis_part2(
    results['windows_df'],
    results['df_features'],
    output_dir='path/to/outputs'
)
```

If running standalone (Step 2 outputs already saved), the script auto-loads `all_windows_data.csv` and regenerates features from the raw behavioral CSVs. Set `output_dir` and data paths at the bottom of `PMF&States.py`.

**Outputs saved to `output_dir/`:**

| File | Contents |
|---|---|
| `pmf_C1.csv` … `pmf_C8.csv` | PMF data per state (contrast level, P(right), CI) |
| `pmf_state_ranking.csv` | State ranking by combined score S |
| `pmf_manuscript_statistics.txt` | All quantile, occupancy, trajectory, and delta stats |
| `pmf_all_states.pdf` | Individual PMF curve per state (ranked) |
| `pmf_combined.pdf` | All states vs. perfect PMF overlay |

---

## Notes

- **Column name flexibility:** `LocalLinearGLM.py` auto-detects subject ID columns (`subject`, `subjectID`, `mouse`, `animal`, `ID`) and renames them to `subject_id`. If your data uses a different column name, add it to the detection list in `run_full_analysis_part1()`.
- **Zero-contrast accuracy** is used throughout as the primary performance metric — it captures unbiased choice behavior independent of contrast-driven responding.
- **Block switch detection** requires a `Block` column in the behavioral data. If absent, drift analysis is skipped gracefully.
- All PDF figures are saved at 300 dpi via `matplotlib`. Output directory defaults to `D:/Neurofn` — change the `output_dir` argument as needed.
- **Shell note:** Because `PMF&States.py` contains `&` in its filename, wrap it in quotes when calling from the terminal: `python "PMF&States.py"`.
