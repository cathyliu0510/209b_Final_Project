# Results Interpretation and Analysis

This write-up summarizes how the current baseline models perform, what those results imply for the project question, and what the next modeling iteration should prioritize.

## 1. Baseline Performance Summary

The baseline-model notebook now compares four models for forecasting `employment_thousands_growth`:

- Linear Regression with metro fixed effects
- Ridge Regression on an expanded lagged panel
- Gradient Boosting Regressor
- LSTM

Two evaluation lenses matter in this project:

1. **Rolling-origin cross-validation**, which is used as the pre-specified model-selection rule.
2. **Official validation and held-out test performance**, which show how the models perform on the final train/validation/test split.

Under the rolling-origin tuning criterion, **Ridge Regression** is the selected reporting baseline. Its average rolling-CV MAE is about **0.848**, slightly better than **Gradient Boosting** at about **0.856**. This makes Ridge the most defensible baseline under the stated selection rule because it is both competitive and still interpretable on a small lagged panel with strong collinearity.

However, the official validation and test results show a more nuanced ranking. After refitting on the full training window, **Gradient Boosting** performs best on the official validation split with **MAE about 0.617**. On the held-out `2021-2023` test period, it also gives the strongest pooled performance in the current comparison, with approximately **R^2 = 0.167** and **MAE = 1.944**. By comparison, tuned **Linear Regression** reaches roughly **R^2 = 0.142** and **MAE = 2.100**, while tuned **Ridge Regression** reaches about **R^2 = 0.115** and **MAE = 2.007**. The newly added **LSTM** serves as the AC209b sequential baseline, but it does not improve the overall picture: it reaches **validation MAE about 0.612**, yet its held-out test performance is notably weaker at **MAE about 2.349** with **test R^2 about -0.832**.

This means the current baseline story is not that one model dominates under every metric. Instead, the baseline results show that:

- **Ridge** is the best model under the project's pre-committed tuning rule.
- **Gradient Boosting** is the strongest holdout performer.
- **Linear Regression** remains a useful transparent benchmark, but it is not the top performer.
- **LSTM** satisfies the requirement to include a sequential AC209b-style model, but its weaker held-out performance does not change the main conclusion.

## 2. What the Visualizations Show

The current figures support three main conclusions.

### 2.1 Model ranking depends on the evaluation frame

`figures/08_rolling_validation_stability.png` shows why the rolling-origin framework matters. A single validation year is too unstable to support model choice on its own. The notebook also shows that a naive persistence rule looks deceptively strong on `2019` with **MAE about 0.272**, but then deteriorates badly on the true held-out period with **MAE about 3.645**. That contrast justifies using rolling-origin validation rather than choosing a model from one favorable split.

`figures/07_baseline_model_comparison.png` then shows the final tuned comparison. The key takeaway is that the model chosen by the pre-specified tuning rule is not identical to the model with the best official holdout metrics. That is an important and honest result: on a small panel, model ranking can change depending on whether we value cross-fold stability or final pooled holdout performance more heavily. The updated figure also shows that although **LSTM** is competitive on the validation split, it does not generalize well to the held-out test years, which is consistent with overfitting or instability on a small panel.

### 2.2 The forecasting problem is easier in some years than others

`figures/10_benchmark_yearwise_performance.png` shows that **2021 is the hardest held-out year**, while `2022` and especially `2023` are easier for the selected benchmark to track. This is substantively plausible. The post-2020 recovery period is noisier and less stable, so a model trained on pre-2020 data faces a tougher extrapolation problem in `2021` than in later years.

The figure also clarifies a metric nuance. The pooled test `R^2` can be positive even if within-year `R^2` is weak or negative, because pooled evaluation captures broader variation across the full `2021-2023` test window, whereas within-year prediction only uses the much narrower spread across metros inside a single year.

### 2.3 Most predictive signal still comes from lagged economic context

`figures/09_baseline_feature_importance.png` indicates that much of the current predictive power comes from **lagged economic-growth and level features**, with a smaller but still meaningful contribution from **lagged satellite summaries**. This is consistent with the EDA: raw satellite summaries contain some temporal signal, but they do not fully capture the urban-development mechanisms that likely drive future economic outcomes.

## 3. Strengths of the Current Baseline

The current baseline has several clear strengths.

- It uses a **time-aware evaluation design**, which is much more appropriate than random cross-validation for a forecasting problem.
- It keeps an **interpretable linear benchmark** while also testing a reasonable nonlinear alternative.
- It now includes a **sequential LSTM baseline**, so the project compares tabular and temporal model classes rather than relying on only one modeling family.
- It directly reflects the EDA finding that **within-metro temporal dynamics matter more than pooled cross-city relationships**.
- It shows that **regularization is useful** when the feature space contains many lagged and correlated predictors.
- It provides a transparent foundation that later feature-engineering or deep-learning stages can be judged against.

These are important strengths because a baseline is not only supposed to perform reasonably well. It is also supposed to make the later project stages easier to evaluate credibly.

## 4. Weaknesses and Current Limitations

The current baseline also has important weaknesses that limit what we can conclude.

- The panel is **small**, so metric estimates are noisy and model rankings can change across splits.
- The added **LSTM does not perform well on the held-out test set**, which suggests that a sequential deep-learning model may be too data-hungry for the current sample size.
- The baseline currently focuses on **one target outcome**, `employment_thousands_growth`, rather than the broader project outcomes such as GDP growth and building permits.
- The satellite inputs are still **raw summary statistics**, which are only indirect proxies for urban structure and expansion.
- The models do not explicitly encode **spatial form**, such as compactness, infill, sprawl, fragmentation, or built-up area growth.
- The exclusion of `2020` is justified, but it also makes the post-COVID test window harder to interpret because `2021` may still contain recovery effects not captured by the pre-2020 training data.

Taken together, these limitations explain why the current baseline should be treated as a useful reference point rather than a final scientific model.

## 5. Improvements for the Next Iteration

The next modeling iteration should improve both the features and the evaluation design.

First, the project should replace or augment raw pixel summaries with **GHSL-derived built-up features**. The EDA and planning documents already suggest that built-up growth is more informative than raw pooled pixel statistics. Features such as built-up area, built-up fraction, compactness, edge density, and infill-versus-sprawl ratios should give the model variables that are much closer to the underlying urban-expansion process.

Second, the team should run **feature-set ablations**. A clean comparison across raw-only, spatial-only, combined, and spatial-plus-economic-lag feature sets would show whether the new spatial representation adds signal beyond the current baseline.

Third, the same benchmark structure should be extended to **additional targets**, especially GDP growth and building permits. This matters because the substantive project question is broader than employment alone.

Fourth, the project should add **sensitivity checks** such as leave-one-metro-out analysis or region-based robustness checks. With a small panel, it is important to verify that the current results are not being driven by a few influential metros.

## 6. Future Directions for the Final Project

For the final project, the most important next step is to move from raw imagery summaries to a more structural measure of urban change.

One path already laid out in the project plan is to train a **U-Net segmentation model** on GHSL built-up masks and MODIS imagery, then use the predicted masks to engineer interpretable urban-form features. This would let the team test whether direct measures of expansion and spatial organization outperform raw pixel summaries. If they do, that would strengthen the scientific claim that satellite-observed urban development helps predict future economic activity.

Another future direction is to keep comparing the fixed-effects baseline against **sequence models** such as an LSTM or GRU, but in a more disciplined way. The current LSTM result shows that simply adding a sequential model is not enough; on this dataset it does not outperform the stronger tabular baselines and appears to generalize poorly. That means any future deep-learning extension should be paired with stronger regularization, careful architecture control, and likely more data before it can be expected to beat Ridge or Gradient Boosting.

The project should also broaden its scope by extending the panel to **more metros and a longer time horizon**. That would improve statistical stability, test whether the current patterns generalize beyond the present sample, and help distinguish temporary post-COVID effects from more persistent urban-economic relationships.

Overall, the current baseline results support a clear final-project strategy: keep the time-aware panel framework, preserve Ridge as a defensible reporting baseline, treat Gradient Boosting as a strong nonlinear comparator, and focus the next iteration on richer spatial features that are more closely tied to actual urban expansion.

## 7. Bottom Line

The baseline models show that lagged economic dynamics already carry substantial predictive signal, while lagged raw satellite summaries add some value but remain incomplete. Ridge Regression is the most defensible reporting baseline under the pre-specified rolling-CV rule, while Gradient Boosting is the strongest holdout performer. The added LSTM baseline is useful as an AC209b comparison, but because it does not improve held-out performance, it does not materially change the project's conclusions. The main lesson is not that the project is finished, but that the current baseline successfully establishes a credible benchmark and also makes clear why the final project should prioritize richer spatial features, broader target coverage, and stronger robustness checks.
