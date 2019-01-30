# Contrastive Multivariate Singular Spectrum Analysis

This repo contains the original implementation of **Contrastive Multivariate Singular Spectrum Analysis (cMSSA)** as outlined in this [paper](papers/neurips2018_workshop.pdf).

```python
model = CMSSA(alpha=1.0, window=100, num_comp=10)

# 1. Fit on foreground time series contrasted against a background dataset.
model.fit(X_fg, X_bg)

# 2. Decompose time series data with your fitted model.
R = model.transform(X, collapse=False)

# 3. Visualize the contrastive sub-signals.
plot_rcs(R)
```

![cMSSA](banner.png)
