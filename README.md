# CausalPanel

*CausalPanel* is a Python package designed for **causal inference in panel data settings**, providing implementations of modern Difference-in-Differences (DiD) estimators and flexible tools for generating analysis-ready treatment windows. This toolkit streamlines causal analysis for researchers working with panel structures, supporting both standard and advanced methodologies.

---

## 📦 Key Components

1. **CSDID**  
   Implementation of the Difference-in-Differences method by **Callaway and Sant' Anna**, ideal for treatment effect estimation with staggered adoption.

2. **LPDID**  
   A **Local Projections Difference-in-Differences** method based on **Dube et al. (2025)**, designed to flexibly estimate dynamic treatment effects across time horizons.

3. **WindowGenerator**  
   Developed by **Eric Frey**, this utility transforms standard panel data into **2x2 DiD windows**, generating treatment and control groups using **nearest-neighbor matching** and customizable window configurations for counterfactual analysis.

---

## 🔧 Installation

After obtaining the package `.whl` file, install using:

```bash
pip install https://github.com/RenatoVassallo/CausalPanel/releases/download/0.1.1/causalpanel-0.1.1-py3-none-any.whl
```

---

## 📖 Quick Start

```python
from CausalPanel import LPDID

# Initialize the Local Projections DiD model
lpdid = LPDID(
    df=df, 
    y='y',                  # Outcome variable
    treat='treat',          # Treatment indicator
    time='year',            # Time variable
    unit='isocode',         # Unit identifier
    pre=5,                  # Number of pre-treatment periods
    post=10,                # Number of post-treatment periods
    lags=4,                 # Number of lags to include
    control_group="never"   # Control group specification
)

# Fit the model
results = lpdid.fit()

# Plot the event study results
plot = lpdid.event_study_plot()
```

---

## 📄 How to Cite

If you use *CausalPanel* in your work, please cite:

> Vassallo, R. (2025). *CausalPanel: A toolkit for causal inference in panel data settings.*
> [GitHub Repository](https://github.com/RenatoVassallo/CausalPanel)

---
