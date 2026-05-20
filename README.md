# Synthics
Generate physics-structured datasets for machine learning.

## Getting started
Use either option to generate synthetic datasets based on given corpus (Feynman equations by default in this project) or randomly based on given settings.

```python
import synthics

equations = synthics.load_feynman_csv()

# Option 1: Generate synthetic datasets based on corpus
datasets_from_corpus = synthics.generate_datasets(equations, n_datasets=100, n_samples_per_dataset=200)

# Option 2: Generate synthetic datasets based on settings
datasets_random = synthics.generate_datasets(
    n_datasets            = 50,
    n_samples_per_dataset = 200,
    uniform_ratio         = 0.5,
    tau                   = 7,
    max_depth             = 8,
    n_vars                = 3,
)
```
## Examples
Check the ```examples.ipynb``` notebook for more details about function usage.