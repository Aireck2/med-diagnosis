# Educational Bayesian Medical Inference Simulator

An interactive local app for exploring how reported symptoms change a probability distribution over illustrative conditions. It uses Python, Streamlit, Bayes' theorem, and Shannon entropy.

[English](README.md) · [Español](README.es.md)

> [!WARNING]
> This project is for education only. Its conditions and probabilities are synthetic examples, not validated epidemiological or clinical data. The output is not a medical diagnosis or treatment recommendation. Do not use it to make health decisions.

## What it does

- Set each symptom to **Present**, **Absent**, or **Not evaluated**. Unevaluated symptoms are excluded.
- Compare prior and posterior probabilities for four illustrative respiratory conditions.
- Inspect the probability of the selected evidence, each condition's likelihood, and its contribution to the total probability.
- Compare prior and posterior Shannon entropy in bits.
- Explore a clearly explained Bayesian model with a compact synthetic catalog.

## Requirements

- Python 3.11 or newer
- `pip`

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
streamlit run app.py
```

On Windows, activate the environment with `.venv\Scripts\activate`, then run the install and launch commands above.

## Model

For a condition `D` and evaluated symptom evidence `E`, the simulator assumes symptoms are conditionally independent given `D`:

```text
P(E|D) = product_i P(E_i|D)
P(E)   = sum_d P(E|d) * P(d)             # law of total probability
P(D|E) = P(E|D) * P(D) / P(E)            # Bayes' theorem
H(D)   = -sum_d P(d) * log2(P(d))        # Shannon entropy, in bits
```

For an absent symptom, the model uses the complement of its probability of being present. With no evaluated symptoms, the posterior equals the prior. Log probabilities and log-sum-exp are used to keep inference stable for very small likelihoods.

## Limitations

Conditional independence is a simplifying assumption; real symptoms can be correlated. The priors and conditional probabilities in the sample catalog are invented for demonstration and are not estimates derived from patient records, studies, or a population. The example conditions do not represent comprehensive medical coverage.

## Project structure

```text
app.py                              # Streamlit entry point
src/diagnosis_simulator/
  domain.py                         # Catalog types and validation
  inference.py                      # Evidence likelihoods and posterior
  entropy.py                        # Shannon entropy
  sample_data.py                    # Synthetic example catalog
  presentation.py                   # Streamlit controls and results
tests/                              # Unit tests with analytical examples
README.md                           # Main documentation (English)
README.es.md                        # Documentación en español
```

## Run checks

```bash
python -m pytest
```
