# SPOTInsight AI v1.3

Narrative operational dashboard for the KnightHacks 2026 SPOT challenge.

Answers the core questions from the challenge brief:
- What happened?
- Where did it happen?
- Why does it matter?
- What requires attention?
- Is performance improving?
- What should an operator investigate next?

Includes a simple Random Forest severity classifier trained on the synthetic data.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Data files should be placed in a `data/` folder next to `app.py`.
