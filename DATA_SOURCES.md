# Fintech Data Sources

## 1. HF Dataset (Priyanshu-24) — pre-made pairs, free
- Loader: `AutoDidactGenerator.load_hf_dataset()` in `fintech_data/autodidact_gen.py`
- Dataset: `Priyanshu-24/adaption-indian-finance-qa` (28,894 rows)
- Filter: banking_regulation / capital_markets / digital_payments, or source=RBI
- Output: `data/fintech_data_autodidact.json` (6,460 pairs, `source=hf_adaption_indian_finance`)
- NOT run in the DeepSeek generation job (pure download, no API cost)
- Can be re-run any time with `python -c "from fintech_data.autodidact_gen import AutoDidactGenerator; AutoDidactGenerator().load_hf_dataset()"`

## 2. DeepSeek-generated (AutoDidact grounded) — paid, grounded in source docs
- Generator: `AutoDidactGenerator.generate_from_docs()` in `fintech_data/autodidact_gen.py`
- API: DeepSeek direct (`api.deepseek.com/v1`, model `deepseek-chat`)
- Sources: FIU-IND (PMLA/rules/orders), SEBI, gov blogs, IIMB/PIB reports
- Output: `data/fintech_data_grounded.json` (domain=`autodidact_grounded`, source=`autodidact`)
- Re-mining: each 900-char chunk can be mined up to `AUTODIDACT_MAX_REGEN` (default 3) times
- Chunk cache: `data/autodidact_progress.json` (skips already-mined chunks, no token waste)

## 3. Reliable base (free, non-API)
- Builder: `fintech_data/build_20k_dataset.py`
- Sources: GLM-5.2-Finance (HF), RakeshMadasani (HF), RBI Master Direction PDFs
- Output: `data/fintech_data_reliable.json` (15,417 pairs)

## 4. Gov crawler (free, robots-polite)
- Crawler: `fintech_data/gov_fetcher.py`
- Output: `data/gov_source_docs.json` (SEBI + blogs + reports), `data/gov_blogs.json`
- Robots.txt verified per site; blocked sites (IRDAI, PFRDA, NPCI, MCA, DigiLocker) skipped
