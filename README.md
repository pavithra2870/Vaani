# The Relay

A question-answering pipeline over three arXiv papers (Attention Is All You Need, Moshi, H-Net), built to work inside Groq's free-tier rate limits. It has a five-step pipeline, a gateway that paces, admits and retries requests, a stand-in model service, and a load generator.

## Setup
```
pip install -r requirements.txt
copy .env.example .env      # add your GROQ_API_KEY
```
Put the three papers in `data/` (1706.03762v7.pdf, 2410.00037v1.pdf, 2507.07955v2.pdf), then:
```
python -m relay.ingest
```

## Layout
| Path | What it is |
|---|---|
| `relay/` | pipeline, gateway, retrieval, model client, grader |
| `standin/` | stand-in model service |
| `eval/` | the 21 evaluation questions and the scripts that run them |
| `probe.py` | probes the real service for its limits |
| `calibrate.py` | fits the stand-in to the real measurements |
| `loadgen.py` | load scenarios (real and stand-in) |
| `analyze.py` | computes the result tables from `results.csv` |
| `results.csv` | one row per request |

## Run
```
python -m eval.run --model openai/gpt-oss-20b --setting small-k8 --mode lean
python loadgen.py --list
python loadgen.py --scenario steady-sim
python loadgen.py --scenario burst-real --model openai/gpt-oss-20b    # uses real quota
python analyze.py
```
