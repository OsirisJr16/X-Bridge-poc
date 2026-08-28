# x-bridge-poc

Proof of concept for schema matching / mapping between a source and a target schema,
combining lexical, semantic, and hybrid matching strategies.

## Structure

```
x-bridge-poc/
├── data/
│   └── schemas/
│       ├── source/          # source schema files
│       └── target/          # target schema files
├── src/
│   ├── parsers/             # schema file parsers (XSD, JSON Schema, CSV, ...)
│   ├── preprocessing/       # normalization, tokenization, feature prep
│   ├── matching/
│   │   ├── lexical.py       # string / token similarity matchers
│   │   ├── semantic.py      # embedding-based matchers
│   │   └── hybrid.py        # combination + scoring strategies
│   ├── evaluation/          # metrics, ground-truth comparison
│   └── models/              # data models / trained artifacts
├── notebooks/               # exploration
├── tests/
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
pytest
```
