# X-Bridge POC — Semantic Schema Matching

**This is a research proof of concept, not the X-Bridge middleware.** It contains no API, no
frontend, no database, no connectors and no deployment tooling. It exists to answer one question
under controlled conditions, on a single pair of schemas.

## 1. Objective

Given a source JSON Schema and a target JSON Schema, automatically identify semantic
correspondences between their fields, score the confidence of each correspondence, and produce
mapping suggestions.

The practical goal is to reduce the manual effort required to map heterogeneous schemas. The POC
therefore measures not only whether mappings are correct, but how many of them a human still has
to validate.

## 2. Research hypothesis

> Combining lexical similarity, multilingual sentence embeddings and data-type compatibility
> produces better schema mappings than any of those signals used alone.

This hypothesis is **not assumed** by the implementation. The three approaches are implemented
behind the same interface and evaluated with the same metrics against the same ground truth, so
the comparison can falsify the hypothesis. On the schemas shipped here, it partly does — see
§7 Results.

## 3. Architecture

```
src/x_bridge/
├── domain/              models and enums, no I/O, no framework dependency
├── application/         orchestration: candidate selection, conflict resolution, reporting
├── infrastructure/      JSON Schema parsing, sentence-transformers encoder
├── matching/            lexical, semantic, hybrid, type compatibility, confidence scoring
└── evaluation/          ground-truth comparison and metrics
```

Two boundaries are deliberate:

- **Parsing is separate from matching.** `infrastructure/schema_analyzer.py` turns a JSON Schema
  into `SchemaField` objects and knows nothing about similarity.
- **The embedding model is separate from the matching algorithms.** Matchers depend on the
  `TextEncoder` protocol, never on `sentence-transformers`. The full test suite runs offline
  against a deterministic fake encoder.

## 4. Matching pipeline

1. **Analyze** both schemas into flat `SchemaField` lists. Nested objects are flattened while
   preserving the full path, so `adresse.code_postal` stays a uniquely identifiable field.
2. **Score** every source/target pair with the selected method, producing an
   `n_source × n_target` similarity matrix.
   - *Lexical*: field names are normalized (camelCase split, separators removed, lowercased,
     known abbreviations expanded), then compared with a blend of token-set overlap and character
     sequence ratio. Normalization performs no French→English translation — that would leak
     semantics into the lexical baseline and invalidate the comparison.
   - *Semantic*: each field is rendered as `field: ... | type: ... | description: ...`, encoded
     with `intfloat/multilingual-e5-base`, and compared by cosine similarity. Embeddings are
     cached per representation string, so a repeated field is encoded once.
   - *Hybrid*: `lexical_weight × lexical + semantic_weight × semantic + type_weight × type`.
3. **Compute type compatibility** for every pair. Incompatible types are *not* rejected outright;
   compatibility is one weighted factor, so a strong semantic match can survive a type mismatch.
4. **Score confidence** (see §5).
5. **Select candidates** — all pairs at or above the review threshold are ranked by confidence and
   assigned greedily. By default a target may be used once (`allow_many_to_one` disables this), so
   when two source fields compete for one target the higher-confidence mapping wins. Ties break on
   field path, making the result deterministic.
6. **Classify** each mapping: `>= 0.90` automatic, `>= 0.70` review required, below that the source
   field is left unmatched. No source field is forced to match.

### Cosine rescaling

E5 models place unrelated short texts around 0.7–0.8 cosine, so raw cosine values are not usable
as 0–1 scores. Scores are rescaled from `[cosine_floor, 1]` to `[0, 1]`, with `cosine_floor = 0.70`
by default. This is an experimental calibration constant, not a property of the model, and it is
configurable.

## 5. Similarity score vs confidence score

These are deliberately distinct, and both are reported.

- **`similarity_score`** is what the matching method itself produced: lexical similarity, rescaled
  cosine similarity, or the hybrid weighted sum. It describes *the pair in isolation*.
- **`confidence_score`** is how much the proposal should be trusted, given its context:

  ```
  confidence = similarity × (1 + margin_bonus_weight × margin_factor
                               − type_penalty_weight × (1 − type_compatibility))
  ```

  `margin_factor` is the gap between this candidate and the best competing candidate for the same
  source field, saturating at `margin_saturation`. A field that matches one target clearly is more
  trustworthy than one that matches three targets equally well, even at identical similarity.
  A type mismatch reduces confidence without vetoing the mapping.

Thresholds are applied to **confidence**, not similarity.

## 6. Installation

Requires Python 3.12+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Dependencies are declared in `pyproject.toml`. `requirements.txt` and `requirements-dev.txt`
mirror it for plain pip workflows:

```bash
pip install -r requirements-dev.txt
```

The first semantic run downloads `intfloat/multilingual-e5-base` (~1.1 GB). The test suite does
not require it.

## 7. Usage and experiments

```bash
python scripts/run_matching.py --method lexical
python scripts/run_matching.py --method semantic
python scripts/run_matching.py --method hybrid
```

Options: `--output-format json`, `--output-file`, `--automatic-threshold`, `--review-threshold`,
`--allow-many-to-one`, `--source`, `--target`.

```bash
python scripts/run_evaluation.py --detailed
python scripts/run_evaluation.py --include-review-required
```

`run_evaluation.py` runs all three methods against `data/ground_truth.json` and prints a
comparison. The encoder is shared across methods so the model is loaded once.

### Results on the bundled schemas

The dataset is 20 French source fields and 20 English target fields (both with nested address
objects), 17 true correspondences, and 3 fields on each side that must not match.

Counting only mappings accepted automatically (confidence ≥ 0.90):

| Method   | Precision | Recall | F1    | Auto rate | Review rate |
|----------|-----------|--------|-------|-----------|-------------|
| Lexical  | 1.000     | 0.059  | 0.111 | 0.050     | 0.000       |
| Semantic | 1.000     | 0.353  | 0.522 | 0.300     | 0.500       |
| Hybrid   | 1.000     | 0.059  | 0.111 | 0.050     | 0.550       |

Counting proposals that reach review as well (`--include-review-required`), which measures the
matcher independently of the acceptance threshold:

| Method   | Precision | Recall | F1    |
|----------|-----------|--------|-------|
| Lexical  | 1.000     | 0.059  | 0.111 |
| Semantic | 1.000     | 0.941  | 0.970 |
| Hybrid   | 1.000     | 0.706  | 0.828 |

**The hypothesis is not confirmed on this dataset.** Semantic matching alone outperforms the
hybrid combination. The reason is visible in the per-pair output: across a French→English
boundary, lexical similarity is near zero for genuinely correct pairs such as
`nom_client → customerName`, so a 0.20-weighted lexical term systematically drags correct
mappings below the thresholds. Lexical similarity only contributes where the two schemas already
share vocabulary (`adresse.code_postal → address.postalCode` is the one pair lexical matching
finds on its own).

These numbers come from one schema pair with one weight configuration. They show that the default
weights are wrong for cross-lingual schemas; they do not show that hybrid matching is worthless.
Weights and thresholds are configuration, precisely so this can be tested rather than argued —
a lower `lexical_weight` is the obvious next experiment.

Precision is 1.000 for every method, which reflects a conservative pipeline on a small, clean
dataset. It should not be read as a general accuracy claim.

## 8. Evaluation metrics

The evaluator reports three groups separately, because they answer different questions:

- **Model performance** — true/false positives, false negatives, true negatives, precision,
  recall, F1, accuracy. Accuracy is reported but is not a primary metric: most source/target pairs
  are non-matches, so it is inflated by design.
- **Matching coverage** — automatic rate, review-required rate, unmatched rate.
- **Human validation requirements** — how many proposals are wrong and how many correct mappings
  were missed, i.e. the residual manual work.

A mapping counts as a true positive only if it reproduces the exact target field from
`data/ground_truth.json`. `data/ground_truth.json` is used for evaluation only and never
influences matching.

## 9. Limitations

- One schema pair, one language pair, 20 fields per side. Nothing here generalizes without more
  datasets.
- Ground truth is hand-written by a single author and encodes one opinion about what "correct"
  means.
- Only one-to-one mappings. No field splitting, merging, or value transformation.
- Arrays are flattened through their `items` schema; tuple-typed arrays, `$ref`, `oneOf`, `anyOf`
  and `allOf` are not resolved.
- Type compatibility is a small fixed table, not format-aware — a `string` holding an ISO date and
  a `string` holding a name are treated identically.
- The hybrid weights and the cosine floor are untuned defaults.
- Matching is O(n×m) over all field pairs, with no blocking or indexing.

## 10. Future work

- Tune weights and thresholds against the ground truth instead of assuming the defaults, and
  report the search rather than the winner alone.
- Add more schema pairs, including same-language pairs where lexical similarity should help, to
  find where hybrid matching actually wins.
- Add an optional LLM-based ambiguity resolver behind a separate interface, as a second experiment
  on top of this baseline, for the cases the baseline leaves in review.
- Global assignment (Hungarian algorithm) instead of greedy selection.
- Format-aware and structure-aware type compatibility.
- Calibrate confidence against observed human acceptance rates.

## Tests

```bash
pytest
```

54 tests covering schema parsing, name normalization, lexical similarity, semantic representation
and rescaling, embedding cache behaviour, type compatibility, hybrid scoring, threshold
classification, duplicate-target handling, determinism, and evaluation metrics. All use
deterministic fixtures and a fake encoder; no network access and no model download is required.
