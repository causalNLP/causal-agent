# Evaluation Script (`eval.py`)

Copy all your .json files to the `results` folder first.

Usage:
```bash
python evaluation.py \
  --results-dir results \
  --data-dir CauSciBench/data \
  --out-dir eval_outputs --match-method samuel
```

Alternative method matching method:
```bash
python evaluation.py \
  --results-dir results \
  --data-dir ../CauSciBench/data \
  --out-dir eval_outputs --match-method sawal
```
