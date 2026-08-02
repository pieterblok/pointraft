# Generated Results

Only this schema is tracked. All generated files below `results/` are ignored
and remain local to the dataset and checkpoint in the current repository.

```text
results/
├── pointraft/
│   ├── predictions.csv
│   ├── train_metrics.csv
│   ├── test_metrics.txt
│   ├── train.provenance.json
│   ├── test.provenance.json
│   └── summary.md
├── model_comparison/
│   ├── predictions/<model>.csv
│   ├── logs/
│   ├── provenance/
│   └── summary.md
└── deployment/
    ├── summary.md
    └── <platform>_<timestamp>/
        ├── samples.csv
        ├── summary.json
        ├── summary.md
        ├── system.json
        └── run.log
```

Prediction CSV files contain `file_name`, `unique_id`, `gt`, `pred`, and `diff`.
Provenance records include dataset, split, checkpoint, prediction, source, and
runtime metadata when available. Historical artifacts may be incomplete; files
that were not captured at execution time must not be invented.
