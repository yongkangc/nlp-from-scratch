# English Phrase Chunking Project

This project implements sequence labeling models for English phrase chunking, including a baseline system, Viterbi algorithm, k-best Viterbi, and an enhanced Structured Perceptron.

## Requirements

- Python 3.10 or higher
- No external ML libraries required (only standard Python libraries are used)

## Project Structure

```
.
├── chunker.py           # Main script for all parts
├── EN/                  # Data directory
│   ├── train            # Training data
│   ├── dev.in           # Development input
│   ├── dev.out          # Development gold standard
│   ├── test.in          # Provided earlier for practice
│   └── testFinal.in     # Final evaluation test set (used with --test)
├── Report.md            # Project report
└── README.md            # This file
```

## Running the Code

### Part 1: Baseline System

```bash
python chunker.py --part 1
```

This will:

- Train the baseline model using emission probabilities.
- Generate predictions in `EN/dev.p1.k1.out`, `EN/dev.p1.k2.out`, `EN/dev.p1.k3.out`, `EN/dev.p1.k5.out`, `EN/dev.p1.k10.out`.

### Part 2: Viterbi Algorithm

```bash
python chunker.py --part 2
```

This will:

- Train the HMM model with Viterbi decoding.
- Generate predictions in `EN/dev.p2.out`.

### Part 3: 4th-Best Sequence (k-best Viterbi)

```bash
python chunker.py --part 3
```

This will:

- Run the k-best Viterbi algorithm (k=3 for 4th-best).
- Generate predictions in `EN/dev.p3.out`.

### Part 4: Enhanced System (Structured Perceptron)

```bash
python chunker.py --part 4
```

This will:

- Train the Enhanced Structured Perceptron.
- Generate predictions in `EN/dev.p4.out`.

For final test set predictions (using `testFinal.in`):

```bash
python chunker.py --part 4 --test
```

This will:

- Generate predictions in `EN/test.p4.out`.

## Implementation Details

### Part 1: Baseline

- Uses Maximum Likelihood Estimation (MLE) for emission parameters.
- Specialized `#UNK#` tokens for rare words (based on smoothing threshold `k`).
- Predicts tags independently without sequence context.

### Part 2: Viterbi Algorithm

- Implements full HMM with transition and emission probabilities.
- Dynamic programming with backtracking to find the best sequence.
- Uses log-probabilities for stability.

### Part 3: K-Best Viterbi

- Extends Viterbi to maintain multiple (top-k) paths at each step.
- Extracts the 4th-best complete sequence.

### Part 4: Enhanced Structured Perceptron

- Rich feature set: word shapes, prefixes, suffixes, context words.
- Beam search for efficient decoding.
- Early stopping and learning rate decay during training.

## Results

| Model                   | Entity Precision | Entity Recall | Entity F-score |
| ------------------------ | ---------------- | ------------- | -------------- |
| Baseline (Part 1, k=1)    | 0.5177            | 0.5131        | 0.5154         |
| Baseline (Part 1, k=2)    | 0.5258            | 0.4824        | 0.5032         |
| Baseline (Part 1, k=3)    | 0.5141            | 0.4673        | 0.4896         |
| Baseline (Part 1, k=5)    | 0.4240            | 0.4456        | 0.4346         |
| Baseline (Part 1, k=10)   | 0.3677            | 0.4087        | 0.3871         |
| Viterbi (Part 2)          | 0.8534            | 0.8464        | 0.8499         |
| 4th-Best (Part 3)         | 0.7707            | 0.6727        | 0.7183         |
| Enhanced (Part 4)         | 0.8262            | 0.8126        | 0.8194         |
