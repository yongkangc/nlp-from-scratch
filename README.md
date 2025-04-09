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
│   ├── train           # Training data
│   ├── dev.in          # Development input
│   ├── dev.out         # Development gold standard
│   └── test.in         # Test input (provided later)
├── Report.md           # Project report
└── README.md           # This file
```

## Running the Code

### Part 1: Baseline System

```bash
python chunker.py --part 1
```

This will:

- Train the baseline model using emission probabilities
- Generate predictions in `EN/dev.p1.out`

### Part 2: Viterbi Algorithm

```bash
python chunker.py --part 2
```

This will:

- Train the HMM model with Viterbi decoding
- Generate predictions in `EN/dev.p2.out`

### Part 3: 4th-Best Sequence

```bash
python chunker.py --part 3
```

This will:

- Run k-best Viterbi algorithm (k=3 for 4th best)
- Generate predictions in `EN/dev.p3.out`

### Part 4: Enhanced System

```bash
python chunker.py --part 4
```

This will:

- Train the Enhanced Structured Perceptron
- Generate predictions in `EN/dev.p4.out`

For test set predictions (when available):

```bash
python chunker.py --part 4 --test
```

This will generate predictions in `EN/test.p4.out`

## Implementation Details

### Part 1: Baseline

- Uses MLE for emission parameters
- Implements smoothing with specialized UNK tokens
- Makes independent predictions for each word

### Part 2: Viterbi

- Implements full HMM with transition and emission probabilities
- Uses log probabilities for numerical stability
- Handles START/STOP transitions

### Part 3: K-Best Viterbi

- Extends Viterbi to track k-best sequences
- Uses efficient data structures for k-best list management
- Implements backtracking for the 4th-best sequence

### Part 4: Enhanced System

- Implements Structured Perceptron with rich features
- Uses beam search and feature caching for efficiency
- Includes early stopping and learning rate decay

## Results

| Model             | Entity    |        |         | Sentiment |        |         |
| ----------------- | --------- | ------ | ------- | --------- | ------ | ------- |
|                   | Precision | Recall | F-score | Precision | Recall | F-score |
| Baseline (Part 1) | 0.5617    | 0.5617 | 0.5617  | 0.4692    | 0.4692 | 0.4692  |
| Viterbi (Part 2)  | 0.8267    | 0.8267 | 0.8267  | 0.7954    | 0.7954 | 0.7954  |
| 4th-Best (Part 3) | 0.7160    | 0.7160 | 0.7160  | 0.6894    | 0.6894 | 0.6894  |
| Enhanced (Part 4) | 0.8460    | 0.8643 | 0.8492  | 0.8219    | 0.8397 | 0.8198  |
