# Hidden Markov Model and Structured Perceptron for English Phrase Chunking

This project implements a sequence labeling system for English phrase chunking using various probabilistic and machine learning approaches. The implementation includes multiple models of increasing complexity, from a simple baseline to more sophisticated approaches.

## Project Overview

The project implements four different approaches to phrase chunking:

1. **Baseline Model (Part 1)**: A simple model using only emission probabilities
2. **Hidden Markov Model (Part 2)**: A full HMM implementation with Viterbi decoding
3. **K-Best Viterbi (Part 3)**: An extension of the HMM to find the 4th best sequence
4. **Structured Perceptron (Part 4)**: A discriminative learning approach for improved accuracy

## Technical Implementation

### 1. Baseline Model (Part 1)

- Implements a simple emission-based model
- Handles rare words using the `#UNK#` token (words with frequency < 3)
- Uses maximum likelihood estimation for emission probabilities
- Makes independent predictions for each word based on emission probabilities

### 2. Hidden Markov Model (Part 2)

- Full implementation of an HMM with:
  - Emission probabilities (word given tag)
  - Transition probabilities (tag given previous tag)
  - Special handling of sentence start/end with START/STOP states
- Uses the Viterbi algorithm for decoding
- Implements log probabilities to prevent numerical underflow
- Includes fallback strategies for unseen transitions

### 3. K-Best Viterbi Algorithm (Part 3)

- Extends the standard Viterbi to find k-best sequences
- Maintains multiple paths during decoding
- Implements efficient path tracking and scoring
- Returns the 4th best sequence as required
- Includes fallback mechanisms for cases with fewer than k valid paths

### 4. Structured Perceptron (Part 4)

- Implements a discriminative sequence labeling model
- Features:
  - Word-tag emission features
  - Tag-tag transition features
- Uses averaged perceptron training for better generalization
- Implements Viterbi decoding with feature weights
- Includes online training with multiple iterations

## Data Processing

- Handles rare words by replacing them with `#UNK#` token
- Processes sentence boundaries appropriately
- Supports both labeled (training) and unlabeled (test) data
- Implements efficient data structures for counting and parameter estimation

## File Structure

- `hmm_chunker.py`: Main implementation file
- `EN/train`: Training data with word-tag pairs
- `EN/dev.in`: Development input data (words only)
- `EN/dev.out`: Gold standard development output
- `EN/test.in`: Test input data
- `EvalScript/evalResult.py`: Evaluation script

## Output Format

All models generate predictions in the following format:

- One word-tag pair per line, separated by a tab
- Empty lines between sentences
- Output files:
  - `EN/dev.p1.out`: Baseline predictions
  - `EN/dev.p2.out`: HMM predictions
  - `EN/dev.p3.out`: 4th best sequence predictions
  - `EN/dev.p4.out`: Structured Perceptron predictions
  - `EN/test.p4.out`: Final predictions on test data

## Usage

1. **Running the System**:

```bash
python hmm_chunker.py
```

2. **Evaluating Results**:

```bash
python EvalScript/evalResult.py EN/dev.out EN/dev.p1.out
python EvalScript/evalResult.py EN/dev.out EN/dev.p2.out
python EvalScript/evalResult.py EN/dev.out EN/dev.p3.out
python EvalScript/evalResult.py EN/dev.out EN/dev.p4.out
```

## Requirements

- Python 3.4 or higher
- No external machine learning packages required
- Standard Python libraries only:
  - collections (Counter, defaultdict)
  - math
  - heapq
  - time

## Implementation Details

### Probability Calculations

- All probability calculations use log probabilities to prevent numerical underflow
- Proper handling of zero probabilities and unseen events
- Efficient storage of parameters using defaultdict

### Performance Optimizations

- Efficient data structures for parameter storage
- Log probabilities for numerical stability
- Smart handling of rare words
- Fallback strategies for unseen events

### Error Handling

- Robust handling of file I/O
- Fallback mechanisms for edge cases
- Proper handling of sentence boundaries
- Graceful handling of missing test data

## Evaluation

The system can be evaluated using the provided evaluation script, which calculates:

- Precision
- Recall
- F-score

The evaluation takes into account both the chunk boundaries and the assigned tags.
