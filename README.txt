# Hidden Markov Model for English Phrase Chunking

This project implements a sequence labeling system for English phrase chunking using:
1. A baseline system using emission probabilities
2. A Hidden Markov Model with Viterbi decoding
3. A K-Best Viterbi algorithm for finding the 4th best sequence
4. A Structured Perceptron for improved predictions

## Requirements
- Python 3.4 or higher
- No external machine learning packages are required (uses only standard Python libraries)

## File Structure
- `hmm_chunker.py`: Main implementation of all models
- `EN/train`: Training data
- `EN/dev.in`: Development input data (words only)
- `EN/dev.out`: Gold development output (word-tag pairs)
- `EN/test.in`: Test input data (if available)
- `EvalScript/evalResult.py`: Evaluation script

## Running the Program
```
python hmm_chunker.py
```

This will:
1. Read training data from `EN/train`
2. Process rare words, replacing them with `#UNK#` token
3. Estimate emission and transition parameters
4. Generate predictions for Parts 1-4 on the development data
5. Generate predictions for Part 4 on the test data (if test.in is available)

## Output Files
- `EN/dev.p1.out`: Baseline predictions (Part 1)
- `EN/dev.p2.out`: HMM Viterbi predictions (Part 2)
- `EN/dev.p3.out`: 4th best Viterbi predictions (Part 3)
- `EN/dev.p4.out`: Structured Perceptron predictions (Part 4)
- `EN/test.p4.out`: Structured Perceptron predictions on test data (if available)

## Evaluating the Results
To evaluate the predictions against the gold standard:
```
python EvalScript/evalResult.py EN/dev.out EN/dev.p1.out
python EvalScript/evalResult.py EN/dev.out EN/dev.p2.out
python EvalScript/evalResult.py EN/dev.out EN/dev.p3.out
python EvalScript/evalResult.py EN/dev.out EN/dev.p4.out
```

Note: You may need to modify the `separator` variable in `evalResult.py` to `'\t'` for proper evaluation, as the output files use tab as the separator between words and tags.

## Model Details
1. **Baseline System**: Uses only emission probabilities to predict tags, and handles rare words with a replacement token.
2. **HMM with Viterbi**: Incorporates transition probabilities and uses the Viterbi algorithm to find the most likely tag sequence.
3. **K-Best Viterbi**: Extends the Viterbi algorithm to find the 4th best sequence by keeping track of multiple paths.
4. **Structured Perceptron**: Implements a discriminative approach for sequence labeling with feature learning. 