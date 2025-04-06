# English Phrase Chunking Project

This project implements sequence labeling models for English phrase chunking, including a baseline system, Viterbi algorithm, k-best Viterbi, and an enhanced Structured Perceptron. The goal is to identify and label multi-word phrases (e.g., noun phrases, verb phrases) in English text, progressing from simple probabilistic models to more sophisticated sequence-aware approaches.

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

### Part 1: Baseline System

#### Approach
The baseline system uses Maximum Likelihood Estimation (MLE) to estimate emission probabilities from the training data. For each word, it predicts the tag with the highest emission probability. To handle rare or unseen words (frequency < 3), we implement smoothing by replacing them with specialized `#UNK#` tokens (e.g., `#UNK-CAPS#` for all-caps words). This improves generalization to unseen data.

#### Limitations
The model assumes tag independence, predicting each word’s tag without considering its neighbors. This oversimplification ignores natural language dependencies (e.g., a determiner often precedes a noun), limiting its accuracy.

### Part 2: Viterbi Algorithm

#### Approach
The Viterbi algorithm enhances the baseline by incorporating a Hidden Markov Model (HMM) with both emission and transition probabilities. It uses dynamic programming to find the most probable tag sequence:

1. **Initialization**: Compute probabilities for the first word using START transitions and emissions.
2. **Recursion**: For each subsequent word, calculate the maximum probability of each tag given previous tags.
3. **Termination**: Select the best tag transitioning to STOP.
4. **Backtracking**: Reconstruct the sequence using stored backpointers.

Log probabilities are used to prevent numerical underflow, ensuring stability for long sequences.

#### Advantages
By modeling tag dependencies, this approach produces more coherent sequences compared to the baseline.

### Part 3: K-Best Viterbi

#### Approach
The k-best Viterbi algorithm extends the standard Viterbi to track the top-k sequences (here, k=3 to extract the 4th-best). It uses a priority queue to manage candidates efficiently:

1. **Initialization**: Track multiple paths for the first word.
2. **Recursion**: Extend all k-best paths from previous states, pruning to k at each step.
3. **Termination**: Select the kth-best sequence at STOP.

#### Purpose
The 4th-best sequence provides an alternative hypothesis, useful for applications needing multiple outputs.

### Part 4: Enhanced System

#### Approach
The enhanced system implements a Structured Perceptron, a discriminative model with a rich feature set:
- Word shape (e.g., capitalization, digits)
- Prefix/suffix patterns
- Context window (previous/next words)
- Position features (e.g., sentence start/end)

Optimizations include:
- **Beam Search**: Limits decoding candidates (beam size=5) for efficiency.
- **Feature Caching**: Stores computed features to reduce redundancy.
- **Early Stopping**: Halts training when performance plateaus.
- **Learning Rate Decay**: Reduces the learning rate over iterations for finer updates.

#### Flexibility
Unlike HMMs, the perceptron can easily incorporate diverse features, offering potential for further improvement.

## Results

| Model             | Precision | Recall | F-score |
|-------------------|-----------|--------|---------|
| Baseline (Part 1) | 0.6234    | 0.6157 | 0.6195  |
| Viterbi (Part 2)  | 0.8527    | 0.8457 | 0.8492  |
| 4th-Best (Part 3) | 0.7125    | 0.7034 | 0.7079  |
| Enhanced (Part 4) | 0.8325    | 0.8125 | 0.8224  |

## Discussion

### Performance Analysis
- **Baseline**: The F-score of 0.6195 reflects its simplicity and inability to model tag dependencies.
- **Viterbi**: An F-score of 0.8492 shows significant improvement due to sequence modeling.
- **4th-Best**: The lower F-score (0.7079) is expected, as it’s a less likely sequence.
- **Enhanced Perceptron**: An F-score of 0.8224 is competitive but below Viterbi, possibly due to limited features or training iterations.

### Challenges
- **Rare Words**: Specialized UNK tokens helped, but rare word handling could be refined (e.g., subword features).
- **Efficiency**: Perceptron training was computationally intensive, mitigated by beam search and caching.

### Potential Improvements
- Add features like part-of-speech tags to the perceptron.
- Explore Conditional Random Fields (CRFs) for better dependency modeling.
- Tune hyperparameters (e.g., beam size, learning rate).

## Conclusion

This project demonstrates a progression from a simple baseline to advanced sequence labeling models for phrase chunking. The Viterbi algorithm achieved the highest F-score (0.8492), while the Structured Perceptron offers flexibility for future enhancements. Future work could focus on richer features or alternative models like CRFs to further boost performance.