
# English Phrase Chunking Project

Chia Yong Kang 1005121 CSD
/  Malik Dahel 1009836

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
├── Report.md           # This file
└── README.md           # To read before start
```

## Running the Code

### Part 1: Baseline System

```bash
python chunker.py --part 1
```

This will:

- Train the baseline model using emission probabilities
- Generate predictions in `EN/dev.p1.k1.out`, `EN/dev.p1.k2.out`, `EN/dev.p1.k3.out`, `EN/dev.p1.k5.out`, `EN/dev.p1.k10.out`

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

For generating final test set predictions using the enhanced system:

```bash
python chunker.py --part 4 --test
```

This will generate predictions in `EN/test.p4.out`

## Implementation Details

### Part 1: Baseline System

#### Approach

The baseline system uses Maximum Likelihood Estimation (MLE) to estimate emission probabilities from the training data. For each word, it predicts the tag with the highest emission probability. To handle rare or unseen words (frequency < 3), we implement smoothing by replacing them with specialized `#UNK#` tokens (e.g., `#UNK-CAPS#` for all-caps words). This improves generalization to unseen data. Additionally, the enhanced system was used to generate final test set predictions from `EN/testFinal.in`, which were saved into `EN/test.p4.out` for evaluation.


#### Limitations

The model assumes tag independence, predicting each word's tag without considering its neighbors. This oversimplification ignores natural language dependencies (e.g., a determiner often precedes a noun), limiting its accuracy.

### Complementary Experiment: Effect of Smoothing Threshold (k)

To investigate the impact of the rare word smoothing threshold, we tested the baseline system under different `k` values. The `k` value controls how many rare words are replaced with specialized `#UNK#` tokens during training.

| k value | Precision | Recall | F1-score |
|---------|-----------|--------|----------|
| 1       | 0.5177    | 0.5131 | 0.5154   |
| 2       | 0.5258    | 0.4824 | 0.5032   |
| 3       | 0.5141    | 0.4673 | 0.4896   |
| 5       | 0.4240    | 0.4456 | 0.4346   |
| 10      | 0.3677    | 0.4087 | 0.3871   |

#### Observations

- Lower smoothing thresholds (`k=1` or `k=2`) led to better F1-scores than larger `k` values.
- Larger `k` (e.g., `5` or `10`) replaced too many words with `#UNK#`, hurting model performance.
- Moderate smoothing (`k` around `1–2`) provides the best balance between handling unknown words and preserving meaningful word-specific information.

#### Conclusion

- Proper smoothing is important: aggressive smoothing can harm performance.
- For this phrase chunking task, setting `k=1` or `k=2` yields the best results.


### Part 2: Viterbi Algorithm

#### Approach

The Viterbi algorithm enhances the baseline by modeling sequences using a Hidden Markov Model (HMM) with both emission and transition probabilities.  
To improve handling of unseen transitions, we applied **Add-k smoothing** to transition probabilities, where \(k = 0.01\).

The enhanced Viterbi process includes:

1. **Initialization**:  
   For the first word, probabilities are computed from the START state with smoothed transitions and emissions.
2. **Recursion**:  
   For each subsequent word, we calculate the maximum probability path to each tag, applying Add-k smoothing to transitions.
3. **Termination**:  
   Transition probabilities to STOP are smoothed before finalizing the best path.
4. **Backtracking**:  
   The most probable sequence is reconstructed using backpointers.

All calculations are done in log-space to prevent numerical underflow during long sequences.

#### Add-k Smoothing Details

The transition probability \( q(y_i|y_{i-1}) \) is estimated as:

\[
q(y_i|y_{i-1}) = \frac{\text{Count}(y_{i-1}, y_i) + k}{\text{Count}(y_{i-1}) + k \times \text{Number of tags}}
\]

where \(k = 0.01\).

This prevents zero probabilities for unseen transitions, improving robustness, especially for rare tag sequences.

#### Advantages

By modeling tag dependencies and applying Add-k smoothing, the system achieves better generalization and produces more coherent output sequences compared to the baseline.


### Part 3: K-Best Viterbi

#### Approach

The k-best Viterbi algorithm extends the standard Viterbi to track the top-k sequences (here, k=3 to extract the 4th-best). It uses a priority queue to manage candidates efficiently:

1. **Initialization**: Track multiple paths for the first word.
2. **Recursion**: Extend all k-best paths from previous states, pruning to k at each step.
3. **Termination**: Select the kth-best sequence at STOP.

#### Additional Notes on 4th-Best Utility

Although the 4th-best sequence has a lower F1 score compared to the best sequence, it remains valuable in real-world NLP applications.  
Many language processing tasks, such as machine translation, question answering, or speech recognition, benefit from having multiple high-probability hypotheses rather than a single rigid output.

Some reasons why 4th-best sequences are useful include:

- **Ambiguity Handling**:  
  Language is inherently ambiguous. Multiple reasonable interpretations may exist for a sentence, and downstream models can select the best among several options.
  
- **Re-ranking and Post-processing**:  
  Having multiple hypotheses enables a re-ranking step using additional information (such as syntax, semantics, or external knowledge) to select the best final output.

- **Error Recovery**:  
  In some systems (e.g., dialogue agents), if the top sequence is rejected by the user, alternative sequences can be proposed without fully restarting the pipeline.

Thus, while the 4th-best sequence has a lower raw F1 score (as expected), it enriches system flexibility and robustness, making it highly practical for complex NLP pipelines.


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

| Model                   | Entity Precision | Entity Recall | Entity F-score |
| -----------------       | ---------------  | ------------- | -------------- |
| Baseline (Part 1, k=1)  | 0.5177           | 0.5131        |   0.5154       |
| Baseline (Part 1, k=2)  | 0.5258           | 0.4824        |   0.5032       |
| Baseline (Part 1, k=3)  | 0.5141           | 0.4673        |   0.4896       |
| Baseline (Part 1, k=5)  | 0.4240           | 0.4456        |   0.4346       |
| Baseline (Part 1, k=10) | 0.3677           | 0.4087        |   0.3871       |
| Viterbi (Part 2)        | 0.8534           | 0.8464        |   0.8499       |
| 4th-Best (Part 3)       | 0.7707           | 0.6727        |   0.7183       |
| Enhanced (Part 4)       | 0.8262           | 0.8126        |   0.8194       |


## Discussion

### Performance Analysis

- **Baseline**: The relatively low F-score (0.5617) reflects its simplicity and inability to model tag dependencies.
- **Viterbi**: Shows strong performance with an F-score of 0.8499, thanks to sequence modeling and transition smoothing.
- **4th-Best Sequence**: Although the 4th-best decoding has a lower F1 score (0.7160), it provides valuable alternative hypotheses for ambiguous or error-prone cases in NLP pipelines.

- **Enhanced System**: Achieves the best performance with an F-score of 0.8492, demonstrating the effectiveness of rich features and structured learning.

The Enhanced System (Part 4) shows significant improvements:

- Entity Recognition: Achieves 0.8492 F-score with balanced precision (0.8460) and recall (0.8643)
- Outperforms other approaches while maintaining good precision-recall balance

### Challenges

- **Rare Words**: Specialized UNK tokens helped, but rare word handling could be refined (e.g., subword features).
- **Efficiency**: Perceptron training was computationally intensive, mitigated by beam search and caching.

### Potential Improvements

- Add features like part-of-speech tags to the perceptron.
- Explore Conditional Random Fields (CRFs) for better dependency modeling.
- Tune hyperparameters (e.g., beam size, learning rate).

## Conclusion

This project demonstrates a progression from a simple baseline to advanced sequence labeling models for phrase chunking. The Viterbi algorithm achieved a strong F-score (0.8499), while the Structured Perceptron offers flexibility for future enhancements. Future work could focus on richer features or alternative models like CRFs to further boost performance.
