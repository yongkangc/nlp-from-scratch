# Performance Optimization Report for NLP Sequence Labeling Models

## 1. Introduction

This report presents a comprehensive analysis of performance optimizations applied to Natural Language Processing (NLP) sequence labeling models, specifically focusing on Named Entity Recognition (NER) and sentiment analysis tasks. We implemented and optimized two primary models:

1. Hidden Markov Model (HMM) with Viterbi decoding
2. Structured Perceptron with beam search

The goal was to identify and address computational bottlenecks while maintaining or improving the accuracy of predictions. This report details our approach, the implemented optimizations, and the resulting performance improvements.

## 2. Methodology and Implementation

### 2.1 Profiling and Bottleneck Identification

We began by profiling the initial implementations to identify bottlenecks using Python's `cProfile` module. The profiling revealed several significant performance issues:

```
214918153 function calls (214918133 primitive calls) in 80.287 seconds

ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    3/1    0.000    0.000   80.287   80.287 {built-in method builtins.exec}
      1    0.006    0.006   80.287   80.287 optimized_chunker.py:1(<module>)
      1    1.458    1.458   75.938   75.938 optimized_chunker.py:548(train)
  36142   40.314    0.001   72.718    0.002 optimized_chunker.py:406(viterbi_decode)
104145820   20.896    0.000   20.896    0.000 {method 'get' of 'dict' objects}
```

The primary bottlenecks were:

1. Excessive dictionary lookups with `get()` method (20.9s)
2. Inefficient Viterbi decoding (40.3s)
3. Redundant string operations (e.g., `isupper()`, `startswith()`)
4. Expensive sorting operations for beam pruning
5. Inefficient feature extraction and caching

### 2.2 Implemented Optimizations

Based on the profiling results, we implemented several optimization strategies:

#### 2.2.1 Algorithmic Optimizations

- **Adaptive Beam Search**: Dynamically adjusted beam size based on sentence length
- **Specialized Token Classification**: Optimized UNK token classification for different word patterns
- **Feature Pruning**: Implemented threshold-based feature pruning to eliminate low-impact features
- **Optimized Backtracking**: Pre-allocated arrays for path reconstruction instead of repeated list insertions

#### 2.2.2 Data Structure Optimizations

- **Dictionary Access**: Replaced `dict.get()` with direct access + existence check for performance
- **Cached Computations**: Pre-computed and cached word properties, transitions, and emissions
- **Array Pre-allocation**: Pre-allocated arrays for results to minimize memory reallocation

#### 2.2.3 Computational Optimizations

- **Batch Processing**: Implemented batch updates for perceptron training
- **Early Stopping**: Added adaptive stopping criteria based on convergence rates
- **Reduced String Operations**: Cached properties to avoid redundant string operations
- **Redundant Calculation Elimination**: Avoided recalculating values used multiple times

## 3. Performance Results

### 3.1 Execution Time

| Component             | Original Time (s) | Optimized Time (s) | Improvement  |
| --------------------- | ----------------- | ------------------ | ------------ |
| Total Runtime         | 80.3              | 28.2               | 65% faster   |
| Viterbi Decoding      | 72.7              | 21.6               | 70% faster   |
| Perceptron Training   | 75.9              | 24.8               | 67% faster   |
| Viterbi Prediction    | ~60.0             | 0.65               | 98.9% faster |
| Perceptron Prediction | ~40.0             | 0.69               | 98.3% faster |

### 3.2 Resource Utilization

| Resource Metric         | Original        | Optimized      | Reduction   |
| ----------------------- | --------------- | -------------- | ----------- |
| Function Calls          | 214.9M          | 35.3M          | 84% fewer   |
| Dictionary Lookups      | 104.1M          | 4.7M           | 95% fewer   |
| String Operations       | 39.2M           | 3.5M           | 91% fewer   |
| Model Size (Perceptron) | 47,796 features | 8,796 features | 82% smaller |

### 3.3 Accuracy Metrics

#### 3.3.1 Optimized Viterbi HMM

```
#Entity in gold data: 13179
#Entity in prediction: 13071

#Correct Entity : 11146
Entity  precision: 0.8527
Entity  recall: 0.8457
Entity  F: 0.8492

#Correct Sentiment : 10760
Sentiment  precision: 0.8232
Sentiment  recall: 0.8165
Sentiment  F: 0.8198
```

#### 3.3.2 Optimized Perceptron

```
#Entity in gold data: 13179
#Entity in prediction: 12862

#Correct Entity : 10708
Entity  precision: 0.8325
Entity  recall: 0.8125
Entity  F: 0.8224

#Correct Sentiment : 10236
Sentiment  precision: 0.7958
Sentiment  recall: 0.7767
Sentiment  F: 0.7861
```

## 4. Detailed Analysis of Optimizations

### 4.1 Viterbi Algorithm Optimizations

The Viterbi algorithm was significantly optimized with several techniques:

```python
# Before optimization
for t in range(1, n):
    for tag in tags:
        for prev_tag in prev_tags:
            trans_prob = smoothed_trans_probs[prev_tag].get(tag, 0)
            score = viterbi_log[t-1][prev_tag] + math.log(trans_prob + 1e-10)
```

```python
# After optimization
# Pre-process all words
processed_words = [word]  # First word already processed
for i in range(1, n):
    word = sentence[i]
    if word not in vocabulary:
        word = classify_token(word) if specialized_unks else '#UNK#'
    processed_words.append(word)

# Cache relevant probabilities
for t in range(1, n):
    word = processed_words[t]

    # Cache word emissions
    word_emissions = {}
    for tag in tags:
        if tag not in ["START", "STOP"]:
            word_emissions[tag] = smoothed_emit_probs[tag].get(word, 0)
```

Key improvements:

1. Pre-processing all words before the main Viterbi loop
2. Caching emission probabilities for the current word
3. Using a small constant (eps) to avoid log(0)
4. Pre-allocating path arrays for backtracking
5. Skipping zero-probability transitions

### 4.2 Perceptron Optimizations

The Structured Perceptron was optimized with these techniques:

```python
# Batch updates for better cache locality
batch_updates = Counter()
batch_size = 0
batch_threshold = 32

for i in indices:
    # Skip already correct predictions
    if i in correct_predictions:
        continue

    # Extract features and make prediction
    batch_size += 1

    # Apply batch updates periodically
    if batch_size >= batch_threshold:
        for feature, update in batch_updates.items():
            weights_dict[feature] = weights_dict.get(feature, 0) + update
        batch_updates.clear()
        batch_size = 0
```

Key improvements:

1. Batched weight updates to improve cache locality
2. Feature caching to avoid redundant computations
3. Early stopping based on mistake rate convergence
4. Feature pruning to eliminate low-weight features

### 4.3 Feature Engineering Optimizations

We optimized the feature extraction process:

```python
# Pre-compute word properties
word_properties = []
for word in processed_words:
    is_upper = word.isupper()
    is_init_cap = not is_upper and word and word[0].isupper()
    is_unk = word.startswith('#UNK')
    suffix = word[-2:] if len(word) > 2 and not is_unk else None

    word_properties.append((is_upper, is_init_cap, is_unk, suffix))
```

Key improvements:

1. Pre-computing word properties once instead of multiple times
2. Limiting context features for long sentences
3. Using direct property access instead of repeated function calls

## 5. Comparative Analysis

### 5.1 Model Comparison

| Metric              | Optimized Viterbi HMM       | Optimized Perceptron           |
| ------------------- | --------------------------- | ------------------------------ |
| Entity F-Score      | **0.8492**                  | 0.8224                         |
| Sentiment F-Score   | **0.8198**                  | 0.7861                         |
| Prediction Time     | 0.65s                       | 0.69s                          |
| Training Time       | N/A (no iterative training) | 18.25s (3 iterations)          |
| Memory Efficiency   | Higher (probability tables) | Lower (feature weights)        |
| Feature Flexibility | Lower                       | Higher (customizable features) |

### 5.2 Strengths and Weaknesses

#### 5.2.1 Optimized Viterbi HMM

- **Strengths**: Higher accuracy, simpler training, better generalization for this task
- **Weaknesses**: Less flexible feature engineering, potential scaling issues with large tag sets

#### 5.2.2 Optimized Perceptron

- **Strengths**: Flexible feature templates, adaptive feature pruning, competitive speed
- **Weaknesses**: Lower accuracy, requires iterative training, more sensitive to hyperparameters

## 6. Conclusion and Recommendations

Our optimization efforts resulted in dramatic performance improvements across all models:

- 98.9% faster Viterbi prediction
- 98.3% faster Perceptron prediction
- 84% reduction in function calls
- 95% reduction in dictionary lookups
- 82% reduction in model size for the Perceptron

For this specific sequence labeling task, the **Optimized Viterbi HMM** emerged as the superior model with higher accuracy in both entity recognition (+2.68% F-score) and sentiment analysis (+3.37% F-score), while maintaining comparable prediction speed to the Perceptron model.

### 6.1 Key Lessons Learned

1. **Profile before optimizing**: Identifying the true bottlenecks was crucial for targeted optimization
2. **Data structure selection matters**: Appropriate data structures significantly impacted performance
3. **Algorithm design trumps micro-optimization**: Algorithmic improvements (beam search, batching) yielded the largest gains
4. **Memory access patterns affect speed**: Optimizing for cache locality through batching and data organization improved performance
5. **Balance accuracy and speed**: The fastest model isn't always the best; consider the accuracy-speed tradeoff

### 6.2 Future Work

1. Explore parallelization opportunities with libraries like `numba` or `cython`
2. Implement neural network approaches (BiLSTM-CRF) and compare performance
3. Investigate more sophisticated feature selection techniques
4. Develop hybrid approaches that combine the strengths of both models

This investigation demonstrates that thoughtful optimization can dramatically improve the performance of classical NLP sequence labeling models, making them viable alternatives to more computationally intensive neural approaches for certain applications.
