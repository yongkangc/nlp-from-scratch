#!/usr/bin/env python3
"""
50.007 Machine Learning Project
English Phrase Chunking using HMM and Enhanced Models

This script implements four approaches to phrase chunking:
1. Baseline system using emission probabilities
2. HMM with Viterbi decoding
3. K-best Viterbi for 4th-best sequence
4. Enhanced system using Structured Perceptron
"""

import argparse
from collections import Counter, defaultdict
import math
import random
import time


def read_data(file_path):
    """Read labeled data (word-tag pairs) from file."""
    sentences = []
    sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    word, tag = parts[0], parts[1]
                    sentence.append((word, tag))
            else:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
        if sentence:
            sentences.append(sentence)
    return sentences


def read_unlabeled_data(file_path):
    """Read unlabeled data (words only) from file."""
    sentences = []
    sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                word = line
                sentence.append(word)
            else:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
        if sentence:
            sentences.append(sentence)
    return sentences


def get_word_freq(sentences):
    """Count word frequencies across all sentences."""
    word_freq = Counter()
    for sentence in sentences:
        for word, _ in sentence:
            word_freq[word] += 1
    return word_freq


def classify_token(word):
    """Classify unknown tokens into specialized categories."""
    if not word:
        return '#UNK#'
    if word.isupper():
        return '#UNK-CAPS#'
    if word[0].isupper():
        return '#UNK-INITCAP#'
    if any(c.isdigit() for c in word):
        return '#UNK-NUM#'
    if '-' in word:
        return '#UNK-HYPHEN#'
    if any(c in '.,;:!?"()[]{}' for c in word):
        return '#UNK-PUNCT#'
    return '#UNK#'


def modify_training_data(sentences, rare_words, use_specialized_unks=True):
    """Replace rare words with UNK tokens."""
    modified_sentences = []
    for sentence in sentences:
        modified_sentence = []
        for word, tag in sentence:
            if word in rare_words:
                if use_specialized_unks:
                    replacement = classify_token(word)
                else:
                    replacement = '#UNK#'
                modified_sentence.append((replacement, tag))
            else:
                modified_sentence.append((word, tag))
        modified_sentences.append(modified_sentence)
    return modified_sentences


def estimate_emission_params(modified_sentences):
    """Estimate emission parameters using MLE."""
    tag_count = defaultdict(int)
    word_tag_count = defaultdict(lambda: defaultdict(int))

    for sentence in modified_sentences:
        for word, tag in sentence:
            tag_count[tag] += 1
            word_tag_count[tag][word] += 1

    return tag_count, word_tag_count


def estimate_transition_params(sentences):
    """Estimate transition parameters using MLE."""
    transition_count = defaultdict(lambda: defaultdict(int))
    tag_count = defaultdict(int)

    for sentence in sentences:
        tags = ["START"] + [tag for _, tag in sentence] + ["STOP"]
        for i in range(len(tags) - 1):
            current_tag, next_tag = tags[i], tags[i + 1]
            transition_count[current_tag][next_tag] += 1
            tag_count[current_tag] += 1

    return transition_count, tag_count


def predict_baseline(word, vocabulary, tag_count, word_tag_count, specialized_unks=True):
    """Predict tag using only emission probabilities (Part 1)."""
    if word not in vocabulary:
        if specialized_unks:
            word = classify_token(word)
        else:
            word = '#UNK#'

    max_prob = -1
    best_tag = None

    for tag in tag_count:
        if tag in ["START", "STOP"]:
            continue
        count_y_x = word_tag_count[tag].get(word, 0)
        count_y = tag_count[tag]
        prob = count_y_x / count_y if count_y > 0 else 0

        if prob > max_prob:
            max_prob = prob
            best_tag = tag

    # Fallback if no tag found
    if best_tag is None and tag_count:
        best_tag = max(tag_count.keys(), key=lambda tag: tag_count[tag]
                       if tag not in ["START", "STOP"] else 0)

    return best_tag


def viterbi_decode(sentence, vocabulary, tags, transition_count, tag_count,
                   word_tag_count, specialized_unks=True):
    """Implement Viterbi algorithm for HMM decoding (Part 2)."""
    n = len(sentence)
    if n == 0:
        return []

    # Initialize DP tables
    viterbi_log = [{} for _ in range(n)]
    backpointer = [{} for _ in range(n)]

    # Process first word
    word = sentence[0]
    if word not in vocabulary:
        if specialized_unks:
            word = classify_token(word)
        else:
            word = '#UNK#'

    # Initialize first position
    for tag in tags:
        if tag in ["START", "STOP"]:
            continue

        # Probability of starting with this tag
        trans_prob = transition_count["START"].get(tag, 0) / tag_count["START"] \
            if tag_count["START"] > 0 else 0
        # Emission probability
        emit_prob = word_tag_count[tag].get(word, 0) / tag_count[tag] \
            if tag_count[tag] > 0 else 0

        if trans_prob > 0 and emit_prob > 0:
            viterbi_log[0][tag] = math.log(trans_prob) + math.log(emit_prob)
            backpointer[0][tag] = "START"

    # Process rest of the words
    for t in range(1, n):
        word = sentence[t]
        if word not in vocabulary:
            if specialized_unks:
                word = classify_token(word)
            else:
                word = '#UNK#'

        for tag in tags:
            if tag in ["START", "STOP"]:
                continue

            max_score = float("-inf")
            best_prev_tag = None

            for prev_tag in tags:
                if prev_tag in ["START", "STOP"]:
                    continue

                if viterbi_log[t-1].get(prev_tag, float("-inf")) == float("-inf"):
                    continue

                trans_prob = transition_count[prev_tag].get(tag, 0) / tag_count[prev_tag] \
                    if tag_count[prev_tag] > 0 else 0

                if trans_prob > 0:
                    score = viterbi_log[t-1][prev_tag] + math.log(trans_prob)
                    if score > max_score:
                        max_score = score
                        best_prev_tag = prev_tag

            # Emission probability
            emit_prob = word_tag_count[tag].get(word, 0) / tag_count[tag] \
                if tag_count[tag] > 0 else 0

            if best_prev_tag is not None and emit_prob > 0:
                viterbi_log[t][tag] = max_score + math.log(emit_prob)
                backpointer[t][tag] = best_prev_tag

    # Find best final tag
    max_final_score = float("-inf")
    best_final_tag = None

    for tag in tags:
        if tag in ["START", "STOP"]:
            continue

        final_score = viterbi_log[n-1].get(tag, float("-inf"))
        if final_score == float("-inf"):
            continue

        trans_prob = transition_count[tag].get("STOP", 0) / tag_count[tag] \
            if tag_count[tag] > 0 else 0

        if trans_prob > 0:
            score = final_score + math.log(trans_prob)
            if score > max_final_score:
                max_final_score = score
                best_final_tag = tag

    # Fallback if no valid path found
    if best_final_tag is None:
        best_final_tag = max(tags, key=lambda tag: tag_count[tag]
                             if tag not in ["START", "STOP"] else 0)
        return [best_final_tag] * n

    # Backtrack to find the best path
    path = [best_final_tag]
    for t in range(n-1, 0, -1):
        prev_tag = backpointer[t][path[0]]
        path.insert(0, prev_tag)

    return path


def k_best_viterbi(sentence, vocabulary, tags, transition_count, tag_count,
                   word_tag_count, k=3, specialized_unks=True):
    """Find the k-best sequences using modified Viterbi (Part 3)."""
    n = len(sentence)
    if n == 0:
        return []

    # Initialize DP tables for k-best paths
    # dp[t][y] = list of (score, backpointer, path_idx) tuples
    dp = [{tag: [] for tag in tags if tag not in ["START", "STOP"]}
          for _ in range(n)]

    # Process first word
    word = sentence[0]
    if word not in vocabulary:
        if specialized_unks:
            word = classify_token(word)
        else:
            word = '#UNK#'

    # Initialize first position
    for tag in tags:
        if tag in ["START", "STOP"]:
            continue

        trans_prob = transition_count["START"].get(tag, 0) / tag_count["START"] \
            if tag_count["START"] > 0 else 0
        emit_prob = word_tag_count[tag].get(word, 0) / tag_count[tag] \
            if tag_count[tag] > 0 else 0

        if trans_prob > 0 and emit_prob > 0:
            log_prob = math.log(trans_prob) + math.log(emit_prob)
            dp[0][tag].append((log_prob, "START", 0))

    # Process rest of the words
    for t in range(1, n):
        word = sentence[t]
        if word not in vocabulary:
            if specialized_unks:
                word = classify_token(word)
            else:
                word = '#UNK#'

        for tag in tags:
            if tag in ["START", "STOP"]:
                continue

            # Collect all possible extensions
            candidates = []
            for prev_tag in tags:
                if prev_tag in ["START", "STOP"]:
                    continue

                # Try all paths from previous state
                for prev_path_idx, (prev_score, _, _) in enumerate(dp[t-1][prev_tag]):
                    if prev_score == float("-inf"):
                        continue

                    trans_prob = transition_count[prev_tag].get(tag, 0) / tag_count[prev_tag] \
                        if tag_count[prev_tag] > 0 else 0
                    if trans_prob > 0:
                        score = prev_score + math.log(trans_prob)
                        candidates.append((score, prev_tag, prev_path_idx))

            # Apply emission probability and keep top k
            emit_prob = word_tag_count[tag].get(word, 0) / tag_count[tag] \
                if tag_count[tag] > 0 else 0
            if emit_prob > 0:
                log_emit = math.log(emit_prob)
                candidates = [(score + log_emit, prev_tag, prev_path_idx)
                              for score, prev_tag, prev_path_idx in candidates]
                candidates.sort(reverse=True)
                dp[t][tag] = candidates[:k]

    # Find k-best final paths
    final_candidates = []
    for tag in tags:
        if tag in ["START", "STOP"]:
            continue

        for path_idx, (score, _, _) in enumerate(dp[n-1][tag]):
            if score == float("-inf"):
                continue

            trans_prob = transition_count[tag].get("STOP", 0) / tag_count[tag] \
                if tag_count[tag] > 0 else 0
            if trans_prob > 0:
                final_score = score + math.log(trans_prob)
                final_candidates.append((final_score, tag, path_idx))

    # Sort and get the kth best
    final_candidates.sort(reverse=True)

    # Fallback if fewer than k paths
    if not final_candidates or len(final_candidates) <= k:
        most_common_tag = max(tags, key=lambda tag: tag_count[tag]
                              if tag not in ["START", "STOP"] else 0)
        return [most_common_tag] * n

    # Extract the kth best path
    _, last_tag, path_idx = final_candidates[k]

    # Backtrack to reconstruct the path
    path = [last_tag]
    for t in range(n-1, 0, -1):
        _, prev_tag, path_idx = dp[t][path[0]][path_idx]
        path.insert(0, prev_tag)

    return path

def evaluate(gold_path, pred_path):
    gold_chunks = read_chunks(gold_path)
    pred_chunks = read_chunks(pred_path)

    correct = 0
    total_pred = 0
    total_gold = 0

    for gold_sent, pred_sent in zip(gold_chunks, pred_chunks):
        gold = extract_entity_chunks(gold_sent)
        pred = extract_entity_chunks(pred_sent)
        correct += len(gold & pred)
        total_pred += len(pred)
        total_gold += len(gold)

    precision = correct / total_pred if total_pred > 0 else 0
    recall = correct / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1



class EnhancedPerceptron:
    """Enhanced Structured Perceptron for sequence labeling (Part 4)."""

    def __init__(self, tags, vocabulary):
        self.tags = [tag for tag in tags if tag not in ["START", "STOP"]]
        self.vocabulary = vocabulary
        self.weights = defaultdict(float)
        self.specialized_unks = True

    def get_features(self, words, tags):
        """Extract rich features for the perceptron."""
        features = Counter()
        prev_tag = "START"

        for i, (word, tag) in enumerate(zip(words, tags)):
            if word not in self.vocabulary:
                word = classify_token(
                    word) if self.specialized_unks else '#UNK#'

            # Basic features
            features[('emission', word, tag)] += 1
            features[('transition', prev_tag, tag)] += 1

            # Word shape features
            if word.isupper():
                features[('shape', 'ALL_CAPS', tag)] += 1
            elif word and word[0].isupper():
                features[('shape', 'INIT_CAP', tag)] += 1

            # Suffix features
            if len(word) > 2 and not word.startswith('#UNK'):
                features[('suffix2', word[-2:], tag)] += 1

            # Position features
            if i == 0:
                features[('position', 'first', tag)] += 1
            elif i == len(words) - 1:
                features[('position', 'last', tag)] += 1

            # Context features (for shorter sentences)
            if len(words) < 20:
                if i > 0:
                    prev_word = words[i-1]
                    if prev_word not in self.vocabulary:
                        prev_word = classify_token(
                            prev_word) if self.specialized_unks else '#UNK#'
                    features[('prev_word', prev_word, tag)] += 1

                if i < len(words) - 1:
                    next_word = words[i+1]
                    if next_word not in self.vocabulary:
                        next_word = classify_token(
                            next_word) if self.specialized_unks else '#UNK#'
                    features[('next_word', next_word, tag)] += 1

            prev_tag = tag

        features[('transition', prev_tag, 'STOP')] += 1
        return features

    def score(self, features):
        """Compute score for a feature set."""
        return sum(self.weights[f] * count for f, count in features.items())

    def viterbi_decode(self, words):
        """Viterbi decoding with feature weights."""
        n = len(words)
        if n == 0:
            return []

        # Adaptive beam size
        beam_size = 5 if n < 15 else (4 if n < 25 else 3)

        # Initialize
        viterbi = [{} for _ in range(n)]
        backpointer = [{} for _ in range(n)]

        # First word
        word = words[0]
        if word not in self.vocabulary:
            word = classify_token(word) if self.specialized_unks else '#UNK#'

        # Score all tags for first word
        for tag in self.tags:
            features = Counter()
            features[('emission', word, tag)] += 1
            features[('transition', "START", tag)] += 1

            if word.isupper():
                features[('shape', 'ALL_CAPS', tag)] += 1
            elif word and word[0].isupper():
                features[('shape', 'INIT_CAP', tag)] += 1

            features[('position', 'first', tag)] += 1

            if len(word) > 2 and not word.startswith('#UNK'):
                features[('suffix2', word[-2:], tag)] += 1

            score = self.score(features)
            viterbi[0][tag] = score
            backpointer[0][tag] = "START"

        # Beam pruning for first word
        if len(viterbi[0]) > beam_size:
            top_tags = sorted(viterbi[0].items(), key=lambda x: x[1], reverse=True)[
                :beam_size]
            viterbi[0] = {tag: score for tag, score in top_tags}
            backpointer[0] = {tag: backpointer[0][tag] for tag, _ in top_tags}

        # Rest of words
        for t in range(1, n):
            word = words[t]
            if word not in self.vocabulary:
                word = classify_token(
                    word) if self.specialized_unks else '#UNK#'

            prev_tags = list(viterbi[t-1].keys())

            for tag in self.tags:
                max_score = float("-inf")
                best_prev_tag = None

                for prev_tag in prev_tags:
                    features = Counter()
                    features[('emission', word, tag)] += 1
                    features[('transition', prev_tag, tag)] += 1

                    if word.isupper():
                        features[('shape', 'ALL_CAPS', tag)] += 1
                    elif word and word[0].isupper():
                        features[('shape', 'INIT_CAP', tag)] += 1

                    if t == n - 1:
                        features[('position', 'last', tag)] += 1

                    if len(word) > 2 and not word.startswith('#UNK'):
                        features[('suffix2', word[-2:], tag)] += 1

                    score = viterbi[t-1][prev_tag] + self.score(features)
                    if score > max_score:
                        max_score = score
                        best_prev_tag = prev_tag

                if best_prev_tag is not None:
                    viterbi[t][tag] = max_score
                    backpointer[t][tag] = best_prev_tag

            # Beam pruning
            if len(viterbi[t]) > beam_size:
                top_tags = sorted(viterbi[t].items(), key=lambda x: x[1], reverse=True)[
                    :beam_size]
                viterbi[t] = {tag: score for tag, score in top_tags}
                backpointer[t] = {tag: backpointer[t][tag]
                                  for tag, _ in top_tags}

        # No valid path
        if not viterbi[n-1]:
            return [self.tags[0]] * n

        # Find best final tag
        max_final_score = float("-inf")
        best_final_tag = None

        for tag in viterbi[n-1]:
            score = viterbi[n-1][tag] + \
                self.weights[('transition', tag, 'STOP')]
            if score > max_final_score:
                max_final_score = score
                best_final_tag = tag

        if best_final_tag is None:
            return [self.tags[0]] * n

        # Backtrack
        path = [best_final_tag]
        for t in range(n-1, 0, -1):
            prev_tag = backpointer[t][path[0]]
            path.insert(0, prev_tag)

        return path

    def train(self, train_sentences, num_iterations=5, learning_rate=1.0, decay_rate=0.8):
        """Train the perceptron with early stopping and learning rate decay."""
        print(
            f"Training enhanced perceptron for {num_iterations} iterations...")

        # Track correct predictions
        correct_predictions = set()

        for iteration in range(num_iterations):
            current_lr = learning_rate * (decay_rate ** iteration)
            print(
                f"Iteration {iteration+1}/{num_iterations} (LR: {current_lr:.4f})")

            start_time = time.time()
            num_mistakes = 0

            # Shuffle training data
            indices = list(range(len(train_sentences)))
            random.shuffle(indices)

            for i in indices:
                if i in correct_predictions:
                    continue

                sentence = train_sentences[i]
                words = [word for word, _ in sentence]
                gold_tags = [tag for _, tag in sentence]

                pred_tags = self.viterbi_decode(words)

                if pred_tags == gold_tags:
                    correct_predictions.add(i)
                    continue

                num_mistakes += 1

                # Update weights
                gold_features = self.get_features(words, gold_tags)
                pred_features = self.get_features(words, pred_tags)

                for feature, count in gold_features.items():
                    self.weights[feature] += current_lr * count

                for feature, count in pred_features.items():
                    self.weights[feature] -= current_lr * count

            elapsed = time.time() - start_time
            remaining = len(train_sentences) - len(correct_predictions)
            mistake_rate = num_mistakes / remaining if remaining > 0 else 0

            print(
                f"  Mistakes: {num_mistakes}/{remaining} ({mistake_rate:.2%})")
            print(f"  Correct: {len(correct_predictions)}/{len(train_sentences)} "
                  f"({len(correct_predictions)/len(train_sentences):.2%})")
            print(f"  Time: {elapsed:.2f} seconds")

            # Early stopping
            if len(correct_predictions) > 0.92 * len(train_sentences):
                print(f"Early stopping: {len(correct_predictions)/len(train_sentences):.2%} "
                      "sentences correct")
                break

        print("Training completed.")


def write_output(file_path, words, predictions):
    """Write predictions to file in the required format."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence_words, sentence_preds in zip(words, predictions):
            for word, pred in zip(sentence_words, sentence_preds):
                f.write(f"{word}\t{pred}\n")
            f.write("\n")


def main():
    """Main function to run the chunker."""
    parser = argparse.ArgumentParser(description="English Phrase Chunking")
    parser.add_argument('--part', type=int, required=True, choices=[1, 2, 3, 4],
                        help="Which part to run (1: Baseline, 2: Viterbi, "
                        "3: 4th-Best, 4: Enhanced)")
    parser.add_argument('--test', action='store_true',
                        help="Run on test data (only for Part 4)")
    args = parser.parse_args()

    # Read training data
    print("Reading training data...")
    train_sentences = read_data('EN/train')

    # Get word frequencies and identify rare words
    word_freq = get_word_freq(train_sentences)
    k_for_smoothing = 3  # Default value; you can manually change it
    rare_words = {word for word, freq in word_freq.items() if freq < k_for_smoothing}
    vocabulary = {word for word, freq in word_freq.items() if freq >= 3}
    vocabulary.update(['#UNK#', '#UNK-CAPS#', '#UNK-INITCAP#',
                      '#UNK-NUM#', '#UNK-HYPHEN#', '#UNK-PUNCT#'])

    # Modify training data with UNK tokens
    print("Processing training data...")
    modified_train_sentences = modify_training_data(
        train_sentences, rare_words)

    # Estimate parameters
    tag_count, word_tag_count = estimate_emission_params(
        modified_train_sentences)
    transition_count, transition_tag_count = estimate_transition_params(
        train_sentences)

    # Get all tags (excluding START/STOP for prediction)
    all_tags = [tag for tag in tag_count.keys() if tag not in [
        "START", "STOP"]]

    # Read development data
    print("Reading development data...")
    dev_sentences = read_unlabeled_data('EN/dev.in')

    if args.part == 1:
    print("Running Baseline system with multiple k smoothing values...")
    
    smoothing_ks = [1, 2, 3, 5, 10]  # List of k values to test
    
    for k_for_smoothing in smoothing_ks:
        print(f"\nTesting k = {k_for_smoothing}...")
        
        # Read training data
        train_sentences = read_data('EN/train')
        
        # Build vocabulary and apply rare word handling
        word_freq = get_word_freq(train_sentences)
        rare_words = {word for word, freq in word_freq.items() if freq < k_for_smoothing}
        modified_train_sentences = modify_training_data(train_sentences, rare_words)
        
        # Estimate emission parameters
        tag_count, word_tag_count = estimate_emission_params(modified_train_sentences)
        
        # Read development data
        dev_sentences = read_unlabeled_data('EN/dev.in')
        
        # Predict tags for development set
        predictions = []
        vocabulary = {word for sentence in modified_train_sentences for word, _ in sentence}
        
        for sentence in dev_sentences:
            sentence_preds = [predict_baseline(word, vocabulary, tag_count, word_tag_count)
                              for word in sentence]
            predictions.append(sentence_preds)
        
        # Write predictions to file named with k
        output_filename = f'EN/dev.p1.k{k_for_smoothing}.out'
        write_output(output_filename, dev_sentences, predictions)
        
        # Evaluate the output
        try:
            precision, recall, f1 = evaluate('EN/dev.out', output_filename)
            print(f"Results for k={k_for_smoothing}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        except Exception as e:
            print(f"Could not evaluate for k={k_for_smoothing}: {e}")

    elif args.part == 2:
        print("Running Viterbi algorithm...")
        predictions = []
        for sentence in dev_sentences:
            sentence_preds = viterbi_decode(sentence, vocabulary, all_tags,
                                            transition_count, transition_tag_count,
                                            word_tag_count)
            predictions.append(sentence_preds)
        write_output('EN/dev.p2.out', dev_sentences, predictions)
        print("Viterbi predictions written to EN/dev.p2.out")

    elif args.part == 3:
        print("Finding 4th-best sequences...")
        predictions = []
        for sentence in dev_sentences:
            sentence_preds = k_best_viterbi(sentence, vocabulary, all_tags,
                                            transition_count, transition_tag_count,
                                            word_tag_count, k=3)
            predictions.append(sentence_preds)
        write_output('EN/dev.p3.out', dev_sentences, predictions)
        print("4th-best predictions written to EN/dev.p3.out")

    elif args.part == 4:
        print("Running enhanced system...")
        perceptron = EnhancedPerceptron(all_tags, vocabulary)
        perceptron.train(train_sentences, num_iterations=5,
                         learning_rate=1.0, decay_rate=0.8)

        if args.test:
            try:
                print("Reading test data...")
                test_sentences = read_unlabeled_data('EN/test.in')
                predictions = []
                for sentence in test_sentences:
                    pred_tags = perceptron.viterbi_decode(sentence)
                    predictions.append(pred_tags)
                write_output('EN/test.p4.out', test_sentences, predictions)
                print("Enhanced predictions written to EN/test.p4.out")
            except FileNotFoundError:
                print("Test file not found.")
        else:
            predictions = []
            for sentence in dev_sentences:
                pred_tags = perceptron.viterbi_decode(sentence)
                predictions.append(pred_tags)
            write_output('EN/dev.p4.out', dev_sentences, predictions)
            print("Enhanced predictions written to EN/dev.p4.out")


if __name__ == "__main__":
    main()
