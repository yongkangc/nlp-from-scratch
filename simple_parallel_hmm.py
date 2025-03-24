from collections import Counter, defaultdict
import math
import time
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import os


def read_data(file_path):
    """
    Read data from file where each line contains a word and tag separated by a tab,
    and sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split by space or tab
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
    """
    Read unlabeled data where each line contains only a word,
    and sentences are separated by empty lines.
    """
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
    """
    Count word frequencies across all sentences.
    """
    word_freq = Counter()
    for sentence in sentences:
        for word, _ in sentence:
            word_freq[word] += 1
    return word_freq


def classify_token(word):
    """
    Classify tokens into different types based on their characteristics.
    """
    if not word:  # Handle empty string
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
    """
    Replace rare words with specialized #UNK# tokens.
    """
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
    """
    Estimate emission parameters from the modified training data.
    """
    tag_count = defaultdict(int)
    word_tag_count = defaultdict(lambda: defaultdict(int))

    for sentence in modified_sentences:
        for word, tag in sentence:
            tag_count[tag] += 1
            word_tag_count[tag][word] += 1

    return tag_count, word_tag_count


def estimate_transition_params(sentences):
    """
    Estimate transition parameters, including START and STOP transitions.
    """
    transition_count = defaultdict(lambda: defaultdict(int))
    tag_count = defaultdict(int)

    for sentence in sentences:
        tags = ["START"] + [tag for _, tag in sentence] + ["STOP"]
        for i in range(len(tags) - 1):
            current_tag, next_tag = tags[i], tags[i + 1]
            transition_count[current_tag][next_tag] += 1
            tag_count[current_tag] += 1

    return transition_count, tag_count


def smooth_probabilities(transition_count, tag_count, word_tag_count, k=0.01):
    """
    Apply add-k smoothing to transition and emission probabilities.
    """
    # Get all possible tags and vocabulary
    all_tags = list(tag_count.keys())
    vocab = set()
    for tag in word_tag_count:
        for word in word_tag_count[tag]:
            vocab.add(word)

    # Smooth transition probabilities
    smoothed_trans_probs = defaultdict(lambda: defaultdict(float))
    for prev_tag in all_tags:
        denominator = tag_count[prev_tag] + k * len(all_tags)
        for next_tag in all_tags:
            count = transition_count[prev_tag].get(next_tag, 0)
            smoothed_trans_probs[prev_tag][next_tag] = (
                count + k) / denominator

    # Smooth emission probabilities
    smoothed_emit_probs = defaultdict(lambda: defaultdict(float))
    for tag in all_tags:
        denominator = tag_count[tag] + k * len(vocab)
        for word in vocab:
            count = word_tag_count[tag].get(word, 0)
            smoothed_emit_probs[tag][word] = (count + k) / denominator

    return smoothed_trans_probs, smoothed_emit_probs


def viterbi_with_smoothing(sentence, vocabulary, tags, smoothed_trans_probs, smoothed_emit_probs, specialized_unks=True, beam_size=5):
    """
    Viterbi algorithm with smoothed probabilities and beam search.
    """
    # Adaptive beam size based on sentence length
    if len(sentence) > 30:
        beam_size = 3
    elif len(sentence) > 15:
        beam_size = 4

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

    for tag in tags:
        # Skip START tag
        if tag == "START":
            continue

        # Get smoothed probabilities
        trans_prob = smoothed_trans_probs["START"][tag]
        emit_prob = smoothed_emit_probs[tag][word]

        if trans_prob > 0 and emit_prob > 0:
            viterbi_log[0][tag] = math.log(trans_prob) + math.log(emit_prob)
            backpointer[0][tag] = "START"

    # Keep only top beam_size states for first word
    if len(viterbi_log[0]) > beam_size:
        top_tags = sorted(viterbi_log[0].items(), key=lambda x: x[1], reverse=True)[
            :beam_size]
        viterbi_log[0] = {tag: score for tag, score in top_tags}
        backpointer[0] = {tag: backpointer[0][tag] for tag, _ in top_tags}

    # Process rest of the words
    for t in range(1, n):
        word = sentence[t]
        if word not in vocabulary:
            if specialized_unks:
                word = classify_token(word)
            else:
                word = '#UNK#'

        # Consider only tags that appeared in the previous position's beam
        prev_tags = list(viterbi_log[t-1].keys())

        for tag in tags:
            # Skip START tag
            if tag == "START":
                continue

            max_score = float("-inf")
            best_prev_tag = None

            for prev_tag in prev_tags:  # Only consider tags in the beam
                if prev_tag == "START":
                    continue

                trans_prob = smoothed_trans_probs[prev_tag][tag]

                if trans_prob > 0:
                    score = viterbi_log[t-1][prev_tag] + math.log(trans_prob)

                    if score > max_score:
                        max_score = score
                        best_prev_tag = prev_tag

            # Emission probability
            emit_prob = smoothed_emit_probs[tag][word]

            if best_prev_tag is not None and emit_prob > 0:
                viterbi_log[t][tag] = max_score + math.log(emit_prob)
                backpointer[t][tag] = best_prev_tag

        # Beam pruning: keep only the top beam_size states
        if len(viterbi_log[t]) > beam_size:
            top_tags = sorted(viterbi_log[t].items(), key=lambda x: x[1], reverse=True)[
                :beam_size]
            viterbi_log[t] = {tag: score for tag, score in top_tags}
            backpointer[t] = {tag: backpointer[t][tag] for tag, _ in top_tags}

    # Find the most likely end state
    max_final_score = float("-inf")
    best_final_tag = None

    for tag in viterbi_log[n-1]:  # Only consider tags in the final beam
        final_score = viterbi_log[n-1][tag]
        trans_prob = smoothed_trans_probs[tag]["STOP"]

        if trans_prob > 0:
            score = final_score + math.log(trans_prob)

            if score > max_final_score:
                max_final_score = score
                best_final_tag = tag

    # No valid path found, use fallback
    if best_final_tag is None:
        # Just pick the most frequent tag
        best_final_tag = max(tags, key=lambda tag: sum(
            smoothed_emit_probs[tag].values()) if tag != "START" else 0)
        return [best_final_tag] * n

    # Backtrack to find the best path
    path = [best_final_tag]
    for t in range(n-1, 0, -1):
        if path[0] in backpointer[t]:
            prev_tag = backpointer[t][path[0]]
            path.insert(0, prev_tag)
        else:
            # If backpointer is missing (shouldn't happen with proper beam), use previous tag
            path.insert(0, path[0])

    return path


def process_sentence_chunk(args):
    """
    Process a chunk of sentences using Viterbi algorithm.
    This function is designed to be used with multiprocessing.
    """
    chunk_start, chunk_end, sentences, vocabulary, tags, smoothed_trans_probs, smoothed_emit_probs, specialized_unks = args

    results = []
    for i in range(chunk_start, chunk_end):
        if i < len(sentences):
            sentence = sentences[i]
            prediction = viterbi_with_smoothing(
                sentence, vocabulary, tags,
                smoothed_trans_probs, smoothed_emit_probs,
                specialized_unks
            )
            results.append((i, prediction))

    return results


def parallel_viterbi_predict(sentences, vocabulary, tags, smoothed_trans_probs, smoothed_emit_probs,
                             specialized_unks=True, num_processes=None):
    """
    Predict tags for sentences in parallel using a simplified approach.
    """
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)

    total_sentences = len(sentences)
    chunk_size = max(1, total_sentences // num_processes)

    # Prepare chunks
    chunks = []
    for i in range(0, total_sentences, chunk_size):
        chunk_end = min(i + chunk_size, total_sentences)
        chunks.append((i, chunk_end, sentences, vocabulary, tags,
                       smoothed_trans_probs, smoothed_emit_probs, specialized_unks))

    # Process chunks in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for chunk_results in executor.map(process_sentence_chunk, chunks):
            results.extend(chunk_results)

    # Sort results by sentence index
    results.sort(key=lambda x: x[0])

    # Extract predictions in order
    predictions = [result[1] for result in results]

    return predictions


class EnhancedStructuredPerceptron:
    """
    Enhanced Structured Perceptron with better performance.
    """

    def __init__(self, tags, vocabulary):
        # Exclude START for simplicity
        self.tags = [tag for tag in tags if tag != "START"]
        self.vocabulary = vocabulary
        self.weights = defaultdict(float)
        self.specialized_unks = True
        self.feature_cache = {}  # Cache for features

    def get_features(self, words, tags):
        """
        Extract minimal but effective features with caching.
        """
        # Create a cache key
        cache_key = (tuple(words), tuple(tags))

        # Return cached features if available
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        features = Counter()
        prev_tag = "START"

        for i, (word, tag) in enumerate(zip(words, tags)):
            # Process word
            if word not in self.vocabulary:
                word = classify_token(
                    word) if self.specialized_unks else '#UNK#'

            # Most important features
            features[('emission', word, tag)] += 1
            features[('transition', prev_tag, tag)] += 1

            # Word shape for capitalization
            if word.isupper():
                features[('shape', 'ALL_CAPS', tag)] += 1
            elif word and word[0].isupper():
                features[('shape', 'INIT_CAP', tag)] += 1

            # Suffix features - most discriminative for unknown words
            if len(word) > 2 and not word.startswith('#UNK'):
                features[('suffix2', word[-2:], tag)] += 1

            # Position features
            if i == 0:
                features[('position', 'first', tag)] += 1
            elif i == len(words) - 1:
                features[('position', 'last', tag)] += 1

            # Only include minimal context to optimize speed
            if len(words) < 20:  # Skip for long sentences
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

        # Final transition to STOP
        features[('transition', prev_tag, 'STOP')] += 1

        # Cache and return
        self.feature_cache[cache_key] = features
        return features

    def compute_score(self, features):
        """
        Compute score for features based on weights.
        """
        score = 0
        for feature, count in features.items():
            score += self.weights[feature] * count
        return score

    def viterbi_decode(self, words):
        """
        Simplified and optimized Viterbi decoding.
        """
        n = len(words)
        if n == 0:
            return []

        # Adaptive beam size based on sentence length
        beam_size = 5 if n < 15 else (4 if n < 25 else 3)

        # Initialize DP tables
        viterbi = [{} for _ in range(n)]
        backpointer = [{} for _ in range(n)]

        # Process first word by scoring all possible tags
        for tag in self.tags:
            # Compute features for single word with this tag
            features = Counter()
            word = words[0]
            if word not in self.vocabulary:
                word = classify_token(
                    word) if self.specialized_unks else '#UNK#'

            # Add basic features for first position
            features[('emission', word, tag)] += 1
            features[('transition', "START", tag)] += 1

            # Add shape features
            if word.isupper():
                features[('shape', 'ALL_CAPS', tag)] += 1
            elif word and word[0].isupper():
                features[('shape', 'INIT_CAP', tag)] += 1

            # Add position feature
            features[('position', 'first', tag)] += 1

            # Add suffix feature
            if len(word) > 2 and not word.startswith('#UNK'):
                features[('suffix2', word[-2:], tag)] += 1

            score = self.compute_score(features)
            viterbi[0][tag] = score
            backpointer[0][tag] = "START"

        # Beam pruning for first word
        if len(viterbi[0]) > beam_size:
            top_tags = sorted(viterbi[0].items(), key=lambda x: x[1], reverse=True)[
                :beam_size]
            viterbi[0] = {tag: score for tag, score in top_tags}
            backpointer[0] = {tag: backpointer[0][tag] for tag, _ in top_tags}

        # Process rest of words with beam search
        for t in range(1, n):
            # Only consider previous tags in the beam
            prev_tags = list(viterbi[t-1].keys())

            for tag in self.tags:
                max_score = float("-inf")
                best_prev_tag = None

                for prev_tag in prev_tags:
                    # Compute features for this transition
                    features = Counter()
                    word = words[t]
                    if word not in self.vocabulary:
                        word = classify_token(
                            word) if self.specialized_unks else '#UNK#'

                    # Add basic features
                    features[('emission', word, tag)] += 1
                    features[('transition', prev_tag, tag)] += 1

                    # Add shape features
                    if word.isupper():
                        features[('shape', 'ALL_CAPS', tag)] += 1
                    elif word and word[0].isupper():
                        features[('shape', 'INIT_CAP', tag)] += 1

                    # Add position feature for last word
                    if t == n - 1:
                        features[('position', 'last', tag)] += 1

                    # Add suffix feature
                    if len(word) > 2 and not word.startswith('#UNK'):
                        features[('suffix2', word[-2:], tag)] += 1

                    # Get score for this transition
                    transition_score = self.compute_score(features)
                    score = viterbi[t-1][prev_tag] + transition_score

                    if score > max_score:
                        max_score = score
                        best_prev_tag = prev_tag

                if best_prev_tag is not None:
                    viterbi[t][tag] = max_score
                    backpointer[t][tag] = best_prev_tag

            # Beam pruning
            if viterbi[t] and len(viterbi[t]) > beam_size:
                top_tags = sorted(viterbi[t].items(), key=lambda x: x[1], reverse=True)[
                    :beam_size]
                viterbi[t] = {tag: score for tag, score in top_tags}
                backpointer[t] = {tag: backpointer[t][tag]
                                  for tag, _ in top_tags}

        # No valid path found
        if not viterbi[n-1]:
            return [self.tags[0]] * n

        # Find best final tag
        max_final_score = float("-inf")
        best_final_tag = None

        for tag in viterbi[n-1]:
            # Add transition to STOP
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
        """
        Train the perceptron with early stopping and learning rate decay.
        """
        print(
            f"Training enhanced perceptron for {num_iterations} iterations...")

        # Keep track of correct predictions
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
                # Skip if already predicted correctly
                if i in correct_predictions:
                    continue

                sentence = train_sentences[i]
                words = [word for word, _ in sentence]
                gold_tags = [tag for _, tag in sentence]

                # Get predicted tags
                pred_tags = self.viterbi_decode(words)

                # Check if prediction is correct
                if pred_tags == gold_tags:
                    correct_predictions.add(i)
                    continue

                # Update weights for incorrect prediction
                num_mistakes += 1

                # Get features for gold and predicted sequences
                gold_features = self.get_features(words, gold_tags)
                pred_features = self.get_features(words, pred_tags)

                # Update weights
                for feature, count in gold_features.items():
                    self.weights[feature] += current_lr * count

                for feature, count in pred_features.items():
                    self.weights[feature] -= current_lr * count

            # End of iteration statistics
            elapsed = time.time() - start_time
            remaining = len(train_sentences) - len(correct_predictions)
            if remaining > 0:
                mistake_rate = num_mistakes / remaining
            else:
                mistake_rate = 0

            print(
                f"  Mistakes: {num_mistakes}/{remaining} ({mistake_rate:.2%})")
            print(
                f"  Correct: {len(correct_predictions)}/{len(train_sentences)} ({len(correct_predictions)/len(train_sentences):.2%})")
            print(f"  Time: {elapsed:.2f} seconds")

            # Early stopping if most sentences are correct
            if len(correct_predictions) > 0.92 * len(train_sentences):
                print(
                    f"Early stopping: {len(correct_predictions)/len(train_sentences):.2%} sentences correct")
                break

        # Clear cache
        self.feature_cache.clear()
        print("Training completed.")


def process_perceptron_chunk(args):
    """
    Process a chunk of sentences with perceptron for parallel prediction.
    """
    chunk_idx, sentences, weights, vocabulary, tags, specialized_unks = args

    # Create a local perceptron with shared weights
    perceptron = EnhancedStructuredPerceptron(tags, vocabulary)
    perceptron.weights = weights
    perceptron.specialized_unks = specialized_unks

    results = []
    for i, sentence in enumerate(sentences):
        prediction = perceptron.viterbi_decode(sentence)
        results.append((chunk_idx * len(sentences) + i, prediction))

    return results


def parallel_perceptron_predict(perceptron, sentences, num_processes=None):
    """
    Predict tags for sentences in parallel using the perceptron.
    """
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)

    total_sentences = len(sentences)
    chunk_size = (total_sentences + num_processes - 1) // num_processes

    # Split sentences into chunks
    sentence_chunks = []
    for i in range(0, total_sentences, chunk_size):
        end = min(i + chunk_size, total_sentences)
        sentence_chunks.append(sentences[i:end])

    # Prepare arguments for parallel processing
    chunk_args = []
    for i, chunk in enumerate(sentence_chunks):
        args = (i, chunk, perceptron.weights, perceptron.vocabulary,
                ["START"] + perceptron.tags, perceptron.specialized_unks)
        chunk_args.append(args)

    # Process chunks in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for chunk_results in executor.map(process_perceptron_chunk, chunk_args):
            results.extend(chunk_results)

    # Sort results by sentence index
    results.sort(key=lambda x: x[0])

    # Extract predictions
    predictions = [result[1] for result in results]

    return predictions


def write_output(file_path, words, predictions):
    """
    Write predictions to file in the required format.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence_words, sentence_preds in zip(words, predictions):
            for word, pred in zip(sentence_words, sentence_preds):
                f.write(f"{word}\t{pred}\n")
            f.write("\n")


if __name__ == "__main__":
    # Determine number of processes
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Simple Parallel HMM Chunker with {num_processes} worker processes")

    # Step 1: Read the training data
    print("Reading training data...")
    train_sentences = read_data('EN/train')

    # Get word frequencies
    word_freq = get_word_freq(train_sentences)

    # Identify rare words (frequency < 3)
    rare_words = {word for word, freq in word_freq.items() if freq < 3}
    vocabulary = {word for word, freq in word_freq.items() if freq >= 3}

    # Add specialized UNK tokens to vocabulary
    vocabulary.update(['#UNK#', '#UNK-CAPS#', '#UNK-INITCAP#',
                      '#UNK-NUM#', '#UNK-HYPHEN#', '#UNK-PUNCT#'])

    # Use specialized UNK tokens
    use_specialized_unks = True

    # Replace rare words with specialized #UNK# tokens
    print("Preprocessing data with specialized UNK tokens...")
    modified_train_sentences = modify_training_data(
        train_sentences, rare_words, use_specialized_unks)

    # Step 2: Estimate emission parameters
    tag_count, word_tag_count = estimate_emission_params(
        modified_train_sentences)

    # Read dev data
    print("Reading development data...")
    dev_sentences = read_unlabeled_data('EN/dev.in')

    # Step 3: Estimate transition parameters
    transition_count, transition_tag_count = estimate_transition_params(
        train_sentences)

    # Get the set of all possible tags (excluding START and STOP)
    all_tags = [tag for tag in tag_count.keys() if tag !=
                "START" and tag != "STOP"]

    # Apply smoothing to probabilities
    print("Applying add-k smoothing to probabilities...")
    smoothing_k = 0.01
    smoothed_trans_probs, smoothed_emit_probs = smooth_probabilities(
        transition_count, transition_tag_count, word_tag_count, k=smoothing_k)

    # Predict using simplified parallel Viterbi
    print("Making predictions with smoothed Viterbi (parallel processing)...")
    start_time = time.time()
    predictions_smooth = parallel_viterbi_predict(
        dev_sentences, vocabulary, all_tags,
        smoothed_trans_probs, smoothed_emit_probs,
        specialized_unks=use_specialized_unks,
        num_processes=num_processes
    )
    elapsed = time.time() - start_time
    print(f"Parallel Viterbi completed in {elapsed:.2f} seconds")

    # Write output
    write_output('EN/dev.simple_parallel_smooth.out',
                 dev_sentences, predictions_smooth)
    print("Smoothed HMM predictions written to EN/dev.simple_parallel_smooth.out")

    # Train enhanced perceptron
    print("\nTraining enhanced perceptron...")
    perceptron = EnhancedStructuredPerceptron(["START"] + all_tags, vocabulary)
    perceptron.train(
        train_sentences,
        num_iterations=6,
        learning_rate=1.0,
        decay_rate=0.8
    )

    # Predict with enhanced perceptron in parallel
    print("Making predictions with enhanced perceptron (parallel)...")
    start_time = time.time()
    predictions_perceptron = parallel_perceptron_predict(
        perceptron, dev_sentences, num_processes=num_processes
    )
    elapsed = time.time() - start_time
    print(f"Parallel perceptron prediction completed in {elapsed:.2f} seconds")

    # Write output
    write_output('EN/dev.simple_parallel_perceptron.out',
                 dev_sentences, predictions_perceptron)
    print("Enhanced perceptron predictions written to EN/dev.simple_parallel_perceptron.out")

    # Try to read test data
    try:
        print("Checking for test data...")
        test_sentences = read_unlabeled_data('EN/test.in')

        # Predict on test data
        print("Making predictions on test data...")
        test_predictions = parallel_perceptron_predict(
            perceptron, test_sentences, num_processes=num_processes
        )

        # Write test output
        write_output('EN/test.simple_parallel.out',
                     test_sentences, test_predictions)
        print("Enhanced perceptron predictions for test written to EN/test.simple_parallel.out")
    except FileNotFoundError:
        print("Test file not found. Skipping test predictions.")

    print("\nAll predictions completed.")
