from collections import Counter, defaultdict
import math
import time
import re
import random


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
    Returns the most specific UNK type for unknown words.
    """
    if word.isupper():
        return '#UNK-CAPS#'
    if word and word[0].isupper():
        return '#UNK-INITCAP#'
    if any(c.isdigit() for c in word):
        return '#UNK-NUM#'
    if '-' in word:
        return '#UNK-HYPHEN#'
    if any(c in '.,;:!?"()[]{}' for c in word):
        return '#UNK-PUNCT#'
    return '#UNK#'


def get_word_shape(word):
    """
    Extract word shape features (capitalization, digits, etc.)
    """
    if word.startswith('#UNK'):
        return 'UNK'
    if word.isupper():
        return 'ALL_CAPS'
    if word and word[0].isupper():
        return 'INIT_CAP'
    if any(c.isdigit() for c in word):
        return 'HAS_DIGIT'
    if all(not c.isalnum() for c in word):
        return 'NO_ALNUM'
    return 'LOWER'


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


def viterbi_with_smoothing(sentence, tags, smoothed_trans_probs, smoothed_emit_probs, vocabulary, specialized_unks=True, beam_size=5):
    """
    Viterbi algorithm with smoothed probabilities and beam search.
    """
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
        # Skip START tag in tags if present
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
        # Keep backpointers only for tags in the beam
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
            # Keep backpointers only for tags in the beam
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
        tag_frequencies = {
            tag: sum(smoothed_emit_probs[tag].values()) for tag in tags if tag != "START"}
        best_final_tag = max(tag_frequencies.items(), key=lambda x: x[1])[0]
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


class OptimizedStructuredPerceptron:
    """
    Optimized Structured Perceptron for sequence labeling with feature caching and beam search.
    """

    def __init__(self, tags, vocabulary):
        # Exclude START for simplicity
        self.tags = [tag for tag in tags if tag != "START"]
        self.vocabulary = vocabulary
        self.weights = defaultdict(float)
        self.specialized_unks = True
        self.learning_rate_schedule = None
        self.feature_cache = {}  # Cache for features
        self.position_feature_cache = {}  # Cache for position features
        self.beam_size = 5  # Beam size for decoding

    def get_features(self, words, tags):
        """
        Extract rich features with caching.
        """
        # Create a cache key (tuple is hashable)
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

            # Basic features (most important)
            features[('emission', word, tag)] += 1
            features[('transition', prev_tag, tag)] += 1

            # Word shape
            shape = get_word_shape(word)
            features[('shape', shape, tag)] += 1

            # Reduced feature set for speed (keep only the most important features)

            # Prefix/suffix features - only use length 2 for speed
            if len(word) > 2 and not word.startswith('#UNK'):
                features[('suffix2', word[-2:], tag)] += 1

            # Position features - very useful
            if i == 0:
                features[('position', 'first', tag)] += 1
            elif i == len(words) - 1:
                features[('position', 'last', tag)] += 1

            # Context window - keep only immediate context
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

    def _get_position_features(self, words, pos, tag, prev_tag):
        """
        Get features for a specific position, with caching.
        """
        # Create a cache key
        cache_key = (tuple(words), pos, tag, prev_tag)

        # Return cached features if available
        if cache_key in self.position_feature_cache:
            return self.position_feature_cache[cache_key]

        features = Counter()
        word = words[pos]
        if word not in self.vocabulary:
            word = classify_token(word) if self.specialized_unks else '#UNK#'

        # Emission and transition features (most important)
        features[('emission', word, tag)] += 1
        features[('transition', prev_tag, tag)] += 1

        # Word shape
        shape = get_word_shape(word)
        features[('shape', shape, tag)] += 1

        # Reduced feature set for speed

        # Suffix only (often more informative than prefix)
        if len(word) > 2 and not word.startswith('#UNK'):
            features[('suffix2', word[-2:], tag)] += 1

        # Position features
        if pos == 0:
            features[('position', 'first', tag)] += 1
        elif pos == len(words) - 1:
            features[('position', 'last', tag)] += 1

        # Context words (immediate context only)
        if pos > 0:
            prev_word = words[pos-1]
            if prev_word not in self.vocabulary:
                prev_word = classify_token(
                    prev_word) if self.specialized_unks else '#UNK#'
            features[('prev_word', prev_word, tag)] += 1

        if pos < len(words) - 1:
            next_word = words[pos+1]
            if next_word not in self.vocabulary:
                next_word = classify_token(
                    next_word) if self.specialized_unks else '#UNK#'
            features[('next_word', next_word, tag)] += 1

        # Cache and return
        self.position_feature_cache[cache_key] = features
        return features

    def score(self, features):
        """
        Compute score for a feature set based on current weights.
        """
        return sum(self.weights[f] * count for f, count in features.items())

    def viterbi_decode(self, words):
        """
        Use Viterbi algorithm with beam search for efficiency.
        """
        n = len(words)
        if n == 0:
            return []

        # Initialize
        viterbi = [{} for _ in range(n)]
        backpointer = [{} for _ in range(n)]

        # First word
        word = words[0]
        if word not in self.vocabulary:
            word = classify_token(word) if self.specialized_unks else '#UNK#'

        # Score first word with all possible tags
        for tag in self.tags:
            features = self._get_position_features(words, 0, tag, "START")
            score = self.score(features)
            viterbi[0][tag] = score
            backpointer[0][tag] = "START"

        # Apply beam search: keep only top beam_size candidates
        if len(viterbi[0]) > self.beam_size:
            top_tags = sorted(viterbi[0].items(), key=lambda x: x[1], reverse=True)[
                :self.beam_size]
            viterbi[0] = {tag: score for tag, score in top_tags}
            backpointer[0] = {tag: backpointer[0][tag] for tag, _ in top_tags}

        # Rest of the words
        for t in range(1, n):
            # Only consider previous tags in the beam
            prev_tags = list(viterbi[t-1].keys())

            for tag in self.tags:
                max_score = float("-inf")
                best_prev_tag = None

                for prev_tag in prev_tags:
                    # Get score for this transition
                    prev_score = viterbi[t-1][prev_tag]
                    features = self._get_position_features(
                        words, t, tag, prev_tag)
                    score = prev_score + self.score(features)

                    if score > max_score:
                        max_score = score
                        best_prev_tag = prev_tag

                if best_prev_tag is not None:
                    viterbi[t][tag] = max_score
                    backpointer[t][tag] = best_prev_tag

            # Beam pruning: keep only top beam_size states
            if len(viterbi[t]) > self.beam_size:
                top_tags = sorted(viterbi[t].items(), key=lambda x: x[1], reverse=True)[
                    :self.beam_size]
                viterbi[t] = {tag: score for tag, score in top_tags}
                backpointer[t] = {tag: backpointer[t][tag]
                                  for tag, _ in top_tags}

        # Find best final tag from the beam
        if not viterbi[n-1]:  # No valid path found
            return [self.tags[0]] * n

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

    def train(self, train_sentences, num_iterations=5, decay_rate=0.8, initial_lr=1.0,
              batch_size=32, averaged=True):
        """
        Train with optimizations:
        - Early stopping for correct predictions
        - Batch updates
        - Learning rate decay
        - Randomized training order
        """
        print(
            f"Training optimized perceptron for {num_iterations} iterations...")

        # Set learning rate schedule
        self.learning_rate_schedule = [
            initial_lr * (decay_rate ** i) for i in range(num_iterations)]

        # For averaging
        if averaged:
            total_weights = defaultdict(float)
            counts = defaultdict(int)
            iteration_count = 0

        # Keep track of correct predictions to skip them
        correct_predictions = set()

        # Progress tracking
        total_sentences = len(train_sentences)

        # Create a list of indices for random sampling
        indices = list(range(total_sentences))

        for iteration in range(num_iterations):
            print(
                f"Iteration {iteration+1}/{num_iterations} (LR: {self.learning_rate_schedule[iteration]:.4f})")

            # Shuffle indices for this iteration
            random.shuffle(indices)

            # Batch accumulation
            batch_updates = defaultdict(float)
            num_mistakes = 0
            batch_count = 0

            start_time = time.time()

            # Process in batches
            for batch_start in range(0, total_sentences, batch_size):
                batch_end = min(batch_start + batch_size, total_sentences)
                batch_indices = indices[batch_start:batch_end]

                # Process each sentence in the batch
                for i in batch_indices:
                    # Skip if correctly predicted before
                    if i in correct_predictions:
                        continue

                    sentence = train_sentences[i]
                    words = [word for word, _ in sentence]
                    gold_tags = [tag for _, tag in sentence]

                    # Get predicted tags
                    pred_tags = self.viterbi_decode(words)

                    # If prediction is correct, mark as such and skip
                    if pred_tags == gold_tags:
                        correct_predictions.add(i)
                        continue

                    # Update for incorrect prediction
                    num_mistakes += 1
                    batch_count += 1

                    # Extract features
                    gold_features = self.get_features(words, gold_tags)
                    pred_features = self.get_features(words, pred_tags)

                    # Accumulate batch updates
                    current_lr = self.learning_rate_schedule[iteration]
                    for feature, count in gold_features.items():
                        batch_updates[feature] += current_lr * count
                    for feature, count in pred_features.items():
                        batch_updates[feature] -= current_lr * count

                # Apply batch updates at the end of each batch
                if batch_count > 0:
                    # Update weights
                    for feature, update in batch_updates.items():
                        self.weights[feature] += update
                        if averaged:
                            # For averaging, track the update weighted by position
                            total_weights[feature] += update * iteration_count

                    # Reset batch updates
                    batch_updates.clear()

                    # Increment averaging counter
                    if averaged:
                        iteration_count += batch_count

                    batch_count = 0

                # Print progress every few batches
                if (batch_start // batch_size) % 10 == 0:
                    elapsed = time.time() - start_time
                    progress = batch_end / total_sentences
                    print(f"  Progress: {batch_end}/{total_sentences} ({progress:.1%}), "
                          f"Time: {elapsed:.1f}s, Mistakes: {num_mistakes}")

            # End of iteration statistics
            elapsed = time.time() - start_time
            print(f"  Completed with {num_mistakes}/{total_sentences - len(correct_predictions)} mistakes "
                  f"({num_mistakes/(total_sentences - len(correct_predictions) or 1):.2%})")
            print(f"  Total correctly predicted: {len(correct_predictions)}/{total_sentences} "
                  f"({len(correct_predictions)/total_sentences:.2%})")
            print(f"  Time: {elapsed:.2f} seconds")

            # Early stopping if most sentences are correct
            if len(correct_predictions) > 0.95 * total_sentences:
                print(
                    f"Early stopping: {len(correct_predictions)/total_sentences:.2%} sentences correct")
                break

        # Average weights
        if averaged:
            print("Averaging weights...")
            for feature in self.weights:
                if iteration_count > 0:  # Avoid division by zero
                    self.weights[feature] -= total_weights[feature] / \
                        iteration_count

        # Clear caches to save memory
        self.feature_cache.clear()
        self.position_feature_cache.clear()

        print("Training completed.")


def write_output(file_path, words, predictions):
    """
    Write predictions to file in the required format:
    word tag (separated by a tab), with empty lines between sentences.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence_words, sentence_preds in zip(words, predictions):
            for word, pred in zip(sentence_words, sentence_preds):
                f.write(f"{word}\t{pred}\n")
            f.write("\n")


if __name__ == "__main__":
    print("Optimized HMM Chunker with Speed Improvements")

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

    # Predict using Viterbi with smoothing and beam search
    print("Making predictions with smoothed Viterbi (beam search)...")
    predictions_smooth = []
    for idx, sentence in enumerate(dev_sentences):
        if idx % 500 == 0:
            print(f"  Processing sentence {idx}/{len(dev_sentences)}")
        sentence_preds = viterbi_with_smoothing(
            sentence, all_tags, smoothed_trans_probs, smoothed_emit_probs,
            vocabulary, specialized_unks=use_specialized_unks, beam_size=5)
        predictions_smooth.append(sentence_preds)

    # Write output for smoothed Viterbi
    write_output('EN/dev.optimized_smooth.out',
                 dev_sentences, predictions_smooth)

    print("Smoothed HMM Viterbi predictions written to EN/dev.optimized_smooth.out")

    # Train optimized perceptron
    print("\nStarting Optimized Structured Perceptron Training")

    # Initialize and train perceptron with optimizations
    perceptron = OptimizedStructuredPerceptron(all_tags, vocabulary)
    perceptron.train(
        train_sentences,
        num_iterations=8,  # Fewer iterations with early stopping
        decay_rate=0.8,
        initial_lr=1.0,
        batch_size=64  # Use batching for faster updates
    )

    # Predict on dev data with optimized perceptron
    print("Making predictions with optimized perceptron...")
    predictions_optimized = []
    for idx, sentence in enumerate(dev_sentences):
        if idx % 500 == 0:
            print(f"  Processing sentence {idx}/{len(dev_sentences)}")
        pred_tags = perceptron.viterbi_decode(sentence)
        predictions_optimized.append(pred_tags)

    # Write dev output for optimized perceptron
    write_output('EN/dev.optimized.out', dev_sentences, predictions_optimized)

    print("Optimized perceptron predictions written to EN/dev.optimized.out")

    # Try to read test data if available
    try:
        print("Checking for test data...")
        test_sentences = read_unlabeled_data('EN/test.in')

        # Predict on test data using optimized perceptron
        print("Making predictions on test data...")
        test_predictions = []
        for idx, sentence in enumerate(test_sentences):
            if idx % 500 == 0:
                print(f"  Processing sentence {idx}/{len(test_sentences)}")
            pred_tags = perceptron.viterbi_decode(sentence)
            test_predictions.append(pred_tags)

        # Write test output
        write_output('EN/test.optimized.out', test_sentences, test_predictions)

        print("Optimized perceptron predictions for test written to EN/test.optimized.out")
    except FileNotFoundError:
        print("Test file not found. Skipping test predictions.")

    print("\nAll optimized predictions completed.")
