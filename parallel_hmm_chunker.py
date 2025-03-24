from collections import Counter, defaultdict
import math
import time
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import functools
import os

# Global variables to be accessible in child processes
_g_vocabulary = None
_g_tags = None
_g_smoothed_trans_probs = None
_g_smoothed_emit_probs = None
_g_perceptron = None
_g_specialized_unks = True


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


def get_word_shape(word):
    """
    Extract word shape features (capitalization, digits, etc.)
    """
    if not word or word.startswith('#UNK'):
        return 'UNK'
    if word.isupper():
        return 'ALL_CAPS'
    if word[0].isupper():
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


def viterbi_with_smoothing_worker(sentence_idx_tuple):
    """
    Worker function for parallel Viterbi decoding.
    """
    global _g_vocabulary, _g_tags, _g_smoothed_trans_probs, _g_smoothed_emit_probs, _g_specialized_unks

    idx, sentence = sentence_idx_tuple

    # Adaptive beam size based on sentence length to save computation on longer sentences
    sentence_length = len(sentence)
    if sentence_length < 10:
        beam_size = 5
    elif sentence_length < 20:
        beam_size = 4
    else:
        beam_size = 3  # More aggressive pruning for longer sentences

    n = len(sentence)
    if n == 0:
        return idx, []

    # Initialize DP tables
    viterbi_log = [{} for _ in range(n)]
    backpointer = [{} for _ in range(n)]

    # Process first word
    word = sentence[0]
    if word not in _g_vocabulary:
        if _g_specialized_unks:
            word = classify_token(word)
        else:
            word = '#UNK#'

    for tag in _g_tags:
        # Skip START tag
        if tag == "START":
            continue

        # Get smoothed probabilities
        trans_prob = _g_smoothed_trans_probs["START"][tag]
        emit_prob = _g_smoothed_emit_probs[tag][word]

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
        if word not in _g_vocabulary:
            if _g_specialized_unks:
                word = classify_token(word)
            else:
                word = '#UNK#'

        # Consider only tags that appeared in the previous position's beam
        prev_tags = list(viterbi_log[t-1].keys())

        for tag in _g_tags:
            # Skip START tag
            if tag == "START":
                continue

            max_score = float("-inf")
            best_prev_tag = None

            for prev_tag in prev_tags:  # Only consider tags in the beam
                if prev_tag == "START":
                    continue

                trans_prob = _g_smoothed_trans_probs[prev_tag][tag]

                if trans_prob > 0:
                    score = viterbi_log[t-1][prev_tag] + math.log(trans_prob)

                    if score > max_score:
                        max_score = score
                        best_prev_tag = prev_tag

            # Emission probability
            emit_prob = _g_smoothed_emit_probs[tag][word]

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
        trans_prob = _g_smoothed_trans_probs[tag]["STOP"]

        if trans_prob > 0:
            score = final_score + math.log(trans_prob)

            if score > max_final_score:
                max_final_score = score
                best_final_tag = tag

    # No valid path found, use fallback
    if best_final_tag is None:
        # Just pick the most frequent tag
        tag_frequencies = {tag: sum(
            _g_smoothed_emit_probs[tag].values()) for tag in _g_tags if tag != "START"}
        best_final_tag = max(tag_frequencies.items(), key=lambda x: x[1])[0]
        return idx, [best_final_tag] * n

    # Backtrack to find the best path
    path = [best_final_tag]
    for t in range(n-1, 0, -1):
        if path[0] in backpointer[t]:
            prev_tag = backpointer[t][path[0]]
            path.insert(0, prev_tag)
        else:
            # If backpointer is missing (shouldn't happen with proper beam), use previous tag
            path.insert(0, path[0])

    return idx, path


def predict_tags_parallel(sentences, vocabulary, tags, smoothed_trans_probs, smoothed_emit_probs, specialized_unks=True, num_processes=None):
    """
    Predict tags for sentences in parallel using multiprocessing.
    """
    global _g_vocabulary, _g_tags, _g_smoothed_trans_probs, _g_smoothed_emit_probs, _g_specialized_unks

    # Set globals for worker processes
    _g_vocabulary = vocabulary
    _g_tags = tags
    _g_smoothed_trans_probs = smoothed_trans_probs
    _g_smoothed_emit_probs = smoothed_emit_probs
    _g_specialized_unks = specialized_unks

    # Default to CPU count if not specified
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)

    # Create a pool of worker processes
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Map sentences to worker processes
        results = list(executor.map(
            viterbi_with_smoothing_worker, enumerate(sentences)))

    # Sort results by index and extract predictions
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]


class ParallelOptimizedPerceptron:
    """
    Optimized Structured Perceptron with parallel processing capabilities.
    """

    def __init__(self, tags, vocabulary):
        # Exclude START for simplicity
        self.tags = [tag for tag in tags if tag != "START"]
        self.vocabulary = vocabulary
        self.weights = defaultdict(float)
        self.specialized_unks = True
        self.learning_rate_schedule = None
        self.feature_cache = {}  # Cache for features
        self.beam_size = 5  # Default beam size

        # Set up max processes for parallel processing
        self.num_processes = max(1, multiprocessing.cpu_count() - 1)

    def _process_word(self, word):
        """Process a word for feature extraction"""
        if word not in self.vocabulary:
            return classify_token(word) if self.specialized_unks else '#UNK#'
        return word

    def get_features(self, words, tags):
        """
        Extract features efficiently with minimal computation.
        """
        # Create a cache key
        cache_key = (tuple(words), tuple(tags))

        # Return cached features if available
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        features = Counter()
        prev_tag = "START"

        for i, (word, tag) in enumerate(zip(words, tags)):
            # Process word - minimize function calls
            if word not in self.vocabulary:
                word = classify_token(
                    word) if self.specialized_unks else '#UNK#'

            # Most important features only
            features[('emission', word, tag)] += 1
            features[('transition', prev_tag, tag)] += 1

            # Word shape - compute only for non-UNK words to save time
            if not word.startswith('#UNK'):
                # Use simplified shape features
                if word.isupper():
                    features[('shape', 'ALL_CAPS', tag)] += 1
                elif word and word[0].isupper():
                    features[('shape', 'INIT_CAP', tag)] += 1

            # Suffix - most discriminative feature for new words
            if len(word) > 2 and not word.startswith('#UNK'):
                features[('suffix2', word[-2:], tag)] += 1

            # Position features - very cheap to compute and helpful
            if i == 0:
                features[('position', 'first', tag)] += 1
            elif i == len(words) - 1:
                features[('position', 'last', tag)] += 1

            # Context - only include for short to medium sentences
            if len(words) < 30:  # Skip for very long sentences
                if i > 0:
                    prev_word = self._process_word(words[i-1])
                    features[('prev_word', prev_word, tag)] += 1

                if i < len(words) - 1:
                    next_word = self._process_word(words[i+1])
                    features[('next_word', next_word, tag)] += 1

            prev_tag = tag

        # Final transition to STOP
        features[('transition', prev_tag, 'STOP')] += 1

        # Cache and return
        self.feature_cache[cache_key] = features
        return features

    def _get_position_features(self, words, pos, tag, prev_tag):
        """
        Minimal position-specific features for beam search.
        """
        word = words[pos]
        if word not in self.vocabulary:
            word = classify_token(word) if self.specialized_unks else '#UNK#'

        # Precompute the most common feature weights to avoid dictionary lookups
        emission_weight = self.weights.get(('emission', word, tag), 0)
        transition_weight = self.weights.get(('transition', prev_tag, tag), 0)

        # Start with the sum of most important features
        score = emission_weight + transition_weight

        # Add a few more important features without creating a full Counter
        if not word.startswith('#UNK'):
            if word.isupper():
                score += self.weights.get(('shape', 'ALL_CAPS', tag), 0)
            elif word and word[0].isupper():
                score += self.weights.get(('shape', 'INIT_CAP', tag), 0)

        # Position
        if pos == 0:
            score += self.weights.get(('position', 'first', tag), 0)
        elif pos == len(words) - 1:
            score += self.weights.get(('position', 'last', tag), 0)

        # Suffix
        if len(word) > 2 and not word.startswith('#UNK'):
            score += self.weights.get(('suffix2', word[-2:], tag), 0)

        return score

    def viterbi_decode(self, words):
        """
        Highly optimized Viterbi decoding with dynamic beam size.
        """
        n = len(words)
        if n == 0:
            return []

        # Use adaptive beam size based on sentence length
        beam_size = 5 if n < 15 else (4 if n < 25 else 3)

        # Initialize DP tables
        viterbi = [{} for _ in range(n)]
        backpointer = [{} for _ in range(n)]

        # Score first word with all possible tags
        for tag in self.tags:
            score = self._get_position_features(words, 0, tag, "START")
            viterbi[0][tag] = score
            backpointer[0][tag] = "START"

        # Apply beam search
        if len(viterbi[0]) > beam_size:
            top_tags = sorted(viterbi[0].items(), key=lambda x: x[1], reverse=True)[
                :beam_size]
            viterbi[0] = {tag: score for tag, score in top_tags}
            backpointer[0] = {tag: backpointer[0][tag] for tag, _ in top_tags}

        # Process rest of the words
        for t in range(1, n):
            # Only consider previous tags in the beam
            prev_tags = list(viterbi[t-1].keys())

            for tag in self.tags:
                max_score = float("-inf")
                best_prev_tag = None

                for prev_tag in prev_tags:
                    # Get score for this transition
                    prev_score = viterbi[t-1][prev_tag]
                    score = prev_score + \
                        self._get_position_features(words, t, tag, prev_tag)

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

        # Find best final tag from the beam
        max_final_score = float("-inf")
        best_final_tag = None

        for tag in viterbi[n-1]:
            score = viterbi[n-1][tag] + \
                self.weights.get(('transition', tag, 'STOP'), 0)
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

    def _process_batch(self, batch_sentences, batch_indices, correct_predictions):
        """Process a batch of sentences and compute updates."""
        mistakes = 0
        batch_updates = defaultdict(float)

        for i, sentence in zip(batch_indices, batch_sentences):
            # Skip if correctly predicted before
            if i in correct_predictions:
                continue

            words = [word for word, _ in sentence]
            gold_tags = [tag for _, tag in sentence]

            # Get predicted tags
            pred_tags = self.viterbi_decode(words)

            # If prediction is correct, mark as such and skip
            if pred_tags == gold_tags:
                correct_predictions.add(i)
                continue

            # Update for incorrect prediction
            mistakes += 1

            # Extract features
            gold_features = self.get_features(words, gold_tags)
            pred_features = self.get_features(words, pred_tags)

            # Compute updates
            for feature, count in gold_features.items():
                batch_updates[feature] += count
            for feature, count in pred_features.items():
                batch_updates[feature] -= count

        return mistakes, batch_updates, correct_predictions

    def _parallel_process_batch(self, train_sentences, batch_indices, current_lr, correct_predictions):
        """Process multiple batches in parallel and combine results."""
        # Split batch for multi-processing
        num_processes = min(len(batch_indices), self.num_processes)
        if num_processes <= 1:
            # Fall back to sequential processing for small batches
            batch_sentences = [train_sentences[i] for i in batch_indices]
            return self._process_batch(batch_sentences, batch_indices, correct_predictions)

        # Divide work among processes
        chunks = [[] for _ in range(num_processes)]
        chunk_indices = [[] for _ in range(num_processes)]

        for i, idx in enumerate(batch_indices):
            chunk_idx = i % num_processes
            chunks[chunk_idx].append(train_sentences[idx])
            chunk_indices[chunk_idx].append(idx)

        # Create a separate perceptron instance with same weights for each process
        perceptrons = []
        for _ in range(num_processes):
            p = ParallelOptimizedPerceptron(
                ["START"] + self.tags, self.vocabulary)
            p.weights = self.weights.copy()  # Share current weights
            p.specialized_unks = self.specialized_unks
            perceptrons.append(p)

        # Process each chunk in a separate process
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for i in range(num_processes):
                if chunks[i]:
                    future = executor.submit(
                        _process_batch_worker,
                        perceptrons[i],
                        chunks[i],
                        chunk_indices[i],
                        {idx for idx in correct_predictions if idx in chunk_indices[i]}
                    )
                    futures.append(future)

        # Combine results
        total_mistakes = 0
        combined_updates = defaultdict(float)
        updated_correct = set(correct_predictions)

        for future in futures:
            mistakes, batch_updates, new_correct = future.result()
            total_mistakes += mistakes
            updated_correct.update(new_correct)

            # Scale updates by learning rate and combine
            for feature, update in batch_updates.items():
                combined_updates[feature] += current_lr * update

        return total_mistakes, combined_updates, updated_correct

    def train(self, train_sentences, num_iterations=5, decay_rate=0.8, initial_lr=1.0,
              batch_size=64, averaged=True):
        """
        Train with parallel processing:
        - Process batches in parallel
        - Early stopping for correct predictions
        - Batch updates
        - Learning rate decay
        """
        print(
            f"Training parallel optimized perceptron for {num_iterations} iterations...")
        print(f"Using {self.num_processes} worker processes for training")

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

            num_mistakes = 0
            start_time = time.time()

            # Process in larger batches for parallel execution
            parallel_batch_size = batch_size * 4

            for batch_start in range(0, total_sentences, parallel_batch_size):
                batch_end = min(
                    batch_start + parallel_batch_size, total_sentences)
                batch_indices = indices[batch_start:batch_end]

                # Skip if all sentences in this batch are correctly predicted
                if all(i in correct_predictions for i in batch_indices):
                    continue

                # Process batch with either parallel or sequential implementation
                batch_mistakes, batch_updates, correct_predictions = self._parallel_process_batch(
                    train_sentences, batch_indices, self.learning_rate_schedule[
                        iteration], correct_predictions
                )

                num_mistakes += batch_mistakes

                # Apply updates
                if batch_updates:
                    for feature, update in batch_updates.items():
                        self.weights[feature] += update
                        if averaged:
                            total_weights[feature] += update * \
                                (iteration_count + batch_start)

                # Progress report
                if batch_end % (parallel_batch_size * 2) == 0 or batch_end == total_sentences:
                    elapsed = time.time() - start_time
                    progress = batch_end / total_sentences
                    print(f"  Progress: {batch_end}/{total_sentences} ({progress:.1%}), "
                          f"Time: {elapsed:.1f}s, Mistakes: {num_mistakes}")

            # Increment averaging counter
            if averaged:
                iteration_count += total_sentences

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
                if iteration_count > 0:
                    self.weights[feature] -= total_weights[feature] / \
                        iteration_count

        # Clear caches to save memory
        self.feature_cache.clear()

        print("Training completed.")

    def predict_parallel(self, sentences, num_processes=None):
        """
        Predict tags for sentences in parallel.
        """
        if num_processes is None:
            num_processes = self.num_processes

        # Keep track of original indices
        sentences_with_indices = list(enumerate(sentences))

        # Split data into chunks for parallel processing
        chunks = [[] for _ in range(num_processes)]
        for i, item in enumerate(sentences_with_indices):
            chunks[i % num_processes].append(item)

        results = []

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for chunk in chunks:
                if chunk:
                    future = executor.submit(self._predict_chunk, chunk)
                    futures.append(future)

            for future in futures:
                chunk_results = future.result()
                results.extend(chunk_results)

        # Sort results by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def _predict_chunk(self, chunk):
        """Process a chunk of sentences for prediction."""
        results = []
        for idx, sentence in chunk:
            tags = self.viterbi_decode(sentence)
            results.append((idx, tags))
        return results


def _process_batch_worker(perceptron, sentences, indices, correct_predictions):
    """Worker function for processing batches in parallel."""
    return perceptron._process_batch(sentences, indices, correct_predictions)


def write_output(file_path, words, predictions):
    """
    Write predictions to file in the required format.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence_words, sentence_preds in zip(words, predictions):
            for word, pred in zip(sentence_words, sentence_preds):
                f.write(f"{word}\t{pred}\n")
            f.write("\n")


def batch_data(data, batch_size):
    """Helper function to batch data for progress reporting."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def get_num_processes():
    """Determine reasonable number of processes to use."""
    cpu_count = multiprocessing.cpu_count()
    # Leave 1 CPU for system processes
    return max(1, cpu_count - 1)


if __name__ == "__main__":
    # Determine number of processes to use
    num_processes = get_num_processes()
    print(f"Parallel HMM Chunker with {num_processes} worker processes")

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

    # Predict using Viterbi with smoothing and beam search in parallel
    print("Making predictions with smoothed Viterbi (parallel processing)...")
    start_time = time.time()
    predictions_smooth = predict_tags_parallel(
        dev_sentences, vocabulary, all_tags,
        smoothed_trans_probs, smoothed_emit_probs,
        specialized_unks=use_specialized_unks,
        num_processes=num_processes
    )
    elapsed = time.time() - start_time
    print(f"Parallel Viterbi completed in {elapsed:.2f} seconds")

    # Write output for smoothed Viterbi
    write_output('EN/dev.parallel_smooth.out',
                 dev_sentences, predictions_smooth)
    print("Smoothed HMM Viterbi predictions written to EN/dev.parallel_smooth.out")

    # Train parallel optimized perceptron
    print("\nStarting Parallel Optimized Structured Perceptron Training")

    # Initialize and train perceptron with parallel optimizations
    perceptron = ParallelOptimizedPerceptron(["START"] + all_tags, vocabulary)
    perceptron.train(
        train_sentences,
        num_iterations=6,  # Fewer iterations with early stopping
        decay_rate=0.8,
        initial_lr=1.0,
        batch_size=64  # Will be multiplied for parallel processing
    )

    # Predict on dev data with optimized perceptron in parallel
    print("Making predictions with parallel perceptron...")
    start_time = time.time()
    predictions_optimized = perceptron.predict_parallel(
        dev_sentences, num_processes=num_processes)
    elapsed = time.time() - start_time
    print(f"Parallel perceptron prediction completed in {elapsed:.2f} seconds")

    # Write dev output for optimized perceptron
    write_output('EN/dev.parallel_optimized.out',
                 dev_sentences, predictions_optimized)
    print("Parallel optimized perceptron predictions written to EN/dev.parallel_optimized.out")

    # Try to read test data if available
    try:
        print("Checking for test data...")
        test_sentences = read_unlabeled_data('EN/test.in')

        # Predict on test data using parallel perceptron
        print("Making predictions on test data...")
        test_predictions = perceptron.predict_parallel(
            test_sentences, num_processes=num_processes)

        # Write test output
        write_output('EN/test.parallel_optimized.out',
                     test_sentences, test_predictions)
        print("Parallel optimized perceptron predictions for test written to EN/test.parallel_optimized.out")
    except FileNotFoundError:
        print("Test file not found. Skipping test predictions.")

    print("\nAll parallel predictions completed.")
