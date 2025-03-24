from collections import Counter, defaultdict
import math
import time
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
    Optimized for performance with reduced conditional checks.
    """
    # Fast path for common case - empty string
    if not word:
        return '#UNK#'

    # Check capitalization - most frequent distinction
    # Avoid repeated calls to isupper() and first character access
    if word.isupper():
        return '#UNK-CAPS#'

    # Cache first character for reuse
    first_char = word[0]
    if first_char.isupper():
        return '#UNK-INITCAP#'

    # Use caching to avoid repeated iterations through the string
    has_digit = False
    has_hyphen = '-' in word
    if has_hyphen:
        return '#UNK-HYPHEN#'

    # Optimize checking for digits and punctuation
    for c in word:
        if c.isdigit():
            return '#UNK-NUM#'

    # Check for punctuation - only if previous checks failed
    punctuation = '.,;:!?"()[]{}'
    for c in word:
        if c in punctuation:
            return '#UNK-PUNCT#'

    # Default case
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
    tag_count = {}
    word_tag_count = {}

    # First pass: count tags
    for sentence in modified_sentences:
        for _, tag in sentence:
            tag_count[tag] = tag_count.get(tag, 0) + 1

    # Second pass: count (word, tag) pairs
    for tag in tag_count:
        word_tag_count[tag] = {}

    for sentence in modified_sentences:
        for word, tag in sentence:
            if tag in word_tag_count:
                word_tag_count[tag][word] = word_tag_count[tag].get(
                    word, 0) + 1

    return tag_count, word_tag_count


def estimate_transition_params(sentences):
    """
    Estimate transition parameters, including START and STOP transitions.
    """
    transition_count = {}
    tag_count = {"START": 0}

    # First pass: count tags including START
    for sentence in sentences:
        tag_count["START"] += 1
        for _, tag in sentence:
            tag_count[tag] = tag_count.get(tag, 0) + 1

    # Initialize transition counts
    for tag1 in tag_count:
        transition_count[tag1] = {}
        for tag2 in tag_count:
            if tag2 != "START" and tag1 != "STOP":
                transition_count[tag1][tag2] = 0

    # Count transitions
    for sentence in sentences:
        prev_tag = "START"
        for _, tag in sentence:
            transition_count[prev_tag][tag] += 1
            prev_tag = tag
        # Add STOP transition
        if prev_tag in transition_count:
            transition_count[prev_tag]["STOP"] = transition_count[prev_tag].get(
                "STOP", 0) + 1

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

    # Smooth transition probabilities - use defaultdict for safer access
    smoothed_trans_probs = defaultdict(lambda: defaultdict(float))
    for prev_tag in all_tags:
        if prev_tag == "STOP":  # Skip STOP as it doesn't transition
            continue

        # Exclude STOP from denominator
        denominator = tag_count[prev_tag] + k * (len(all_tags) - 1)

        for next_tag in all_tags:
            if next_tag == "START":  # Nothing transitions to START
                continue

            count = transition_count.get(prev_tag, {}).get(next_tag, 0)
            smoothed_trans_probs[prev_tag][next_tag] = (
                count + k) / denominator

    # Smooth emission probabilities - use defaultdict for safer access
    smoothed_emit_probs = defaultdict(lambda: defaultdict(float))
    for tag in all_tags:
        if tag in ["START", "STOP"]:  # START and STOP don't emit
            continue

        denominator = tag_count[tag] + k * len(vocab)

        for word in vocab:
            count = word_tag_count.get(tag, {}).get(word, 0)
            smoothed_emit_probs[tag][word] = (count + k) / denominator

    return smoothed_trans_probs, smoothed_emit_probs


def viterbi_with_smoothing(sentence, vocabulary, tags, smoothed_trans_probs, smoothed_emit_probs, specialized_unks=True):
    """
    Viterbi algorithm with smoothed probabilities and adaptive beam search.
    """
    n = len(sentence)
    if n == 0:
        return []

    # Adaptive beam size based on sentence length - more aggressive pruning
    if n > 30:
        beam_size = 3
    elif n > 15:
        beam_size = 4
    else:
        beam_size = 5

    # Small constant to avoid log(0)
    eps = 1e-10

    # Initialize DP tables
    viterbi_log = [{} for _ in range(n)]
    backpointer = [{} for _ in range(n)]

    # Process and cache first word to avoid repeated lookups
    word = sentence[0]
    if word not in vocabulary:
        word = classify_token(word) if specialized_unks else '#UNK#'

    # Get START transitions for all tags once
    start_trans = smoothed_trans_probs["START"]

    # Process first word more efficiently
    for tag in tags:
        if tag in ["START", "STOP"]:
            continue

        # Direct access to avoid repeated dictionary lookups
        trans_prob = start_trans.get(tag, 0)
        emit_prob = smoothed_emit_probs[tag].get(word, 0)

        # Faster log calculation with offset
        score = math.log(trans_prob + eps) + math.log(emit_prob + eps)
        viterbi_log[0][tag] = score
        backpointer[0][tag] = "START"

    # Faster beam pruning for first word
    if len(viterbi_log[0]) > beam_size:
        # Pre-allocate sorted array for better performance
        items = list(viterbi_log[0].items())
        items.sort(key=lambda x: x[1], reverse=True)

        viterbi_log[0] = {tag: score for tag, score in items[:beam_size]}
        backpointer[0] = {tag: backpointer[0][tag]
                          for tag, _ in items[:beam_size]}

    # Pre-process all words to avoid repeated operations
    processed_words = [word]  # First word already processed
    for i in range(1, n):
        word = sentence[i]
        if word not in vocabulary:
            word = classify_token(word) if specialized_unks else '#UNK#'
        processed_words.append(word)

    # Process rest of the words with optimized operations
    for t in range(1, n):
        word = processed_words[t]

        # Cache relevant emission probabilities for current word
        word_emissions = {}
        for tag in tags:
            if tag not in ["START", "STOP"]:
                word_emissions[tag] = smoothed_emit_probs[tag].get(word, 0)

        # Cache previous beam tags once as tuple (faster than repeated list conversion)
        prev_tags = tuple(viterbi_log[t-1].keys())

        for tag in tags:
            if tag in ["START", "STOP"]:
                continue

            max_score = float("-inf")
            best_prev_tag = None

            # Direct access to transitions for current tag
            tag_trans = smoothed_trans_probs

            # Optimize inner loop - most computationally intensive part
            for prev_tag in prev_tags:
                # Get transition probability only once per prev_tag
                trans_prob = tag_trans[prev_tag].get(tag, 0)
                if trans_prob > 0:  # Skip zero probability transitions
                    prev_score = viterbi_log[t-1][prev_tag]
                    score = prev_score + math.log(trans_prob + eps)

                    # Faster max check
                    if score > max_score:
                        max_score = score
                        best_prev_tag = prev_tag

            # Use cached emission probability
            emit_prob = word_emissions.get(tag, 0)

            if best_prev_tag is not None:
                viterbi_log[t][tag] = max_score + math.log(emit_prob + eps)
                backpointer[t][tag] = best_prev_tag

        # Optimize beam pruning with pre-allocation
        if len(viterbi_log[t]) > beam_size:
            items = list(viterbi_log[t].items())
            items.sort(key=lambda x: x[1], reverse=True)

            # Use dict comprehensions for faster creation
            viterbi_log[t] = {tag: score for tag, score in items[:beam_size]}
            backpointer[t] = {tag: backpointer[t][tag]
                              for tag, _ in items[:beam_size]}

    # Fast final path selection
    if not viterbi_log[n-1]:
        # Fallback to most common tag if no valid path
        tag_counts = {tag: 1 for tag in tags if tag not in ["START", "STOP"]}
        best_final_tag = next(iter(tag_counts.keys()))
        return [best_final_tag] * n

    # Optimize final state selection
    max_final_score = float("-inf")
    best_final_tag = None
    stop_trans = smoothed_trans_probs

    for tag in viterbi_log[n-1]:
        final_score = viterbi_log[n-1][tag]
        trans_prob = stop_trans[tag].get("STOP", 0)

        score = final_score + math.log(trans_prob + eps)
        if score > max_final_score:
            max_final_score = score
            best_final_tag = tag

    # No valid path found, use fallback
    if best_final_tag is None:
        # Use most common tag
        tag_counts = {tag: 1 for tag in tags if tag not in ["START", "STOP"]}
        best_final_tag = next(iter(tag_counts.keys()))
        return [best_final_tag] * n

    # Pre-allocate path array for faster backtracking
    path = [None] * n
    path[n-1] = best_final_tag

    # Efficient backtracking
    for t in range(n-2, -1, -1):
        next_tag = path[t+1]
        path[t] = backpointer[t+1][next_tag]

    return path


class OptimizedPerceptron:
    """
    Optimized Structured Perceptron implementation.
    """

    def __init__(self, tags, vocabulary):
        # Exclude START and STOP from prediction tags
        self.tags = [tag for tag in tags if tag not in ["START", "STOP"]]
        self.vocabulary = vocabulary
        self.weights = {}  # Use regular dict for speed
        self.specialized_unks = True
        self.feature_cache = {}

    def get_features(self, words, tags):
        """
        Extract minimal but effective features with optimized caching.
        """
        # Create a cache key
        cache_key = (tuple(words), tuple(tags))

        # Return cached features if available - fast cache lookup
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        # Pre-allocate Counter with estimated size to reduce resizing
        features = Counter()
        prev_tag = "START"

        # Pre-process words once to avoid repeated vocabulary checks
        processed_words = []
        for word in words:
            if word not in self.vocabulary:
                word = classify_token(
                    word) if self.specialized_unks else '#UNK#'
            processed_words.append(word)

        # Pre-compute word properties (capitalization, etc.) once
        word_properties = []
        for word in processed_words:
            is_upper = word.isupper()
            is_init_cap = not is_upper and word and word[0].isupper()
            is_unk = word.startswith('#UNK')
            suffix = word[-2:] if len(word) > 2 and not is_unk else None

            word_properties.append((is_upper, is_init_cap, is_unk, suffix))

        # Use direct iteration with index to avoid zip()
        n = len(words)
        for i in range(n):
            word = processed_words[i]
            tag = tags[i]
            is_upper, is_init_cap, is_unk, suffix = word_properties[i]

            # Add core features efficiently
            features[('emission', word, tag)] += 1
            features[('transition', prev_tag, tag)] += 1

            # Add shape features - use pre-computed properties
            if is_upper:
                features[('shape', 'ALL_CAPS', tag)] += 1
            elif is_init_cap:
                features[('shape', 'INIT_CAP', tag)] += 1

            # Add suffix - use pre-computed suffix
            if suffix:
                features[('suffix2', suffix, tag)] += 1

            # Add position features - avoid expensive checks
            if i == 0:
                features[('position', 'first', tag)] += 1
            elif i == n - 1:
                features[('position', 'last', tag)] += 1

            # Add minimal context for short sentences only with bounds checking
            if n < 20:  # Skip for long sentences
                if i > 0:
                    features[('prev_word', processed_words[i-1], tag)] += 1

                if i < n - 1:
                    features[('next_word', processed_words[i+1], tag)] += 1

            prev_tag = tag

        # Add final transition to STOP
        features[('transition', prev_tag, 'STOP')] += 1

        # Cache and return - limit cache size to prevent memory bloat
        if len(self.feature_cache) < 10000:  # Limit cache size
            self.feature_cache[cache_key] = features
        return features

    def viterbi_decode(self, words):
        """
        Highly optimized Viterbi decoding with adaptive beam size.
        """
        n = len(words)
        if n == 0:
            return []

        # Adaptive beam size based on sentence length
        beam_size = 5 if n < 15 else (4 if n < 25 else 3)

        # Initialize DP tables - use arrays instead of nested dictionaries when possible
        viterbi = [{} for _ in range(n)]
        backpointer = [{} for _ in range(n)]

        # Cache the first word to avoid repeated lookups
        word = words[0]
        if word not in self.vocabulary:
            word = classify_token(word) if self.specialized_unks else '#UNK#'

        # Pre-compute scores for all tags at position 0
        # Using direct attribute lookups instead of .get() when possible
        common_weights = {}  # Cache frequently accessed weights

        # Process first word with vectorized operations
        for tag in self.tags:
            score = 0
            emission_key = ('emission', word, tag)
            transition_key = ('transition', 'START', tag)

            # Faster direct access with fallback
            score += self.weights[emission_key] if emission_key in self.weights else 0
            score += self.weights[transition_key] if transition_key in self.weights else 0

            # Add shape feature efficiently - avoid repeated isupper checks
            word_is_upper = word.isupper()
            word_first_upper = not word_is_upper and word and word[0].isupper()

            if word_is_upper:
                shape_key = ('shape', 'ALL_CAPS', tag)
                score += self.weights[shape_key] if shape_key in self.weights else 0
            elif word_first_upper:
                shape_key = ('shape', 'INIT_CAP', tag)
                score += self.weights[shape_key] if shape_key in self.weights else 0

            # Add position feature
            pos_key = ('position', 'first', tag)
            score += self.weights[pos_key] if pos_key in self.weights else 0

            # Add suffix feature - avoid repeated startswith checks
            if len(word) > 2 and not word.startswith('#UNK'):
                suffix_key = ('suffix2', word[-2:], tag)
                score += self.weights[suffix_key] if suffix_key in self.weights else 0

            viterbi[0][tag] = score
            backpointer[0][tag] = 'START'

        # Fast beam pruning for first word using sorted
        if len(viterbi[0]) > beam_size:
            # Use a more efficient approach for finding top k elements
            top_tags = sorted(viterbi[0].items(), key=lambda x: x[1], reverse=True)[
                :beam_size]
            viterbi[0] = {tag: score for tag, score in top_tags}
            backpointer[0] = {tag: backpointer[0][tag] for tag, _ in top_tags}

        # Process rest of words with optimized beam search
        for t in range(1, n):
            # Cache the current word
            word = words[t]
            if word not in self.vocabulary:
                word = classify_token(
                    word) if self.specialized_unks else '#UNK#'

            # Pre-compute word-specific scores once for all tags
            word_is_upper = word.isupper()
            word_first_upper = not word_is_upper and word and word[0].isupper()
            is_last_word = (t == n - 1)
            suffix = word[-2:] if len(
                word) > 2 and not word.startswith('#UNK') else None

            # Cache previous tags from beam - avoid repeated list() conversion
            prev_tags = tuple(viterbi[t-1].keys())

            for tag in self.tags:
                # Pre-compute emission score for current position
                emission_key = ('emission', word, tag)
                emission_score = self.weights[emission_key] if emission_key in self.weights else 0

                # Pre-compute shape features
                shape_score = 0
                if word_is_upper:
                    shape_key = ('shape', 'ALL_CAPS', tag)
                    shape_score = self.weights[shape_key] if shape_key in self.weights else 0
                elif word_first_upper:
                    shape_key = ('shape', 'INIT_CAP', tag)
                    shape_score = self.weights[shape_key] if shape_key in self.weights else 0

                # Position feature - only for last word
                position_score = 0
                if is_last_word:
                    pos_key = ('position', 'last', tag)
                    position_score = self.weights[pos_key] if pos_key in self.weights else 0

                # Suffix features
                suffix_score = 0
                if suffix:
                    suffix_key = ('suffix2', suffix, tag)
                    suffix_score = self.weights[suffix_key] if suffix_key in self.weights else 0

                # The total word-specific score (independent of transition)
                word_score = emission_score + shape_score + position_score + suffix_score

                # Calculate best transition efficiently
                max_score = float("-inf")
                best_prev_tag = None

                # Avoid recreating lambda for each iteration
                for prev_tag in prev_tags:
                    # Fast transition lookup with caching
                    trans_key = ('transition', prev_tag, tag)
                    trans_score = self.weights[trans_key] if trans_key in self.weights else 0

                    # Calculate total score
                    score = viterbi[t-1][prev_tag] + trans_score + word_score

                    # Faster max check
                    if score > max_score:
                        max_score = score
                        best_prev_tag = prev_tag

                if best_prev_tag is not None:
                    viterbi[t][tag] = max_score
                    backpointer[t][tag] = best_prev_tag

            # Optimized beam pruning
            if viterbi[t] and len(viterbi[t]) > beam_size:
                # More efficient top-k selection
                top_tags = sorted(viterbi[t].items(), key=lambda x: x[1], reverse=True)[
                    :beam_size]
                viterbi[t] = {tag: score for tag, score in top_tags}
                backpointer[t] = {tag: backpointer[t][tag]
                                  for tag, _ in top_tags}

        # No valid path found
        if not viterbi[n-1]:
            return [self.tags[0]] * n

        # Optimized final tag selection and transition to STOP
        max_final_score = float("-inf")
        best_final_tag = None

        for tag in viterbi[n-1]:
            stop_key = ('transition', tag, 'STOP')
            stop_score = self.weights[stop_key] if stop_key in self.weights else 0
            score = viterbi[n-1][tag] + stop_score

            if score > max_final_score:
                max_final_score = score
                best_final_tag = tag

        if best_final_tag is None:
            return [self.tags[0]] * n

        # Optimized backtracking - pre-allocate path array
        path = [None] * n
        path[n-1] = best_final_tag

        for t in range(n-2, -1, -1):
            path[t] = backpointer[t+1][path[t+1]]

        return path

    def train(self, train_sentences, num_iterations=5, learning_rate=1.0, decay_rate=0.8):
        """
        Train perceptron with early stopping, learning rate decay, and optimized weight updates.
        """
        print(
            f"Training optimized perceptron for {num_iterations} iterations...")

        # Pre-allocate larger sets for performance
        correct_predictions = set()

        # Use a more efficient structure for weights access
        weights_dict = self.weights
        if not isinstance(weights_dict, dict):
            weights_dict = dict(weights_dict)
            self.weights = weights_dict

        # Pre-compute and cache feature vectors for gold standard
        print("Preprocessing training data...")
        gold_features_cache = {}
        words_cache = {}
        gold_tags_cache = {}

        # Process all training data at once to avoid re-analyzing in each iteration
        for i, sentence in enumerate(train_sentences):
            words = tuple(word for word, _ in sentence)
            gold_tags = tuple(tag for _, tag in sentence)

            # Store in cache for quick lookup
            words_cache[i] = words
            gold_tags_cache[i] = gold_tags
            gold_features_cache[i] = self.get_features(words, gold_tags)

        print("Starting training iterations...")
        avg_mistakes_per_iter = []

        for iteration in range(num_iterations):
            # Compute learning rate once per iteration
            current_lr = learning_rate * (decay_rate ** iteration)
            print(
                f"Iteration {iteration+1}/{num_iterations} (LR: {current_lr:.4f})")

            start_time = time.time()
            num_mistakes = 0
            num_correct = len(correct_predictions)

            # Pre-shuffle indices instead of rebuilding list each time
            indices = list(range(len(train_sentences)))
            random.shuffle(indices)

            # Track batch updates for optimized updates
            batch_updates = Counter()
            batch_size = 0
            batch_threshold = 32  # Apply updates in small batches

            # Process training examples more efficiently
            for i in indices:
                # Skip already correct predictions
                if i in correct_predictions:
                    continue

                # Get cached data
                words = words_cache[i]
                gold_tags = gold_tags_cache[i]
                gold_features = gold_features_cache[i]

                # Get predicted tags - this is the most expensive operation
                pred_tags = self.viterbi_decode(words)

                # Check if prediction matches gold
                if pred_tags == gold_tags:
                    correct_predictions.add(i)
                    num_correct += 1
                    continue

                # Update weights for incorrect prediction
                num_mistakes += 1

                # Extract features for predicted sequence
                pred_features = self.get_features(words, pred_tags)

                # Accumulate updates in batch for better performance
                for feature, count in gold_features.items():
                    batch_updates[feature] += current_lr * count

                for feature, count in pred_features.items():
                    batch_updates[feature] -= current_lr * count

                batch_size += 1

                # Apply batch updates periodically to improve cache locality
                if batch_size >= batch_threshold:
                    for feature, update in batch_updates.items():
                        weights_dict[feature] = weights_dict.get(
                            feature, 0) + update
                    batch_updates.clear()
                    batch_size = 0

            # Apply any remaining updates
            if batch_updates:
                for feature, update in batch_updates.items():
                    weights_dict[feature] = weights_dict.get(
                        feature, 0) + update

            # End of iteration statistics
            elapsed = time.time() - start_time
            total_sentences = len(train_sentences)
            accuracy = num_correct / total_sentences

            # Store for early stopping decisions
            avg_mistakes_per_iter.append(num_mistakes)

            print(
                f"  Mistakes: {num_mistakes}/{total_sentences - len(correct_predictions)} (in remaining sentences)")
            print(
                f"  Correct: {num_correct}/{total_sentences} ({accuracy:.2%})")
            print(f"  Time: {elapsed:.2f} seconds")

            # Improved early stopping criteria
            if num_correct > 0.92 * total_sentences:
                print(f"Early stopping: {accuracy:.2%} accuracy achieved")
                break

            # Also stop if showing no improvement for 2 consecutive iterations
            if iteration >= 2 and avg_mistakes_per_iter[-1] >= avg_mistakes_per_iter[-2] * 0.98:
                print(f"Early stopping: No significant improvement in mistake rate")
                break

        # Clear caches to free memory
        self.feature_cache.clear()
        gold_features_cache.clear()
        words_cache.clear()
        gold_tags_cache.clear()

        # Prune low-weight features to improve prediction speed
        features_removed = self.optimize_feature_set(threshold=0.01)
        print(
            f"Feature pruning: removed {features_removed} low-weight features")

        print("Training completed.")
        print(f"Final model: {len(self.weights)} features")

    def optimize_feature_set(self, threshold=0.01):
        """
        Prune low-weight features to reduce memory footprint and improve prediction speed.
        Only keeps features with absolute weight above threshold.

        Args:
            threshold: Minimum absolute weight value to keep a feature

        Returns:
            Number of features removed
        """
        if not self.weights:
            return 0

        initial_size = len(self.weights)

        # Find max weight for normalization
        max_weight = max(abs(w) for w in self.weights.values())
        if max_weight == 0:
            return 0

        # Normalize threshold
        norm_threshold = threshold * max_weight

        # Create new weights dictionary with only significant features
        new_weights = {}
        for feature, weight in self.weights.items():
            if abs(weight) >= norm_threshold:
                new_weights[feature] = weight

        # Replace weights
        self.weights = new_weights

        # Return number of features removed
        return initial_size - len(self.weights)


def batch_predict(model, sentences, batch_size=100):
    """
    Predict tags for sentences in batches with optimized progress reporting.
    """
    predictions = []
    total_sentences = len(sentences)
    total_batches = (total_sentences + batch_size - 1) // batch_size

    # Pre-allocate results array
    predictions = [None] * total_sentences

    print(
        f"Processing {total_sentences} sentences in {total_batches} batches...")
    start_time = time.time()

    # Cache the viterbi_decode method to avoid repeated lookups
    viterbi_decode = model.viterbi_decode

    # Process in larger batches for better throughput
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_sentences)

        # Only print progress occasionally to reduce overhead
        if batch_idx % 2 == 0 or batch_idx == total_batches - 1:
            elapsed = time.time() - start_time
            progress = batch_start / total_sentences
            print(
                f"  Batch {batch_idx+1}/{total_batches} ({progress:.1%}) - Time: {elapsed:.1f}s")

        # Process current batch
        batch_sentences = sentences[batch_start:batch_end]

        # Use direct indexing instead of extend()
        for i, sentence in enumerate(batch_sentences):
            predictions[batch_start + i] = viterbi_decode(sentence)

    total_time = time.time() - start_time
    print(f"Prediction completed in {total_time:.2f} seconds")

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
    print("Optimized HMM Chunker (Single-threaded with Performance Optimizations)")

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
    print("Estimating emission parameters...")
    tag_count, word_tag_count = estimate_emission_params(
        modified_train_sentences)

    # Read dev data
    print("Reading development data...")
    dev_sentences = read_unlabeled_data('EN/dev.in')

    # Step 3: Estimate transition parameters
    print("Estimating transition parameters...")
    transition_count, transition_tag_count = estimate_transition_params(
        train_sentences)

    # Get the set of all possible tags (excluding START and STOP for prediction)
    all_tags = [tag for tag in tag_count.keys() if tag not in [
        "START", "STOP"]]

    # Apply smoothing to probabilities
    print("Applying add-k smoothing to probabilities...")
    smoothing_k = 0.01
    smoothed_trans_probs, smoothed_emit_probs = smooth_probabilities(
        transition_count, transition_tag_count, word_tag_count, k=smoothing_k)

    # Predict using optimized Viterbi
    print("Making predictions with smoothed Viterbi...")
    start_time = time.time()

    # Process in batches for progress reporting
    predictions_smooth = []

    total_batches = (len(dev_sentences) + 100 - 1) // 100
    print(
        f"Processing {len(dev_sentences)} sentences in {total_batches} batches...")

    for batch_idx in range(total_batches):
        batch_start = batch_idx * 100
        batch_end = min(batch_start + 100, len(dev_sentences))

        if batch_idx % 2 == 0:
            elapsed = time.time() - start_time
            progress = batch_start / len(dev_sentences)
            print(
                f"  Batch {batch_idx+1}/{total_batches} ({progress:.1%}) - Time: {elapsed:.1f}s")

        batch_sentences = dev_sentences[batch_start:batch_end]
        batch_predictions = [
            viterbi_with_smoothing(
                sentence, vocabulary, all_tags,
                smoothed_trans_probs, smoothed_emit_probs,
                specialized_unks=use_specialized_unks
            ) for sentence in batch_sentences
        ]
        predictions_smooth.extend(batch_predictions)

    viterbi_time = time.time() - start_time
    print(f"Viterbi prediction completed in {viterbi_time:.2f} seconds")

    # Write output
    write_output('EN/dev.optimized.out', dev_sentences, predictions_smooth)
    print("Smoothed HMM Viterbi predictions written to EN/dev.optimized.out")

    # Train optimized perceptron
    print("\nTraining optimized perceptron...")
    perceptron = OptimizedPerceptron(
        ["START"] + all_tags + ["STOP"], vocabulary)
    perceptron.train(
        train_sentences,
        num_iterations=6,
        learning_rate=1.0,
        decay_rate=0.8
    )

    # Predict with optimized perceptron
    print("Making predictions with optimized perceptron...")
    predictions_perceptron = batch_predict(perceptron, dev_sentences)

    # Write output
    write_output('EN/dev.optimized_perceptron.out',
                 dev_sentences, predictions_perceptron)
    print("Optimized perceptron predictions written to EN/dev.optimized_perceptron.out")

    # Try to read test data
    try:
        print("Checking for test data...")
        test_sentences = read_unlabeled_data('EN/test.in')

        # Predict on test data with perceptron (usually better than HMM)
        print("Making predictions on test data...")
        test_predictions = batch_predict(perceptron, test_sentences)

        # Write test output
        write_output('EN/test.optimized.out', test_sentences, test_predictions)
        print("Optimized predictions for test written to EN/test.optimized.out")
    except FileNotFoundError:
        print("Test file not found. Skipping test predictions.")

    print("\nAll predictions completed.")
