from collections import Counter, defaultdict
import math
import time
import re


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


def smooth_probabilities(transition_count, tag_count, word_tag_count, k=0.1):
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


def predict_baseline(word, vocabulary, tag_count, word_tag_count, specialized_unks=True):
    """
    Predict tag based on emission probabilities.
    Replace unseen words with appropriate #UNK# tokens.
    """
    if word not in vocabulary:
        if specialized_unks:
            word = classify_token(word)
        else:
            word = '#UNK#'

    max_prob = -1
    best_tag = None

    for tag in tag_count:
        count_y_x = word_tag_count[tag].get(word, 0)
        count_y = tag_count[tag]
        prob = count_y_x / count_y if count_y > 0 else 0

        if prob > max_prob:
            max_prob = prob
            best_tag = tag

    # Fallback if no tag found
    if best_tag is None and tag_count:
        best_tag = max(tag_count.keys(), key=lambda tag: tag_count[tag])

    return best_tag


def viterbi_with_smoothing(sentence, tags, smoothed_trans_probs, smoothed_emit_probs, vocabulary, specialized_unks=True):
    """
    Viterbi algorithm with smoothed probabilities.
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
        else:
            viterbi_log[0][tag] = float("-inf")

    # Process rest of the words
    for t in range(1, n):
        word = sentence[t]
        if word not in vocabulary:
            if specialized_unks:
                word = classify_token(word)
            else:
                word = '#UNK#'

        for tag in tags:
            # Skip START tag
            if tag == "START":
                continue

            max_score = float("-inf")
            best_prev_tag = None

            for prev_tag in tags:
                # Skip START for anything but the first position
                if prev_tag == "START":
                    continue

                # Skip if previous state had zero probability
                if viterbi_log[t-1].get(prev_tag, float("-inf")) == float("-inf"):
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
            else:
                viterbi_log[t][tag] = float("-inf")

    # Find the most likely end state
    max_final_score = float("-inf")
    best_final_tag = None

    for tag in tags:
        if tag == "START":
            continue

        final_score = viterbi_log[n-1].get(tag, float("-inf"))
        trans_prob = smoothed_trans_probs[tag]["STOP"]

        if trans_prob > 0 and final_score != float("-inf"):
            score = final_score + math.log(trans_prob)

            if score > max_final_score:
                max_final_score = score
                best_final_tag = tag

    # No valid path found, use fallback
    if best_final_tag is None:
        # Just pick the most frequent tag as fallback
        best_final_tag = max(tags, key=lambda t: sum(
            smoothed_emit_probs[t].values()) if t != "START" else 0)
        if best_final_tag == "START":
            best_final_tag = tags[1] if len(tags) > 1 else "O"
        return [best_final_tag] * n

    # Backtrack to find the best path
    path = [best_final_tag]
    for t in range(n-1, 0, -1):
        prev_tag = backpointer[t][path[0]]
        path.insert(0, prev_tag)

    return path


class EnhancedStructuredPerceptron:
    """
    Enhanced Structured Perceptron for sequence labeling with rich features.
    """

    def __init__(self, tags, vocabulary):
        self.tags = tags
        self.vocabulary = vocabulary
        self.weights = defaultdict(float)  # Feature weights
        self.specialized_unks = True
        self.learning_rate_schedule = None  # Will be set during training

    def get_features(self, words, tags):
        """
        Extract rich features from a tagged sentence.
        """
        features = Counter()
        prev_tag = "START"

        for i, (word, tag) in enumerate(zip(words, tags)):
            # Process the word (handle unknown words)
            if word not in self.vocabulary:
                if self.specialized_unks:
                    word = classify_token(word)
                else:
                    word = '#UNK#'

            # Basic features
            features[('emission', word, tag)] += 1
            features[('transition', prev_tag, tag)] += 1

            # Word shape features
            shape = get_word_shape(word)
            features[('shape', shape, tag)] += 1

            # Prefix/suffix features (length 2-3)
            if len(word) > 2 and not word.startswith('#UNK'):
                features[('prefix2', word[:2], tag)] += 1
                features[('suffix2', word[-2:], tag)] += 1
            if len(word) > 3 and not word.startswith('#UNK'):
                features[('prefix3', word[:3], tag)] += 1
                features[('suffix3', word[-3:], tag)] += 1

            # Position features
            if i == 0:
                features[('position', 'first', tag)] += 1
            elif i == len(words) - 1:
                features[('position', 'last', tag)] += 1

            # Context window features
            if i > 0:
                prev_word = words[i-1] if words[i-1] in self.vocabulary else classify_token(
                    words[i-1]) if self.specialized_unks else '#UNK#'
                features[('prev_word', prev_word, tag)] += 1
                features[('bigram', prev_word + "_" + word, tag)] += 1

            if i < len(words) - 1:
                next_word = words[i+1] if words[i+1] in self.vocabulary else classify_token(
                    words[i+1]) if self.specialized_unks else '#UNK#'
                features[('next_word', next_word, tag)] += 1

            # Tag context features (for training data only when we know previous tags)
            if i > 0 and len(tags) > i-1:
                features[('tag_bigram', tags[i-1], tag)] += 1

            prev_tag = tag

        # Final transition to STOP
        features[('transition', prev_tag, 'STOP')] += 1

        return features

    def score(self, features):
        """
        Compute score for a feature set based on current weights.
        """
        return sum(self.weights[f] * count for f, count in features.items())

    def viterbi_decode(self, words):
        """
        Use Viterbi algorithm with feature weights to find best tag sequence.
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
            if self.specialized_unks:
                word = classify_token(word)
            else:
                word = '#UNK#'

        # Get features for each possible tag at the first position
        for tag in self.tags:
            # Skip START tag
            if tag == "START":
                continue

            # Get features for this position with this tag
            features = self._get_position_features(words, 0, tag, "START")
            score = self.score(features)

            viterbi[0][tag] = score
            backpointer[0][tag] = "START"

        # Rest of the words
        for t in range(1, n):
            word = words[t]
            if word not in self.vocabulary:
                if self.specialized_unks:
                    word = classify_token(word)
                else:
                    word = '#UNK#'

            for tag in self.tags:
                # Skip START tag
                if tag == "START":
                    continue

                max_score = float("-inf")
                best_prev_tag = None

                for prev_tag in self.tags:
                    # Skip START for anything but the first position
                    if prev_tag == "START":
                        continue

                    # Skip if previous state had zero probability
                    if viterbi[t-1].get(prev_tag, float("-inf")) == float("-inf"):
                        continue

                    # Get features for this position with this tag and previous tag
                    features = self._get_position_features(
                        words, t, tag, prev_tag)
                    score = viterbi[t-1][prev_tag] + self.score(features)

                    if score > max_score:
                        max_score = score
                        best_prev_tag = prev_tag

                if best_prev_tag is not None:
                    viterbi[t][tag] = max_score
                    backpointer[t][tag] = best_prev_tag
                else:
                    viterbi[t][tag] = float("-inf")

        # Find best final tag
        max_final_score = float("-inf")
        best_final_tag = None

        for tag in self.tags:
            if tag == "START":
                continue

            score = viterbi[n-1].get(tag, float("-inf"))
            if score == float("-inf"):
                continue

            # Add transition to STOP
            final_score = score + self.weights[('transition', tag, 'STOP')]
            if final_score > max_final_score:
                max_final_score = score
                best_final_tag = tag

        # Fallback if no path found
        if best_final_tag is None:
            return [self.tags[0] if self.tags[0] != "START" else (self.tags[1] if len(self.tags) > 1 else "O")] * n

        # Backtrack
        path = [best_final_tag]
        for t in range(n-1, 0, -1):
            prev_tag = backpointer[t][path[0]]
            path.insert(0, prev_tag)

        return path

    def _get_position_features(self, words, pos, tag, prev_tag):
        """
        Get features for a specific position, tag, and previous tag.
        """
        features = Counter()
        word = words[pos]
        if word not in self.vocabulary:
            if self.specialized_unks:
                word = classify_token(word)
            else:
                word = '#UNK#'

        # Emission feature
        features[('emission', word, tag)] += 1

        # Transition feature
        features[('transition', prev_tag, tag)] += 1

        # Word shape
        shape = get_word_shape(word)
        features[('shape', shape, tag)] += 1

        # Prefix/suffix
        if len(word) > 2 and not word.startswith('#UNK'):
            features[('prefix2', word[:2], tag)] += 1
            features[('suffix2', word[-2:], tag)] += 1
        if len(word) > 3 and not word.startswith('#UNK'):
            features[('prefix3', word[:3], tag)] += 1
            features[('suffix3', word[-3:], tag)] += 1

        # Position in sentence
        if pos == 0:
            features[('position', 'first', tag)] += 1
        elif pos == len(words) - 1:
            features[('position', 'last', tag)] += 1

        # Context words
        if pos > 0:
            prev_word = words[pos-1] if words[pos-1] in self.vocabulary else classify_token(
                words[pos-1]) if self.specialized_unks else '#UNK#'
            features[('prev_word', prev_word, tag)] += 1
            features[('bigram', prev_word + "_" + word, tag)] += 1

        if pos < len(words) - 1:
            next_word = words[pos+1] if words[pos+1] in self.vocabulary else classify_token(
                words[pos+1]) if self.specialized_unks else '#UNK#'
            features[('next_word', next_word, tag)] += 1

        return features

    def train(self, train_sentences, num_iterations=10, decay_rate=0.8, initial_lr=1.0, averaged=True):
        """
        Train the perceptron on labeled data with learning rate decay.
        """
        print(
            f"Training enhanced structured perceptron for {num_iterations} iterations...")

        # Set learning rate schedule
        self.learning_rate_schedule = [
            initial_lr * (decay_rate ** i) for i in range(num_iterations)]

        # For averaging
        if averaged:
            total_weights = defaultdict(float)
            counts = defaultdict(int)
            iteration_count = 0

        for iteration in range(num_iterations):
            print(
                f"Iteration {iteration+1}/{num_iterations} (learning rate: {self.learning_rate_schedule[iteration]:.4f})")
            num_mistakes = 0

            start_time = time.time()
            for i, sentence in enumerate(train_sentences):
                words = [word for word, _ in sentence]
                gold_tags = [tag for _, tag in sentence]

                # Get predicted tags based on current weights
                pred_tags = self.viterbi_decode(words)

                # Update weights if prediction is incorrect
                if pred_tags != gold_tags:
                    num_mistakes += 1

                    # Extract features for gold and predicted sequences
                    gold_features = self.get_features(words, gold_tags)
                    pred_features = self.get_features(words, pred_tags)

                    # Current learning rate
                    current_lr = self.learning_rate_schedule[iteration]

                    # Update weights
                    for feature, count in gold_features.items():
                        self.weights[feature] += current_lr * count
                        if averaged:
                            total_weights[feature] += current_lr * \
                                count * (iteration_count + i)
                            counts[feature] += 1

                    for feature, count in pred_features.items():
                        self.weights[feature] -= current_lr * count
                        if averaged:
                            total_weights[feature] -= current_lr * \
                                count * (iteration_count + i)
                            counts[feature] += 1

            iteration_count += len(train_sentences)
            elapsed = time.time() - start_time
            print(
                f"  Mistakes: {num_mistakes}/{len(train_sentences)} ({num_mistakes/len(train_sentences):.2%})")
            print(f"  Time: {elapsed:.2f} seconds")

        # Average weights
        if averaged:
            print("Averaging weights...")
            total_train_instances = iteration_count
            for feature in self.weights:
                # Adjust for final iteration
                if counts[feature] > 0:
                    self.weights[feature] -= total_weights[feature] / \
                        total_train_instances

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
    print("Enhanced HMM Chunker with Rich Features and Smoothing")

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
    smoothing_k = 0.01  # Smaller k for less smoothing, preserving sharpness
    smoothed_trans_probs, smoothed_emit_probs = smooth_probabilities(
        transition_count, transition_tag_count, word_tag_count, k=smoothing_k)

    # Predict using Viterbi with smoothing
    print("Making predictions with smoothed HMM Viterbi...")
    predictions_smooth = []
    for sentence in dev_sentences:
        sentence_preds = viterbi_with_smoothing(
            sentence, all_tags, smoothed_trans_probs, smoothed_emit_probs,
            vocabulary, specialized_unks=use_specialized_unks)
        predictions_smooth.append(sentence_preds)

    # Write output for smoothed Viterbi
    write_output('EN/dev.smoothed.out', dev_sentences, predictions_smooth)

    print("Smoothed HMM Viterbi predictions written to EN/dev.smoothed.out")

    # Train enhanced perceptron
    print("\nStarting Enhanced Structured Perceptron with Rich Features")

    # Initialize and train perceptron with learning rate decay
    perceptron = EnhancedStructuredPerceptron(all_tags, vocabulary)
    perceptron.train(train_sentences, num_iterations=15,
                     decay_rate=0.9, initial_lr=1.0)

    # Predict on dev data
    print("Making predictions with enhanced perceptron...")
    predictions_enhanced = []
    for sentence in dev_sentences:
        pred_tags = perceptron.viterbi_decode(sentence)
        predictions_enhanced.append(pred_tags)

    # Write dev output for enhanced perceptron
    write_output('EN/dev.enhanced.out', dev_sentences, predictions_enhanced)

    print("Enhanced perceptron predictions written to EN/dev.enhanced.out")

    # Try to read test data if available
    try:
        print("Checking for test data...")
        test_sentences = read_unlabeled_data('EN/test.in')

        # Predict on test data using enhanced perceptron
        print("Making predictions on test data...")
        test_predictions = []
        for sentence in test_sentences:
            pred_tags = perceptron.viterbi_decode(sentence)
            test_predictions.append(pred_tags)

        # Write test output
        write_output('EN/test.enhanced.out', test_sentences, test_predictions)

        print("Enhanced perceptron predictions for test written to EN/test.enhanced.out")
    except FileNotFoundError:
        print("Test file not found. Skipping test predictions.")

    print("\nAll enhanced predictions completed.")
