from collections import Counter, defaultdict
import math
import heapq
import time


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
        if sentence:  # Don't forget the last sentence if file doesn't end with an empty line
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


def modify_training_data(sentences, rare_words):
    """
    Replace rare words with #UNK# token.
    """
    modified_sentences = []
    for sentence in sentences:
        modified_sentence = [
            (word if word not in rare_words else '#UNK#', tag) for word, tag in sentence]
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


def predict_baseline(word, vocabulary, tag_count, word_tag_count):
    """
    Predict tag based on emission probabilities.
    Replace unseen words with #UNK#.
    """
    if word not in vocabulary:
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


def viterbi(sentence, tags, transition_count, tag_count, word_tag_count, vocabulary):
    """
    Implement the Viterbi algorithm to find the most likely tag sequence.
    Uses log probabilities to prevent numerical underflow.
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
        word = '#UNK#'

    for tag in tags:
        # Skip START tag in tags if present
        if tag == "START":
            continue

        # Probability of starting with this tag
        trans_prob = transition_count["START"].get(
            tag, 0) / tag_count["START"] if tag_count["START"] > 0 else 0
        # Emission probability
        emit_prob = word_tag_count[tag].get(
            word, 0) / tag_count[tag] if tag_count[tag] > 0 else 0

        if trans_prob > 0 and emit_prob > 0:
            viterbi_log[0][tag] = math.log(trans_prob) + math.log(emit_prob)
            backpointer[0][tag] = "START"
        else:
            viterbi_log[0][tag] = float("-inf")

    # Process rest of the words
    for t in range(1, n):
        word = sentence[t]
        if word not in vocabulary:
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

                trans_prob = transition_count[prev_tag].get(
                    tag, 0) / tag_count[prev_tag] if tag_count[prev_tag] > 0 else 0

                if trans_prob > 0:
                    score = viterbi_log[t-1][prev_tag] + math.log(trans_prob)

                    if score > max_score:
                        max_score = score
                        best_prev_tag = prev_tag

            # Emission probability
            emit_prob = word_tag_count[tag].get(
                word, 0) / tag_count[tag] if tag_count[tag] > 0 else 0

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
        trans_prob = transition_count[tag].get(
            "STOP", 0) / tag_count[tag] if tag_count[tag] > 0 else 0

        if trans_prob > 0 and final_score != float("-inf"):
            score = final_score + math.log(trans_prob)

            if score > max_final_score:
                max_final_score = score
                best_final_tag = tag

    # No valid path found, use fallback
    if best_final_tag is None:
        # Just pick the most frequent tag as fallback
        best_final_tag = max(
            tag_count.items(), key=lambda x: x[1] if x[0] != "START" else 0)[0]
        if best_final_tag == "START":  # Ensure we don't return START as a tag
            best_final_tag = max(
                tag_count.items(), key=lambda x: x[1] if x[0] != "START" else 0)[0]
        return [best_final_tag] * n

    # Backtrack to find the best path
    path = [best_final_tag]
    for t in range(n-1, 0, -1):
        prev_tag = backpointer[t][path[0]]
        path.insert(0, prev_tag)

    return path


def k_best_viterbi(sentence, tags, transition_count, tag_count, word_tag_count, vocabulary, k=4):
    """
    Find the k-best sequences using a modified Viterbi algorithm.
    Returns the kth best sequence (0-indexed, so k=3 returns 4th best).
    """
    n = len(sentence)
    if n == 0:
        return []

    # Initialize DP tables
    # dp[t][y] = list of (score, backpointer, path_idx) tuples for top k paths
    dp = [{tag: [] for tag in tags if tag != "START"} for _ in range(n)]

    # Process first word
    word = sentence[0]
    if word not in vocabulary:
        word = '#UNK#'

    for tag in tags:
        if tag == "START":
            continue

        # Probability of starting with this tag
        trans_prob = transition_count["START"].get(
            tag, 0) / tag_count["START"] if tag_count["START"] > 0 else 0
        # Emission probability
        emit_prob = word_tag_count[tag].get(
            word, 0) / tag_count[tag] if tag_count[tag] > 0 else 0

        if trans_prob > 0 and emit_prob > 0:
            log_prob = math.log(trans_prob) + math.log(emit_prob)
            # (score, prev_tag, prev_path_idx)
            dp[0][tag].append((log_prob, "START", 0))

    # Process rest of the words
    for t in range(1, n):
        word = sentence[t]
        if word not in vocabulary:
            word = '#UNK#'

        for tag in tags:
            if tag == "START":
                continue

            # Collect all possible extensions from previous states
            candidates = []
            for prev_tag in tags:
                if prev_tag == "START":
                    continue

                # Try all paths from previous state
                for prev_path_idx, (prev_score, _, _) in enumerate(dp[t-1][prev_tag]):
                    # Skip if previous path had zero probability
                    if prev_score == float("-inf"):
                        continue

                    trans_prob = transition_count[prev_tag].get(
                        tag, 0) / tag_count[prev_tag] if tag_count[prev_tag] > 0 else 0
                    if trans_prob > 0:
                        score = prev_score + math.log(trans_prob)
                        candidates.append((score, prev_tag, prev_path_idx))

            # Keep top k candidates
            emit_prob = word_tag_count[tag].get(
                word, 0) / tag_count[tag] if tag_count[tag] > 0 else 0
            if emit_prob > 0:
                log_emit = math.log(emit_prob)
                # Apply emission probability and sort by score (descending)
                candidates = [(score + log_emit, prev_tag, prev_path_idx)
                              for score, prev_tag, prev_path_idx in candidates]
                candidates.sort(reverse=True)
                dp[t][tag] = candidates[:k]  # Keep top k

    # Find the k-best final paths
    final_candidates = []
    for tag in tags:
        if tag == "START":
            continue

        for path_idx, (score, _, _) in enumerate(dp[n-1][tag]):
            if score == float("-inf"):
                continue

            trans_prob = transition_count[tag].get(
                "STOP", 0) / tag_count[tag] if tag_count[tag] > 0 else 0
            if trans_prob > 0:
                final_score = score + math.log(trans_prob)
                final_candidates.append((final_score, tag, path_idx))

    # Sort and get the kth best (or best available if < k paths)
    final_candidates.sort(reverse=True)

    # No valid path found or fewer than k paths
    if not final_candidates or len(final_candidates) <= k:
        # Return the most frequent tag as fallback
        most_common_tag = max(
            tag_count.items(), key=lambda x: x[1] if x[0] != "START" else 0)[0]
        if most_common_tag == "START":
            most_common_tag = max(tag_count.items(
            ), key=lambda x: x[1] if x[0] != "START" and x[0] != "STOP" else 0)[0]
        return [most_common_tag] * n

    # Extract the kth best path
    _, last_tag, path_idx = final_candidates[k]

    # Backtrack to reconstruct the path
    path = [last_tag]
    for t in range(n-1, 0, -1):
        _, prev_tag, path_idx = dp[t][path[0]][path_idx]
        path.insert(0, prev_tag)

    return path


class StructuredPerceptron:
    """
    Structured Perceptron for sequence labeling.
    """

    def __init__(self, tags, vocabulary):
        self.tags = tags
        self.vocabulary = vocabulary
        self.weights = defaultdict(float)  # Feature weights

    def get_features(self, words, tags):
        """
        Extract features from a tagged sentence.
        Features include:
        - ('emission', word, tag): Word-tag pairs
        - ('transition', prev_tag, tag): Tag transitions
        """
        features = Counter()
        prev_tag = "START"

        for i, (word, tag) in enumerate(zip(words, tags)):
            word = word if word in self.vocabulary else '#UNK#'

            # Emission features
            features[('emission', word, tag)] += 1

            # Transition features
            features[('transition', prev_tag, tag)] += 1

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
        Use Viterbi algorithm with current feature weights to find best tag sequence.
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
            word = '#UNK#'

        for tag in self.tags:
            # Score includes emission and transition from START
            score = self.weights[('emission', word, tag)] + \
                self.weights[('transition', 'START', tag)]
            viterbi[0][tag] = score
            backpointer[0][tag] = "START"

        # Rest of the words
        for t in range(1, n):
            word = words[t]
            if word not in self.vocabulary:
                word = '#UNK#'

            for tag in self.tags:
                max_score = float("-inf")
                best_prev_tag = None

                for prev_tag in self.tags:
                    # Score from previous best path plus transition and emission scores
                    score = viterbi[t-1].get(prev_tag, float("-inf"))
                    if score == float("-inf"):
                        continue

                    score += self.weights[('transition', prev_tag, tag)]
                    if score > max_score:
                        max_score = score
                        best_prev_tag = prev_tag

                if best_prev_tag is not None:
                    viterbi[t][tag] = max_score + \
                        self.weights[('emission', word, tag)]
                    backpointer[t][tag] = best_prev_tag
                else:
                    viterbi[t][tag] = float("-inf")

        # Find best final tag
        max_final_score = float("-inf")
        best_final_tag = None

        for tag in self.tags:
            score = viterbi[n-1].get(tag, float("-inf"))
            if score == float("-inf"):
                continue

            score += self.weights[('transition', tag, 'STOP')]
            if score > max_final_score:
                max_final_score = score
                best_final_tag = tag

        # Fallback if no path found
        if best_final_tag is None:
            return [self.tags[0]] * n

        # Backtrack
        path = [best_final_tag]
        for t in range(n-1, 0, -1):
            prev_tag = backpointer[t][path[0]]
            path.insert(0, prev_tag)

        return path

    def train(self, train_sentences, num_iterations=10, averaged=True):
        """
        Train the perceptron on labeled data.
        Implements averaging for better generalization.
        """
        print(
            f"Training structured perceptron for {num_iterations} iterations...")

        # For averaging
        if averaged:
            total_weights = defaultdict(float)
            counts = defaultdict(int)
            iteration_count = 0

        for iteration in range(num_iterations):
            print(f"Iteration {iteration+1}/{num_iterations}")
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

                    # Update weights
                    for feature, count in gold_features.items():
                        self.weights[feature] += count
                        if averaged:
                            total_weights[feature] += count * \
                                (iteration_count + i)
                            counts[feature] += 1

                    for feature, count in pred_features.items():
                        self.weights[feature] -= count
                        if averaged:
                            total_weights[feature] -= count * \
                                (iteration_count + i)
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
    # Step 1: Read the training data
    train_sentences = read_data('EN/train')

    # Get word frequencies
    word_freq = get_word_freq(train_sentences)

    # Identify rare words (frequency < 3)
    rare_words = {word for word, freq in word_freq.items() if freq < 3}
    vocabulary = {word for word, freq in word_freq.items() if freq >= 3}
    vocabulary.add('#UNK#')  # Add #UNK# to vocabulary

    # Replace rare words with #UNK#
    modified_train_sentences = modify_training_data(
        train_sentences, rare_words)

    # Step 2: Estimate emission parameters
    tag_count, word_tag_count = estimate_emission_params(
        modified_train_sentences)

    # Read dev data
    dev_sentences = read_unlabeled_data('EN/dev.in')

    # Predict using baseline (emission probabilities only)
    predictions_p1 = []
    for sentence in dev_sentences:
        sentence_preds = [predict_baseline(
            word, vocabulary, tag_count, word_tag_count) for word in sentence]
        predictions_p1.append(sentence_preds)

    # Write output for Part 1
    write_output('EN/dev.p1.out', dev_sentences, predictions_p1)

    print("Baseline predictions written to EN/dev.p1.out")

    # Step 3: Estimate transition parameters for Part 2
    transition_count, transition_tag_count = estimate_transition_params(
        train_sentences)

    # Get the set of all possible tags (excluding START and STOP)
    all_tags = [tag for tag in tag_count.keys() if tag !=
                "START" and tag != "STOP"]

    # Predict using Viterbi
    predictions_p2 = []
    for sentence in dev_sentences:
        sentence_preds = viterbi(sentence, all_tags, transition_count,
                                 transition_tag_count, word_tag_count, vocabulary)
        predictions_p2.append(sentence_preds)

    # Write output for Part 2
    write_output('EN/dev.p2.out', dev_sentences, predictions_p2)

    print("HMM Viterbi predictions written to EN/dev.p2.out")

    # Step 4: K-Best Viterbi for Part 3 (k=3 for 4th best)
    predictions_p3 = []
    for sentence in dev_sentences:
        sentence_preds = k_best_viterbi(
            sentence, all_tags, transition_count, transition_tag_count, word_tag_count, vocabulary, k=3)
        predictions_p3.append(sentence_preds)

    # Write output for Part 3
    write_output('EN/dev.p3.out', dev_sentences, predictions_p3)

    print("4th-best Viterbi predictions written to EN/dev.p3.out")

    # Step 5: Structured Perceptron for Part 4
    print("\nStarting Part 4: Structured Perceptron")

    # Initialize and train perceptron
    perceptron = StructuredPerceptron(all_tags, vocabulary)
    perceptron.train(train_sentences, num_iterations=10)

    # Predict on dev data
    predictions_p4 = []
    for sentence in dev_sentences:
        pred_tags = perceptron.viterbi_decode(sentence)
        predictions_p4.append(pred_tags)

    # Write dev output for Part 4
    write_output('EN/dev.p4.out', dev_sentences, predictions_p4)

    print("Perceptron predictions for dev written to EN/dev.p4.out")

    # Try to read test data if available
    try:
        test_sentences = read_unlabeled_data('EN/test.in')

        # Predict on test data using perceptron
        test_predictions = []
        for sentence in test_sentences:
            pred_tags = perceptron.viterbi_decode(sentence)
            test_predictions.append(pred_tags)

        # Write test output
        write_output('EN/test.p4.out', test_sentences, test_predictions)

        print("Perceptron predictions for test written to EN/test.p4.out")
    except FileNotFoundError:
        print("Test file not found. Skipping test predictions.")

    print("\nAll predictions completed.")
