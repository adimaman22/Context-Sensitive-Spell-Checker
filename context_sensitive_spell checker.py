import re
import sys
import random
import math
from collections import Counter, defaultdict
import nltk


class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """

    def __init__(self,  lm=None):
        """Initializing a spell checker object with a language model as an
        instance  variable.

        Args:
            lm: a language model object. Defaults to None.
        """
        self.lm = lm
        self.error_tables = None
        self.max_edit_distance = 2  # allowing for easy adjustment of the maximum edit distance considered

    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM dictionary if set)

            Args:
                lm: a Spell_Checker.Language_Model object
        """
        self.lm = lm

    def add_error_tables(self, error_tables):
        """ Adds the specified dictionary of error tables as an instance variable.
            (Replaces an older value dictionary if set)


            Args:
            error_tables (dict): a dictionary of error tables in the format
            of the provided confusion matrices:
            https://www.dropbox.com/s/ic40soda29emt4a/spelling_confusion_matrices.py?dl=0
        """
        self.error_tables = error_tables

    def evaluate_text(self, text):
        """Returns the log-likelihood of the specified text given the language
            model in use. Smoothing should be applied on texts containing OOV words

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        # Raise an error in the event that LM was not defined
        if not self.lm:
            raise ValueError("Please note that Language model has not been set.")

        return self.lm.evaluate_text(text)

    def spell_check(self, text, alpha, max_iterations=3):
        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model if the number of tokens in the specified text is
            smaller than the length (n) of the language model.

            Performs spell checking on the given text using an iterative approach - correct spelling errors in the input
            text by iteratively examining and potentially correcting each word.
            It uses the language model and error probabilities to determine the most likely corrections.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.
                max_iterations (int, optional): The maximum number of passes to make over the text. Defaults to 3.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """
        original_text = text
        normalized_text = normalize_text(text)
        words = normalized_text.split()
        n = self.lm.get_model_window_size()

        if len(words) < n:
            # For short texts, use a simple noisy channel model
            return self.simple_spell_check(normalized_text, alpha)

        # Find the word with the lowest probability
        word_probs = []
        for i, word in enumerate(words):
            context = words[max(0, i-n+1):i] + words[i+1:min(len(words), i+n)]
            prob = self.lm.evaluate_text(' '.join([*context[:n - 1], word]))
            word_probs.append((word, prob, i))

        # Sort words by their probability (lowest first)
        word_probs.sort(key=lambda x: x[1])

        # Attempt to correct the word with the lowest probability
        # Set a threshold for correction (e.g., log probability less than -10)
        threshold = -10
        for original_word, prob, index in word_probs:
            if prob < threshold and original_word not in self.lm.vocab:
                context = words[max(0, index - n + 1):index] + words[index + 1:min(len(words), index + n)]
                best_word, _ = self.find_best_correction(original_word, context, alpha)
                words[index] = best_word
                break  # Stop after correcting one word

        corrected_text = ' '.join(words)

        # If the corrected text is the same as the original (normalized) text, return the original text
        if corrected_text == normalize_text(original_text):
            return original_text
        else:
            # Attempt to preserve original capitalization and punctuation
            return self.preserve_original_format(original_text, corrected_text)

    def simple_spell_check(self, text, alpha):
        """Spell check for short texts using a simple noisy channel model."""
        words = text.split()
        corrected_words = []
        for word in words:
            best_word, _ = self.find_best_correction(word, [], alpha)
            corrected_words.append(best_word)
        return ' '.join(corrected_words)

    def preserve_original_format(self, original_text, corrected_text):
        """Preserve the original capitalization and punctuation."""
        original_words = original_text.split()
        corrected_words = corrected_text.split()

        final_words = []
        for orig, corr in zip(original_words, corrected_words):
            if orig.lower() == corr:
                final_words.append(orig)  # Preserve original capitalization
            else:
                final_words.append(corr)  # Use corrected word

        # Add any remaining words (in case of insertion or deletion)
        final_words.extend(corrected_words[len(original_words):])

        # Attempt to preserve final punctuation
        if original_text and original_text[-1] in '.!?':
            return ' '.join(final_words) + original_text[-1]
        else:
            return ' '.join(final_words)

    def find_best_correction(self, word, context, alpha):
        """
        Finds the best correction for a given word considering its context.
        Generates candidate corrections for the input word and evaluates each candidate based on both the error
        probability and the language model probability given the context.

        Args:
            word (str): The word to potentially correct.
            context (list): A list of words representing the context of the target word.
            alpha (float): The probability of keeping a lexical word as is.

        Returns:
            tuple: A pair (best_word, best_probability) where best_word is the most likely
                   correction (or the original word if no better correction is found), and
                   best_probability is the probability score of this correction.
        """
        if word in self.lm.vocab:
            return word, alpha

        candidates = self.generate_candidates(word)
        if not candidates:
            return word, alpha  # Return the original word if no candidates are found

        context_str = ' '.join(context)

        def score_candidate(candidate):
            error_prob = self.calculate_error_probability(word, candidate)
            context_prob = math.exp(self.lm.evaluate_text(f"{context_str} {candidate}"))
            return error_prob * context_prob

        if candidates:
            best_candidate = max(candidates, key=score_candidate)
            return best_candidate, score_candidate(best_candidate)
        else:
            return word, alpha

    def generate_candidates(self, word):
        """
        Generates a set of candidate corrections for a given word.
        Creates variations of the input word up to a maximum edit distance (defined by self.max_edit_distance).
        It then filters these candidates to include only words that exist in the language model's vocabulary.

        Args:
            word (str): The word for which to generate correction candidates.

        Returns:
            set: A set of candidate words that exist in the language model's vocabulary.
        """
        candidates = {word}  # set([word])
        for i in range(1, self.max_edit_distance + 1):
            candidates.update(self.generate_multi_edit_variations(word, i))
        return candidates.intersection(self.lm.vocab)

    def generate_multi_edit_variations(self, word, n):
        """
        Generates all possible edits of a word up to n edit distances.

        This function recursively generates variations of the input word by applying
        edits (insertions, deletions, substitutions, and transpositions) up to n times.

        Args:
            word (str): The word to edit.
            n (int): The maximum number of edits to apply.

        Returns:
            set: A set of all possible variations of the word within n edits.
        """
        if n == 0:
            return {word}
        if n == 1:
            return self.generate_single_edit_variations(word)
        return set(e2 for e1 in self.generate_multi_edit_variations(word, n - 1)
                   for e2 in self.generate_single_edit_variations(e1))

    def generate_single_edit_variations(self, word):
        """
        Generates all possible single-edit variations of a word using a character-by-character approach.

        This function creates four types of edits:
        1. Deletions: Remove one letter
        2. Transpositions: Swap two adjacent letters
        3. Replacements: Change one letter to another
        4. Insertions: Add a letter

        Args:
            word (str): The word to edit.

        Returns:
            set: A set of all possible variations of the word with one edit.
        """
        letters = 'abcdefghijklmnopqrstuvwxyz'  # all the letters a-z
        word_length = len(word)
        variations = defaultdict(set)

        # Deletions
        for i in range(word_length):
            new_word = word[:i] + word[i + 1:]
            variations['deletions'].add(new_word)

        # Transpositions
        for i in range(word_length - 1):
            new_word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
            variations['transpositions'].add(new_word)

        # Replacements
        for i in range(word_length):
            for c in letters:
                if c != word[i]:
                    new_word = word[:i] + c + word[i + 1:]
                    variations['replacements'].add(new_word)

        # Insertions
        for i in range(word_length + 1):
            for c in letters:
                new_word = word[:i] + c + word[i:]
                variations['insertions'].add(new_word)

        # Combine all variations
        all_variations = set()
        for variation_type in variations.values():
            all_variations.update(variation_type)

        return all_variations

    def calculate_error_probability(self, original, correction):
        """
        Calculates the probability of an error transforming the original word into the correction.

        This function compares the original word and the correction, identifying the type of
        error (insertion, deletion, substitution, or transposition) and calculating the
        probability of this error occurring based on the error tables.

        Args:
            original (str): The original (potentially misspelled) word.
            correction (str): The proposed correction.

        Returns:
            float: The calculated probability of the error. Returns 1.0 if the words are identical,
                   and a very small probability (1e-10) if no specific error probability is found.
        """
        if original == correction:
            return 1.0

        m, n = len(original), len(correction)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if original[i - 1] == correction[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j],  # deletion
                                       dp[i][j - 1],  # insertion
                                       dp[i - 1][j - 1])  # substitution

        edit_distance = dp[m][n]
        return math.exp(-edit_distance)  # Convert edit distance to probability

    def get_error_prob(self, error_type, chars):
        """
       Retrieves the probability of a specific error from the error tables.

       This function looks up the probability of a specific error (characterized by the error
       type and the characters involved) in the pre-computed error tables.

       Args:
           error_type (str): The type of error ('deletion', 'insertion', 'substitution', or 'transposition').
           chars (str): The character(s) involved in the error.

       Returns:
           float: The probability of the specified error occurring. Returns a very small
                  probability (1e-10) if the error is not found in the tables.
       """
        if self.error_tables and error_type in self.error_tables:
            return self.error_tables[error_type].get(chars, 1e-10)
        return 1e-10

    #####################################################################
    #                   Inner class                                     #
    #####################################################################

    class Language_Model:
        """The class implements a Markov Language Model that learns a model from a given text.
            It supports language generation and the evaluation of a given string.
            The class can be applied on both word level and character level.
        """

        def __init__(self, n=3, chars=False):
            """Initializing a language model object.
            Args:
                n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
                chars (bool): True iff the model consists of ngrams of characters rather than word tokens.
                              Defaults to False
            """
            self.n = n
            self.chars = chars
            self.vocab = set()
            self.unigram_counts = Counter()
            self.bigram_counts = Counter()
            self.model_dict = defaultdict(int)  # {ngram:count}, holding counts of all ngrams in the specified text.

            # NOTE: This dictionary format is inefficient and insufficient and a better data structure can be used.
            # However, we were requested to support this format.

        def build_model(self, text):
            """populates the instance variable model_dict.
            Combined the logic for character-level and word-level processing.

                Args:
                    text (str): the text to construct the model from.
            """
            text = normalize_text(text)
            tokens = list(text) if self.chars else text.split()

            # Update vocabulary and count unigrams and bigrams
            self.vocab.update(tokens)
            self.unigram_counts.update(tokens)
            self.bigram_counts.update(zip(tokens, tokens[1:]))

            # Build n-gram model
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                self.model_dict[ngram] += 1

        def get_model_dictionary(self):
            """
            Returns the dictionary class object.
            Get the n-gram model dictionary.

            Returns:
                dict: The n-gram model dictionary.
            """
            return self.model_dict

        def get_model_window_size(self):
            """
            Returning the size of the context window (the n in "n-gram").
            Get the size of the n-grams used in the model.

            Returns:
                int: The size of the n-grams.
            """
            return self.n

        def generate(self, context=None, n=20):
            """Returns a string of the specified length, generated by applying the language model
            to the specified seed context. If no context is specified the context should be sampled
            from the models' contexts distribution. Generation should stop before the n'th word if the
            contexts are exhausted. If the length of the specified context exceeds (or equal to)
            the specified n, the method should return a prefix of length n of the specified context.

                Args:
                    context (str): a seed context to start the generated string from. Defaults to None
                    n (int): the length of the string to be generated.

                Return:
                    String. The generated text.
            """
            if not self.model_dict:
                raise ValueError("The language model is empty. Please build the model first.")

            if context:
                context = normalize_text(context)
                context = tuple(context if self.chars else context.split())

                # If context length exceeds or equals the specified length, return a prefix:
                if len(context) >= n:
                    return ''.join(context[:n]) if self.chars else ' '.join(context[:n])

                # Use the last n-1 tokens as the starting context
                start_context = context[-(self.n - 1):]

            else:
                # Sample a random starting context from the model's distribution
                start_context = random.choice([key[:-1] for key in self.model_dict.keys() if len(key) == self.n])

            generated = list(start_context)

            while len(generated) < n:
                current_context = tuple(generated[-(self.n - 1):])
                possible_next = [key[-1] for key in self.model_dict.keys() if key[:-1] == current_context]

                if not possible_next:
                    # Context is exhausted, stop generation
                    break

                next_token = random.choices(
                    possible_next,
                    weights=[self.model_dict[current_context + (token,)] for token in possible_next],
                    k=1
                )[0]

                generated.append(next_token)

                # For word-level models, check if we've reached n-1 words
                if not self.chars and len(generated) == n - 1:
                    break

            # Join the generated tokens
            return ''.join(generated) if self.chars else ' '.join(generated)

        def evaluate_text(self, text):
            """Returns the log-likelihood of the specified text to be a product of the model.
               Laplace smoothing applied if necessary.

               Args:
                   text (str): Text to evaluate.

               Returns:
                   Float. The float should reflect the (log) probability.
            """
            text = normalize_text(text)
            tokens = list(text) if self.chars else text.split()
            log_likelihood = 0

            if len(tokens) < self.n:
                # For texts shorter than n, we evaluate the probability of the whole text:
                return math.log(self.smooth(' '.join(tokens)))

            for i in range(len(tokens) - self.n + 1):
                prob = self.smooth(' '.join(tokens[i:i + self.n]))
                if prob > 0:
                    log_likelihood += math.log(prob)
                else:
                    log_likelihood += math.log(1e-10)  # Very small probability for zero probabilities

            return log_likelihood

        def smooth(self, ngram, alpha=1):
            """Returns the smoothed (Laplace) probability of the specified ngram.

                Args:
                    ngram (str): the ngram to have its probability smoothed
                    alpha (float): smoothing factor. Defaults to 1 for Laplace smoothing.

                Returns:
                    float. The smoothed probability.
            """
            ngram = tuple(ngram.split())  # for this function usage
            vocab_size = len(self.vocab)

            if len(ngram) < self.n:
                # For ngrams shorter than n, we return alpha/|V|
                return alpha / vocab_size

            context = ngram[:-1]
            token = ngram[-1]

            # Count of the full ngram
            token_count = self.model_dict[ngram]

            # Total count of all ngrams with the same context
            context_count = sum(self.model_dict[context + (t,)] for t in self.vocab)

            if context_count == 0:
                return alpha / vocab_size

            # Laplace smoothing
            lambda1 = max(context_count - alpha, 0) / context_count
            lambda2 = alpha / vocab_size
            laplace_smoothed_probability = lambda1 * (token_count / context_count) + lambda2

            return laplace_smoothed_probability


def normalize_text(text, lowercase=True, remove_punctuation=True, handle_numbers=False, remove_stopwords=False):
    """Returns a normalized version of the specified string.

    This function performs the following normalizations:
    1. Converts text to lowercase (optional)
    2. Removes punctuation (optional)
    3. Handles numbers: replaces dates, decimals, and other numbers with placeholders (optional)
    4. Replaces multiple spaces with a single space
    5. Strips leading and trailing whitespace
    6. Removes stopwords (optional)

      Args:
        text (str): the text to normalize.
        lowercase (bool): Whether to convert the text to lowercase. Defaults to True.
        remove_punctuation (bool): Whether to remove punctuation. Defaults to True.
        handle_numbers (bool): Whether to remove numbers. Defaults to False.
        remove_stopwords (bool): Whether to remove stopwords. Defaults to False.

      Returns:
        string: the normalized text.
    """
    # Convert to lowercase if specified
    if lowercase:
        text = text.lower()

    # Handle numbers if specified
    if handle_numbers:
        # Replace dates (format: dd/mm/yyyy or mm/dd/yyyy or yyyy/mm/dd, with variations)
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 'DATE', text)
        text = re.sub(r'\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b', 'DATE', text)
        # Replace decimal numbers (optionally followed by a percent sign)
        text = re.sub(r'\b\d+\.\d+%?\b', 'DECIMAL', text)
        # Replace whole numbers (optionally followed by a percent sign)
        text = re.sub(r'\b\d+%?\b', 'NUM', text)
        # Remove any remaining digits
        text = re.sub(r'\d+', '', text)

    # Remove punctuation if specified
    if remove_punctuation:
        # Define a set of punctuation characters
        punctuation_chars = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        # Use a list comprehension to keep only non-punctuation characters
        text = ''.join(char for char in text if char not in punctuation_chars)

    # Remove stopwords if specified
    if remove_stopwords:
        from nltk.corpus import stopwords
        nltk.download('stopwords')
        stopwords = stopwords.words('english')
        text = ' '.join(word for word in text.split() if word not in stopwords)

    # Replace multiple spaces with a single space
    text = ' '.join(text.split())
    # Strip leading and trailing whitespace
    text = text.strip()

    return text







