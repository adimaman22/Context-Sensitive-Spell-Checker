# Context-Sensitive Spell Checker

This NLP course project implements a context-sensitive spell checker using the Noisy Channel framework, leveraging a language model and error distribution models for robust text correction.

## Project Overview

The spell checker is designed to:
- Evaluate the likelihood of a given text using a language model.
- Correct misspellings based on contextual probabilities and error tables.
- Handle both simple and complex input scenarios, including short and long texts.

### Key Features:
- **Noisy Channel Model:** Corrects text based on both language model probabilities and error likelihood.
- **Error Modeling:** Uses confusion matrices to estimate error probabilities.
- **Context-Sensitive Corrections:** Utilizes n-gram language models to incorporate context during corrections.

## Implementation Details

### Main Components
1. **Spell_Checker Class:**
   - Core class implementing the spell-checking logic.
   - Includes methods for text evaluation, error correction, and candidate generation.

2. **Language_Model Inner Class:**
   - Implements a Markov language model (character or word-based).
   - Supports building n-gram models, text evaluation, and language generation.

3. **Error Tables:**
   - Stores error probabilities for various operations (insertions, deletions, substitutions, and transpositions).

### Algorithm Workflow
1. Normalize input text.
2. Tokenize the text into words.
3. Evaluate word probabilities using the language model.
4. Identify and correct the least probable words iteratively.
5. Use confusion matrices and context probabilities for candidate scoring.
