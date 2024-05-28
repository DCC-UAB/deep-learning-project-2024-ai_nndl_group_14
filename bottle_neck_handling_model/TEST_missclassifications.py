from collections import Counter

import numpy as np


import matplotlib.pyplot as plt

def plot_top_misclassifications(misclassifications, top_n=150, title='Top Misclassifications'):
    labels, values = zip(*misclassifications.most_common(top_n))
    indexes = range(len(labels))
    labels = [f'{chr(l[0])}->{chr(l[1])}' if isinstance(l, tuple) else ''.join([chr(char) for char in l if char != -1]) for l in labels]

    plt.figure(figsize=(10, 8))
    plt.bar(indexes, values)
    plt.xlabel('Misclassifications')
    plt.ylabel('Counts')
    plt.title(title)
    plt.xticks(indexes, labels, rotation='vertical')
    plt.show()


def display_top_misclassifications(letter_misclassifications, word_misclassifications, top_n=20):
    """
    Displays the top misclassified letters and words, handling potential data type issues.
    
    Parameters:
    - letter_misclassifications (Counter): Counter of misclassified letter pairs
    - word_misclassifications (Counter): Counter of misclassified words
    - top_n (int): Number of top misclassifications to display
    """
    print(f"Top {top_n} Misclassified Letters:")
    for (true_letter, pred_letter), count in letter_misclassifications.most_common(top_n):
        # Ensure the letters are integers before converting to characters
        if isinstance(true_letter, int) and isinstance(pred_letter, int):
            print(f"True: '{chr(true_letter)}', Predicted: '{chr(pred_letter)}', Count: {count}")
        else:
            print(f"True: '{true_letter}', Predicted: '{pred_letter}', Count: {count}")

    print(f"\nTop {top_n} Misclassified Words:")
    for (true_word,), count in word_misclassifications.most_common(top_n):
        # Handle both list of ints and strings correctly
        if all(isinstance(char, int) for char in true_word):
            readable_true_word = ''.join([chr(char) for char in true_word if char != -1])
        else:
            readable_true_word = true_word  # assuming true_word might be directly a readable string
        print(f"Word: '{readable_true_word}', Count: {count}")




def update_misclassifications(pred, target, letter_misclassifications, word_misclassifications):
    """
    Updates the misclassification counters for both letters and words.

    Parameters:
    - pred (numpy array): Decoded predictions from the model.
    - target (numpy array): Actual labels.
    - letter_misclassifications (Counter): Counter to track letter-level misclassifications.
    - word_misclassifications (Counter): Counter to track word-level misclassifications.
    """
    batch_size = target.shape[0]

    for idx in range(batch_size):
        pred_str = "".join([chr(p) for p in pred[idx] if p != -1])
        true_str = "".join([chr(t) for t in target[idx] if t != -1])

        # Update word-level misclassification if the whole word is incorrect
        if pred_str != true_str:
            word_misclassifications[true_str] += 1

        # Convert strings to lists of characters for letter-level comparison
        pred_chars = list(pred_str)
        true_chars = list(true_str)
        min_length = min(len(pred_chars), len(true_chars))

        # Update letter-level misclassifications
        for i in range(min_length):
            if pred_chars[i] != true_chars[i]:
                letter_misclassifications[(true_chars[i], pred_chars[i])] += 1

        # Consider additional characters in longer predictions or labels
        if len(pred_chars) > len(true_chars):
            for char in pred_chars[min_length:]:
                letter_misclassifications[('-', char)] += 1
        elif len(true_chars) > len(pred_chars):
            for char in true_chars[min_length:]:
                letter_misclassifications[(char, '-')] += 1

    return letter_misclassifications, word_misclassifications
