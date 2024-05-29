from collections import Counter

import numpy as np


import matplotlib.pyplot as plt

alphabet = u" ABCDEFGHIJKLMNOPQRSTUVWXYZ-'"

# Function to analyze misclassifications
def analyze_misclassifications(predictions, true_labels):
    letter_misclassifications = Counter()

    for pred_seq, true_seq in zip(predictions, true_labels):
        pred_seq = [p for p in pred_seq if p != -1]
        true_seq = [t for t in true_seq if t != -1]

        for pred_char, true_char in zip(pred_seq, true_seq):
            if pred_char != true_char:
                letter_misclassifications[(true_char, pred_char)] += 1

    return letter_misclassifications

# Function to display common misclassifications
def display_common_misclassifications(misclassifications, top_n=10):
    print("Top Misclassifications:")
    for (true_char, pred_char), count in misclassifications.most_common(top_n):
        true_display = alphabet[true_char] if true_char < len(alphabet) else f"Unknown: {true_char}"
        pred_display = alphabet[pred_char] if pred_char < len(alphabet) else f"Unknown: {pred_char}"
        print(f"True: '{true_display}', Predicted: '{pred_display}', Count: {count}")



def display_top_letter_errors(letter_misclassifications, top_n=8):
    """
    Displays the top mispredicted letters for each letter.

    Parameters:
    - letter_misclassifications (Counter): Counter of misclassified letter pairs
    - top_n (int): Number of top mispredictions to display for each letter
    """
    # Create a dictionary to hold data organized by true letter
    organized_misclassifications = {}

    # Organize misclassifications by true letter
    for (true_letter, pred_letter), count in letter_misclassifications.items():
        if true_letter not in organized_misclassifications:
            organized_misclassifications[true_letter] = []
        organized_misclassifications[true_letter].append((pred_letter, count))

    # Sort and display the results
    for true_letter in sorted(organized_misclassifications.keys()):
        mispredictions = organized_misclassifications[true_letter]
        mispredictions.sort(key=lambda x: x[1], reverse=True)  # Sort by count, descending
        true_display = alphabet[true_letter] if true_letter < len(alphabet) else f"Unknown: {true_letter}"
        print(f"Top mispredictions for '{true_display}':")
        for pred_letter, count in mispredictions[:top_n]:
            pred_display = alphabet[pred_letter] if pred_letter < len(alphabet) else f"Unknown: {pred_letter}"
            print(f"  Predicted: '{pred_display}', Count: {count}")
        print()  # Adds a newline for better readability between letters
