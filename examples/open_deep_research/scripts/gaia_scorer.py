# [MODULE DOC]
# HELPER: GAIA answer-scoring utilities (official-style quasi-exact match). Exports number/string
#   normalization, list splitting, question_scorer and a lenient close-call checker.
# Inputs: depends only on re, string and warnings.
# Outputs: functions normalize_number_str, split_string, is_float, question_scorer,
#   check_prediction_contains_answer_letters_in_order, check_close_call, normalize_str.
# Exceptions: none at import.
# Side effects: none at import.
import re
import string
import warnings


def normalize_number_str(number_str: str) -> float:
    # [FUNCTION DOC]
    # HELPER: Converts a numeric string to float after removing dollar signs, percent signs and commas.
    # Inputs: number_str (str).
    # Outputs: the float value, or positive infinity if it cannot be parsed.
    # Exceptions: none raised (ValueError is caught).
    # Side effects: prints a message when parsing fails.
    # we replace these common units and commas to allow
    # conversion to float
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        print(f"String {number_str} cannot be normalized to number str.")
        return float("inf")


def split_string(
    s: str,
    char_list: list[str] = [",", ";"],
) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Splits a string on any of the given separator characters using a regex character class.
    # Inputs: s (str), char_list (list of single characters, default comma and semicolon; mutable default, not
    #   mutated).
    # Outputs: list of substrings (not stripped).
    # Exceptions: re.error if separators form an invalid character class.
    # Side effects: none.
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def is_float(element: any) -> bool:
    # [FUNCTION DOC]
    # HELPER: Reports whether a value can be converted with float.
    # Inputs: element (any).
    # Outputs: True if conversion succeeds, False on ValueError.
    # Exceptions: TypeError propagates for values float rejects with TypeError (e.g. None).
    # Side effects: none.
    try:
        float(element)
        return True
    except ValueError:
        return False


def question_scorer(
    model_answer: str,
    ground_truth: str,
) -> bool:
    # [FUNCTION DOC]
    # Scores a model answer against a GAIA ground truth. If the truth parses as a number, compares the
    #   normalized number exactly; if it contains a comma or semicolon, splits both into lists and requires
    #   equal length and element-wise match (numbers numerically, strings normalized without punctuation
    #   removal); otherwise compares fully normalized strings.
    # Inputs: model_answer (str), ground_truth (str).
    # Outputs: bool, True if correct.
    # Exceptions: none raised directly; AttributeError if model_answer is not a str in the list/string
    #   branches.
    # Side effects: emits a UserWarning when list lengths differ; normalize_number_str may print.
    # if gt is a number
    if is_float(ground_truth):
        normalized_answer = normalize_number_str(str(model_answer))
        return normalized_answer == float(ground_truth)

    # if gt is a list
    elif any(char in ground_truth for char in [",", ";"]):
        # question with the fish: normalization removes punct

        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        # check length is the same
        if len(gt_elems) != len(ma_elems):
            warnings.warn("Answer lists have different lengths, returning False.", UserWarning)
            return False

        # compare each element as float or str
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                # we do not remove punct since comparisons can include punct
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False) == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    # if gt is a str
    else:
        return normalize_str(model_answer) == normalize_str(ground_truth)


def check_prediction_contains_answer_letters_in_order(prediction, true_answer):
    # [FUNCTION DOC]
    # HELPER: Case-insensitive fuzzy check that each letter of the true answer appears in the prediction in
    #   order.
    # Inputs: prediction (str), true_answer (str).
    # Outputs: False if the prediction is more than three times longer than the answer or a letter is missing;
    #   else True.
    # Exceptions: none raised directly.
    # Side effects: none.
    # KNOWN ISSUE: the search index is advanced to the match position rather than past it, so repeated letters
    #   can match the same character (e.g. "aa" matches "a").
    prediction = prediction.lower()
    true_answer = true_answer.lower()
    if len(prediction) > len(true_answer) * 3:
        return False
    i = 0
    for letter in true_answer:
        if letter in prediction[i:]:
            i += prediction[i:].index(letter)
        else:
            return False
    return True


def check_close_call(prediction, true_answer, is_correct):
    # [FUNCTION DOC]
    # Lenient correctness: returns True if already correct, or for non-numeric answers when letters appear in
    #   order and the prediction length is between half and double the answer length.
    # Inputs: prediction (any, stringified), true_answer (any), is_correct (bool).
    # Outputs: bool.
    # Exceptions: none raised directly.
    # Side effects: prints "Close call" when the lenient rule accepts.
    if is_correct:
        return True
    else:
        if is_float(true_answer):
            return is_correct
        else:
            if (
                check_prediction_contains_answer_letters_in_order(str(prediction), str(true_answer))
                and len(str(true_answer)) * 0.5 <= len(str(prediction)) <= len(str(true_answer)) * 2
            ):
                print(f"Close call: {prediction} vs {true_answer}")
                return True
            else:
                return False


def normalize_str(input_str, remove_punct=True) -> str:
    # [FUNCTION DOC]
    # HELPER: Normalizes a string by removing all whitespace, lowercasing and optionally stripping ASCII
    #   punctuation.
    # Inputs: input_str (str), remove_punct (bool, default True).
    # Outputs: the normalized string.
    # Exceptions: TypeError if input_str is not a str.
    # Side effects: none.
    """
    Normalize a string by:
    - Removing all white spaces
    - Optionally removing punctuation (if remove_punct is True)
    - Converting to lowercase
    Parameters:
    - input_str: str, the string to normalize
    - remove_punct: bool, whether to remove punctuation (default: True)
    Returns:
    - str, the normalized string
    """
    # Remove all white spaces. Required e.g for seagull vs. sea gull
    no_spaces = re.sub(r"\s", "", input_str)

    # Remove punctuation, if specified.
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()
