# Shamelessly stolen from Microsoft Autogen team: thanks to them for this great resource!
# https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py
# [MODULE DOC]
# HELPER: Final-answer reformulation for GAIA (adapted from Microsoft Autogen). Exports prepare_response.
# Inputs: depends on copy and smolagents.models (MessageRole, Model).
# Outputs: function prepare_response.
# Exceptions: none at import.
# Side effects: none at import.
import copy

from smolagents.models import MessageRole, Model


def prepare_response(original_task: str, inner_messages, reformulation_model: Model) -> str:
    # [FUNCTION DOC]
    # Asks a model to restate a team transcript as a strictly formatted GAIA final answer. Builds a system
    #   message with the original task, appends deep copies of every non-empty inner message re-labelled as
    #   user role, then a user message with GAIA formatting rules (numbers without commas/units, short
    #   strings, comma lists) and the template "FINAL ANSWER: ...".
    # Inputs: original_task (str), inner_messages (iterable of ChatMessage-like objects with content and
    #   role), reformulation_model (smolagents Model, called with the message list).
    # Outputs: the text after the last "FINAL ANSWER: " in the reply, stripped (whole reply if the marker is
    #   absent).
    # Exceptions: model errors propagate; errors while copying messages fall back to one assistant message
    #   holding the stringified list.
    # Side effects: one model call; prints the reformulated answer.
    # KNOWN ISSUE: the fallback message uses a plain string content while other messages use content-part
    #   lists, and appended copies may be ChatMessage objects mixed with dicts.
    messages = [
        {
            "role": MessageRole.SYSTEM,
            "content": [
                {
                    "type": "text",
                    "text": f"""Earlier you were asked the following:

{original_task}

Your team then worked diligently to address that request. Read below a transcript of that conversation:""",
                }
            ],
        }
    ]

    # The first message just repeats the question, so remove it
    # if len(inner_messages) > 1:
    #    del inner_messages[0]

    # copy them to this context
    try:
        for message in inner_messages:
            if not message.content:
                continue
            message = copy.deepcopy(message)
            message.role = MessageRole.USER
            messages.append(message)
    except Exception:
        messages += [{"role": MessageRole.ASSISTANT, "content": str(inner_messages)}]

    # ask for the final answer
    messages.append(
        {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text",
                    "text": f"""
Read the above conversation and output a FINAL ANSWER to the question. The question is repeated here for convenience:

{original_task}

To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
If you are unable to determine the final answer, output 'FINAL ANSWER: Unable to determine'
""",
                }
            ],
        }
    )

    response = reformulation_model(messages).content

    final_answer = response.split("FINAL ANSWER: ")[-1].strip()
    print("> Reformulated answer: ", final_answer)

    #     if "unable to determine" in final_answer.lower():
    #         messages.append({"role": MessageRole.ASSISTANT, "content": response })
    #         messages.append({"role": MessageRole.USER, "content": [{"type": "text", "text": """
    # I understand that a definitive answer could not be determined. Please make a well-informed EDUCATED GUESS based on the conversation.

    # To output the educated guess, use the following template: EDUCATED GUESS: [YOUR EDUCATED GUESS]
    # Your EDUCATED GUESS should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. DO NOT OUTPUT 'I don't know', 'Unable to determine', etc.
    # ADDITIONALLY, your EDUCATED GUESS MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
    # If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and don't include units such as $ or percent signs unless specified otherwise.
    # If you are asked for a string, don't use articles or abbreviations (e.g. cit for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
    # If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
    # """.strip()}]})

    #         response = model(messages).content
    #         print("\n>>>Making an educated guess.\n", response)
    #         final_answer = response.split("EDUCATED GUESS: ")[-1].strip()
    return final_answer
