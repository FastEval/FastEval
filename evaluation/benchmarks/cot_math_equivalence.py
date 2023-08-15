# https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py
# https://github.com/nlpxucan/WizardLM/blob/main/WizardMath/inference/util.py
# https://github.com/EleutherAI/lm-evaluation-harness/blob/fd1c71964f8571407eaacf183196f709c8151825/lm_eval/tasks/hendrycks_math.py

import re

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string

def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string

def is_equiv(str1, str2):
    try:
        return strip_string(str1) == strip_string(str2)
    except Exception:
        return str1 == str2

def extract_model_answer(model_answer):
    things_to_open_and_close = {
        '\\frac{': '}',
        '\\boxed{': '}',
        '\\[': '\\]',
        '$$': '$$',
        '$': '$',
        '{': '}',
        '(': ')',
    }

    position = 0
    math_parts = []
    things_to_close = []
    while position < len(model_answer):
        remaining = model_answer[position:]
        if len(things_to_close) != 0:
            thing_to_close = things_to_close[-1]
            if remaining.startswith(thing_to_close):
                things_to_close.pop()
                position += len(thing_to_close)
                math_parts[-1] += thing_to_close
                continue
        next = False
        for special_symbol_start in things_to_open_and_close.keys():
            if remaining.startswith(special_symbol_start):
                if len(things_to_close) == 0:
                    math_parts.append(special_symbol_start)
                else:
                    math_parts[-1] += special_symbol_start
                things_to_close.append(things_to_open_and_close[special_symbol_start])
                position += len(special_symbol_start)
                next = True
                break
        if next:
            continue
        if len(things_to_close) == 0:
            if remaining.startswith(' '):
                position += 1
                continue
            part_until_next_space = remaining.split(' ')[0]
            if len(part_until_next_space) != 1 and part_until_next_space.endswith('.'):
                part_until_next_space = part_until_next_space[:-1]
            try:
                math_parts.append(str(int(part_until_next_space)))
            except:
                try:
                    math_parts.append(str(float(part_until_next_space)))
                except:
                    if re.match('^[0-9]', part_until_next_space):
                        math_parts.append(part_until_next_space)
                    pass
            position += len(part_until_next_space)
            continue
        else:
            math_parts[-1] += remaining[0]
            position += 1

    if len(math_parts) == 0:
        return ''
    else:
        answer = math_parts[-1]

    if answer.endswith('.'):
        answer = answer[:-1]

    if answer.startswith('$$') and answer.endswith('$$'):
        answer = answer[2:-2]
    if answer.startswith('$') and answer.endswith('$'):
        answer = answer[1:-1]
    if answer.startswith('\\[') and answer.endswith('\\]'):
        answer = answer[2:-2]

    if answer.startswith('\\boxed{'):
        answer = answer[7:-1]

    if '=' in answer:
        answer = answer.split('=')[1]

    answer = answer.replace('∞', '\\infty')

    return answer

def is_math_correct(model_answer, correct_answer):
    correct_answer = remove_boxed(last_boxed_only_string(correct_answer))
    model_answer_extracted = extract_model_answer(model_answer)
    is_correct = is_equiv(model_answer_extracted, correct_answer)
    #if not is_correct:
    #    print('MODEL:   ' + model_answer)
    #    print('EXTRACT: ' + model_answer_extracted)
    #    print('CORRECT: ' + correct_answer)
    #    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    return is_correct
