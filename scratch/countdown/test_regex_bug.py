#!/usr/bin/env python3
"""
Test to demonstrate the CRITICAL REGEX BUG in countdown.py reward function.

The bug: Using r"-?\d+" incorrectly treats minus operators as part of numbers
when there are NO SPACES around the minus sign, causing valid subtraction
expressions to be rejected.

KEY FINDING: The bug ONLY occurs when expressions have NO SPACES like "44-19+35"
             With spaces like "44 - 19 + 35", the bug doesn't trigger.
"""

import sys
import re
from collections import Counter

sys.path.insert(0, '/datadrive/difan/self-reflection/verl')
from verl.utils.reward_score.countdown import compute_countdown_score

print("=" * 80)
print("DEMONSTRATING THE REGEX BUG IN countdown.py")
print("=" * 80)

# Test cases WITHOUT SPACES (how models often generate)
test_cases = [
    {
        "name": "Simple subtraction (NO SPACES)",
        "expr": "44-19+35",
        "answer_tag": "<answer>44-19+35</answer>",
        "target": 60,
        "nums": [44, 19, 35],
        "expected": "CORRECT (evaluates to 60)",
        "has_bug": True
    },
    {
        "name": "Subtraction with multiplication (NO SPACES)",
        "expr": "44*19-35",
        "answer_tag": "<answer>44*19-35</answer>",
        "target": 801,
        "nums": [44, 19, 35],
        "expected": "CORRECT (evaluates to 801)",
        "has_bug": True
    },
    {
        "name": "Parentheses with subtraction (NO SPACES)",
        "expr": "(44-19)*35",
        "answer_tag": "<answer>(44-19)*35</answer>",
        "target": 875,
        "nums": [44, 19, 35],
        "expected": "CORRECT (evaluates to 875)",
        "has_bug": True
    },
    {
        "name": "Negative intermediate (NO SPACES)",
        "expr": "19-44+35",
        "answer_tag": "<answer>19-44+35</answer>",
        "target": 10,
        "nums": [44, 19, 35],
        "expected": "CORRECT (evaluates to 10)",
        "has_bug": True
    },
    {
        "name": "Mixed operators (NO SPACES)",
        "expr": "44+19*35",
        "answer_tag": "<answer>44+19*35</answer>",
        "target": 709,
        "nums": [44, 19, 35],
        "expected": "CORRECT (evaluates to 709)",
        "has_bug": False  # No subtraction
    },
    # Comparison: WITH SPACES (bug doesn't trigger)
    {
        "name": "Simple subtraction (WITH SPACES)",
        "expr": "44 - 19 + 35",
        "answer_tag": "<answer>44 - 19 + 35</answer>",
        "target": 60,
        "nums": [44, 19, 35],
        "expected": "CORRECT (evaluates to 60)",
        "has_bug": False  # Spaces prevent the bug
    },
]

print("\n" + "=" * 80)
print("PART 1: SHOWING HOW THE BUGGY REGEX PARSES EXPRESSIONS")
print("=" * 80)

for i, test in enumerate(test_cases, 1):
    expr = test["expr"]
    allowed = test["nums"]

    # Current BUGGY regex (what's in the code)
    buggy_pattern = r"-?\d+"
    buggy_digits = list(map(int, re.findall(buggy_pattern, expr)))

    # Correct regex (what it should be)
    correct_pattern = r"\d+"
    correct_digits = list(map(int, re.findall(correct_pattern, expr)))

    # Check if they match
    buggy_match = Counter(buggy_digits) == Counter(allowed)
    correct_match = Counter(correct_digits) == Counter(allowed)

    print(f"\n{i}. {test['name']}")
    print(f"   Expression: '{expr}'")
    print(f"   Allowed numbers: {allowed}")
    print(f"   Expected behavior: {test['expected']}")
    print()
    print(f"   BUGGY regex r'{buggy_pattern}':")
    print(f"     Extracted: {buggy_digits}")
    print(f"     Counter:   {dict(Counter(buggy_digits))}")
    print(f"     Matches allowed? {buggy_match} {'✗ WRONG!' if not buggy_match and correct_match else ''}")
    print()
    print(f"   CORRECT regex r'{correct_pattern}':")
    print(f"     Extracted: {correct_digits}")
    print(f"     Counter:   {dict(Counter(correct_digits))}")
    print(f"     Matches allowed? {correct_match} {'✓ CORRECT' if correct_match else ''}")

    if not buggy_match and correct_match:
        print()
        print(f"   >>> ⚠️  BUG DETECTED!")
        print(f"   >>> The buggy regex treats '-' as part of the number when there's no space")
        print(f"   >>> In '{expr}', it extracts {buggy_digits} instead of {correct_digits}")
        print(f"   >>> Negative numbers like {[x for x in buggy_digits if x < 0]} are NOT in allowed list!")
        print(f"   >>> This causes numbers_ok=False → score=0.0 for a VALID answer!")

print("\n" + "=" * 80)
print("PART 2: RUNNING ACTUAL REWARD FUNCTION TO SHOW THE BUG")
print("=" * 80)

bugs_found = 0
for i, test in enumerate(test_cases, 1):
    result = compute_countdown_score(
        data_source='test',
        solution_str=test["answer_tag"],
        ground_truth={'target': test['target'], 'numbers': test['nums']},
        extra_info={}
    )

    # Calculate what the expression actually evaluates to
    actual_value = eval(test["expr"])  # Safe here since we control the input

    should_pass = actual_value == test["target"]
    did_pass = result['score'] == 1.0

    status = "✓ PASS" if did_pass else "✗ FAIL"
    if should_pass and not did_pass:
        status += " ⚠️  (BUG!)"
        bugs_found += 1

    print(f"\n{i}. {test['name']:45s} {status}")
    print(f"   Expression evaluates to: {actual_value} (target: {test['target']})")
    print(f"   Should get score=1.0? {should_pass}")
    print(f"   Actual result:")
    print(f"     score:        {result['score']}")
    print(f"     exact_match:  {result['exact_match']}")
    print(f"     numbers_ok:   {result['numbers_ok']}")
    print(f"     value:        {result['value']}")

    if should_pass and not did_pass:
        print()
        print(f"   ⚠️  BUG IMPACT:")
        print(f"       Expression is mathematically CORRECT (evaluates to {test['target']})")
        print(f"       But numbers_ok={result['numbers_ok']} due to regex bug on line 76")
        print(f"       Regex extracted negative numbers from '{test['expr']}'")
        print(f"       Scoring gives 0.0 instead of 1.0!")
        print(f"       If models generate compact expressions WITHOUT SPACES,")
        print(f"       they will be PENALIZED for correct answers!")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total test cases: {len(test_cases)}")
print(f"Cases that should pass: {sum(1 for t in test_cases if eval(t['expr']) == t['target'])}")
print(f"Cases that actually passed: {sum(1 for t in test_cases if eval(t['expr']) == t['target']) - bugs_found}")
print(f"Bugs triggered: {bugs_found}")

if bugs_found > 0:
    print()
    print("✗✗✗ CRITICAL BUG CONFIRMED ✗✗✗")
    print()
    print("BUG CONDITION:")
    print("  The bug triggers when expressions have NO SPACES around minus '-'")
    print("  Example: '44-19+35' triggers bug, '44 - 19 + 35' does not")
    print()
    print("ROOT CAUSE:")
    print("  Line 76 in _numbers_match():")
    print("    digits = list(map(int, re.findall(r'-?\\d+', expr)))")
    print("  Line 88 in _reasoning_uses_numbers():")
    print("    digits = list(map(int, re.findall(r'-?\\d+', reasoning)))")
    print()
    print("  The regex r'-?\\d+' matches '-19' as a negative number when no space")
    print("  But '-19' != '19' in the Counter comparison")
    print()
    print("FIX:")
    print("  Change r'-?\\d+' to r'\\d+' in BOTH locations")
    print("  This will extract only positive integers (correct behavior)")
    print()
    print("IMPACT ON TRAINING:")
    print(f"  {bugs_found}/{len(test_cases)} test cases affected")
    print("  If models generate compact expressions (common!), they will be penalized")
    print("  Models will learn to AVOID subtraction or ALWAYS add spaces")
    print("  This CORRUPTS the training signal with wrong rewards")
    print()
    print("SEVERITY: CRITICAL - Fix before training!")
else:
    print()
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")

print("=" * 80)
