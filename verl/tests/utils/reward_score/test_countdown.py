from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.countdown import compute_score


def test_countdown_compute_score_correct():
    gt = {"numbers": [44, 19, 35], "target": 98}
    res = compute_score("some text <answer>(44 + 19) + 35</answer>", gt)
    assert res["acc"] is True
    assert float(res["score"]) == 1.0


def test_countdown_allows_equation_with_target_rhs():
    gt = {"numbers": [44, 19, 35], "target": 98}
    res = compute_score("<answer>(44 + 19) + 35 = 98</answer>", gt)
    assert res["acc"] is False
    assert float(res["score"]) == 0.0


def test_countdown_rejects_extra_number():
    gt = {"numbers": [44, 19, 35], "target": 98}
    res = compute_score("<answer>(44 + 19) + 35 + 0</answer>", gt)
    assert res["acc"] is False
    assert float(res["score"]) == 0.0


def test_countdown_supports_duplicate_numbers():
    gt = {"numbers": [99, 26, 26], "target": 100}
    res = compute_score("<answer>99 + 26 / 26</answer>", gt)
    assert res["acc"] is True
    assert float(res["score"]) == 1.0


def test_countdown_rejects_code_execution():
    gt = {"numbers": [44, 19, 35], "target": 98}
    res = compute_score("<answer>__import__('os').system('echo hi')</answer>", gt)
    assert res["acc"] is False
    assert float(res["score"]) == 0.0


def test_countdown_numbers_match_without_spaces_in_subtraction():
    gt = {"numbers": [44, 19, 35], "target": 28}
    res = compute_score("<answer>44+19-35</answer>", gt)
    assert res["acc"] is True
    assert float(res["score"]) == 1.0


def test_default_compute_score_dispatches_countdown():
    gt = {"numbers": [44, 19, 35], "target": 98}
    res = default_compute_score("countdown", "<answer>(44 + 19) + 35</answer>", gt)
    assert isinstance(res, dict)
    assert res["acc"] is True
    assert float(res["score"]) == 1.0
