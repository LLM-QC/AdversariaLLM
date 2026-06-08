from adversariallm.defenses import DEFENSE_COMPATIBLE_ATTACKS


def test_runtime_defense_support_matrix():
    assert "actor" in DEFENSE_COMPATIBLE_ATTACKS
    assert "crescendo" in DEFENSE_COMPATIBLE_ATTACKS
    assert "inpainting" in DEFENSE_COMPATIBLE_ATTACKS
    assert "gcg" not in DEFENSE_COMPATIBLE_ATTACKS
