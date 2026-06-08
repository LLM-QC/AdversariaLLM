from adversariallm.attacks.attack import Attack


def test_runtime_defense_support_matrix():
    assert Attack.supports_runtime_text_defense("actor")
    assert Attack.supports_runtime_text_defense("crescendo")
    assert Attack.supports_runtime_text_defense("inpainting")
    assert not Attack.supports_runtime_text_defense("gcg")
