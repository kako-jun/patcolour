from patcolour.cli import _parse_ellipse, _parse_point, _parse_rect


def test_parse_rect() -> None:
    assert _parse_rect("10,20,30,40") == (10, 20, 30, 40)


def test_parse_ellipse() -> None:
    assert _parse_ellipse("50,60,70,80") == (50, 60, 70, 80)


def test_parse_point() -> None:
    assert _parse_point("12,34") == (12, 34)
