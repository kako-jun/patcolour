import pytest

from patcolour.cli import (
    _parse_ellipse,
    _parse_ellipse_rel,
    _parse_point,
    _parse_point_rel,
    _parse_rect,
    _parse_rect_rel,
)


def test_parse_rect() -> None:
    assert _parse_rect("10,20,30,40") == (10, 20, 30, 40)


def test_parse_ellipse() -> None:
    assert _parse_ellipse("50,60,70,80") == (50, 60, 70, 80)


def test_parse_point() -> None:
    assert _parse_point("12,34") == (12, 34)


# ---------------------------------------------------------------------------
# _parse_point_rel / _parse_rect_rel / _parse_ellipse_rel
# ---------------------------------------------------------------------------


def test_parse_point_rel_valid() -> None:
    assert _parse_point_rel("0.5,0.5") == (0.5, 0.5)


def test_parse_rect_rel_valid() -> None:
    assert _parse_rect_rel("0.1,0.2,0.5,0.6") == (0.1, 0.2, 0.5, 0.6)


def test_parse_ellipse_rel_valid() -> None:
    assert _parse_ellipse_rel("0.5,0.5,0.25,0.25") == (0.5, 0.5, 0.25, 0.25)


def test_parse_point_rel_boundary_zero() -> None:
    assert _parse_point_rel("0.0,0.0") == (0.0, 0.0)


def test_parse_point_rel_boundary_one() -> None:
    assert _parse_point_rel("1.0,1.0") == (1.0, 1.0)


def test_parse_rect_rel_boundary_all_zero() -> None:
    assert _parse_rect_rel("0.0,0.0,0.0,0.0") == (0.0, 0.0, 0.0, 0.0)


def test_parse_rect_rel_boundary_all_one() -> None:
    assert _parse_rect_rel("1.0,1.0,1.0,1.0") == (1.0, 1.0, 1.0, 1.0)


def test_parse_point_rel_negative_raises() -> None:
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_point_rel("-0.1,0.5")


def test_parse_point_rel_over_one_raises() -> None:
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_point_rel("1.1,0.5")


def test_parse_rect_rel_negative_w_raises() -> None:
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_rect_rel("0.1,0.2,-0.1,0.5")


def test_parse_ellipse_rel_over_one_rry_raises() -> None:
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_ellipse_rel("0.5,0.5,0.25,1.5")


def test_parse_point_rel_wrong_component_count_raises() -> None:
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_point_rel("0.5,0.5,0.5")


def test_parse_rect_rel_wrong_component_count_raises() -> None:
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_rect_rel("0.1,0.2,0.3")


def test_parse_ellipse_rel_non_numeric_raises() -> None:
    import argparse
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        _parse_ellipse_rel("a,b,c,d")


def test_parse_point_rel_empty_string_raises() -> None:
    import argparse
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        _parse_point_rel("")
