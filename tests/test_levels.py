from tajator.market.levels import detect_levels

from conftest import make_bar, ts, walk


def rth_flat(price: float, minutes: int = 30):
    return walk(ts(9, 30), [price] * minutes)


def test_prev_day_levels_and_kind():
    bars = rth_flat(500.0)
    levels = detect_levels(bars, prev_day_high=503.5, prev_day_low=497.0)
    by_label = {l.label: l for l in levels}
    assert by_label["prev_day_high"].kind == "resistance"
    assert by_label["prev_day_low"].kind == "support"


def test_premarket_levels_detected():
    pre = [
        make_bar(ts(8, 0), 501.0, h=501.8, lo=500.5),
        make_bar(ts(8, 30), 499.0, h=499.5, lo=498.2),
    ]
    bars = pre + rth_flat(500.0)
    labels = {l.label: l.price for l in detect_levels(bars)}
    assert labels["premarket_high"] == 501.8
    assert labels["premarket_low"] == 498.2


def test_double_top_from_clustered_swings():
    # Rally to ~502 twice with a pullback in between, then sell off.
    path = (
        [500 + 0.2 * i for i in range(11)]  # 500 -> 502 first high
        + [502 - 0.15 * i for i in range(1, 9)]  # pull back to ~500.8
        + [500.8 + 0.15 * i for i in range(1, 9)]  # back up to ~502 second high
        + [502 - 0.2 * i for i in range(1, 11)]  # sell off to ~500
    )
    bars = walk(ts(9, 30), path)
    levels = detect_levels(bars)
    double_tops = [l for l in levels if l.label == "double_top"]
    assert double_tops, f"no double top in {levels}"
    assert abs(double_tops[0].price - 502.0) < 0.5
    assert double_tops[0].kind == "resistance"


def test_dedupe_prefers_stronger_label():
    # Premarket high sits exactly on prev-day high: keep prev_day_high only.
    pre = [make_bar(ts(8, 0), 503.4, h=503.5, lo=503.0)]
    bars = pre + rth_flat(500.0)
    levels = detect_levels(bars, prev_day_high=503.5, prev_day_low=490.0)
    near = [l for l in levels if abs(l.price - 503.5) < 1.0]
    assert len(near) == 1
    assert near[0].label == "prev_day_high"
