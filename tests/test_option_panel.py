from tajator.backtest.audit import paired_panel_rows, print_option_panel


def test_panel_comparison_discloses_incomplete_variant(capsys):
    trade = {
        "day": "2026-07-13", "closed": True, "pnl": 10.0,
        "return_on_premium": 0.05,
    }
    report = {
        "symbol": "MSFT",
        "trades": [trade],
        "option_panel": {
            "itm_1_near": {
                "trades": [trade], "complete": False,
                "missing_contracts": [{"reason": "missing"}],
            }
        },
    }
    print_option_panel(report)
    output = capsys.readouterr().out
    assert "base_atm_near" in output
    assert "itm_1_near" in output
    assert "incomplete" in output
    assert "base options edge confirmed: NO" in output


def test_paired_panel_requires_fifty_complete_pairs_for_advantage():
    base = []
    alternative = []
    for index in range(50):
        common = {
            "day": f"2026-07-{index % 20 + 1:02d}",
            "entry_ts": f"2026-07-13T10:{index:02d}:00-04:00",
            "exit_ts": f"2026-07-13T11:{index:02d}:00-04:00",
            "direction": "call", "qty": 1, "closed": True, "pnl": 10.0,
        }
        base.append({**common, "return_on_premium": 0.02})
        alternative.append({**common, "return_on_premium": 0.05})
    report = {
        "trades": base,
        "option_panel": {"otm_1_near": {"trades": alternative, "complete": True}},
    }
    row = paired_panel_rows(report)["otm_1_near"]
    assert row.pairs == 50
    assert row.mean_return_improvement == 0.03
    assert row.verdict == "positive_paired_advantage"


def test_paired_panel_matches_by_signal_not_list_position():
    base = [{
        "day": "2026-07-13", "entry_ts": "10:00", "exit_ts": "10:05",
        "direction": "call", "qty": 1, "return_on_premium": 0.02,
    }]
    variant = [{
        "day": "2026-07-14", "entry_ts": "10:00", "exit_ts": "10:05",
        "direction": "call", "qty": 1, "return_on_premium": 0.10,
    }]
    row = paired_panel_rows({
        "trades": base,
        "option_panel": {"otm_1_near": {"trades": variant, "complete": True}},
    })["otm_1_near"]
    assert row.pairs == 0
