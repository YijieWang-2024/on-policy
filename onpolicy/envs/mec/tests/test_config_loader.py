"""Unit tests for the v2 scenario loader (spec section 6, physics-correctness).

Runnable standalone (`python3 test_config_loader.py`) or via pytest.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from onpolicy.envs.mec.config_loader import (  # noqa: E402
    backhaul_service_radius_m,
    derive_cost_weights,
    load_scenario,
)


def _snr_db_at_horizontal(cfg, horiz_m: float) -> float:
    d = cfg["derived"]
    dz = float(cfg["env"]["hap"]["altitude_m"]) - float(cfg["env"]["uav"]["altitude_m"])
    dist = math.hypot(horiz_m, dz)
    return (d["bh_link_budget_const_db"] - 20.0 * math.log10(dist)
            - d["bh_kappa_o2_db_per_km"] * dist / 1000.0)


def test_locked_scenario_loads_with_locked_values():
    cfg = load_scenario()
    assert cfg["env"]["fleet_size_k"] == 24
    assert cfg["env"]["region"]["lx_m"] == 6000.0
    assert cfg["env"]["hap"]["altitude_m"] == 1500.0
    assert cfg["env"]["slot_timing"]["service_position"] == "mid_move"
    assert len(cfg["env"]["uav"]["initial_xy_m"]) == 24
    assert len(cfg["env"]["uav"]["initial_queue_bits"]) == 24
    # grid deployment inside [0.2, 0.8] of the region
    for x, y in cfg["env"]["uav"]["initial_xy_m"]:
        assert 1200.0 - 1e-9 <= x <= 4800.0 + 1e-9
        assert 1200.0 - 1e-9 <= y <= 4800.0 + 1e-9


def test_compute_capacities_match_spec():
    cfg = load_scenario()
    d = cfg["derived"]
    assert abs(d["uav_compute_capacity_bits"] - 0.25e6) < 1e-6   # 1 GHz / 4000
    assert abs(d["hap_compute_capacity_bits"] - 7.5e6) < 1e-6    # 30 GHz / 4000
    # offered load sanity (spec 1.3): hot 2*pi*sigma^2*rho*p_hot*b0 ~ 32 Mbit
    hot = (2 * math.pi * 800.0 ** 2 * 1e-3 * 8e-3 * 1e6)
    assert 30e6 < hot < 35e6


def test_cost_weights_formula_derived_and_k_dependent():
    cfg = load_scenario()
    w = cfg["cost"]["weights"]
    assert abs(w["queue_ref_bits"] - (24 * 80e6 + 150e6)) < 1e-3
    assert abs(w["omega_queue_per_bit"] - 1.0 / 2.07e9) < 1e-22
    assert abs(w["omega_src_per_bit"] - 5.0e-8) < 1e-20   # lambda_src=5 / D_ref=1e8
    assert abs(w["omega_ovf_per_bit"] - 5.0e-8) < 1e-20   # lambda_ovf=5 (equal; never > src)
    assert w["omega_ovf_per_bit"] <= w["omega_src_per_bit"] + 1e-20
    assert abs(w["omega_energy_per_j"] - 3.0e-6) < 1e-18
    assert abs(w["omega_safety"] - 0.5 / 276.0) < 1e-15
    # R2: K sweep recomputes every K-dependent quantity
    cfg8 = load_scenario(fleet_size_k=8)
    w8 = cfg8["cost"]["weights"]
    assert abs(w8["queue_ref_bits"] - (8 * 80e6 + 150e6)) < 1e-3
    assert abs(w8["omega_safety"] - 0.5 / 28.0) < 1e-15
    assert len(cfg8["env"]["uav"]["initial_xy_m"]) == 8
    assert abs(cfg8["communication"]["access"]["bandwidth_per_uav_hz"] - 1e7) < 1e-6
    assert derive_cost_weights(cfg8) == w8


def test_access_constants_match_origin_repo():
    cfg = load_scenario()
    d = cfg["derived"]
    # identical formulas to origin mfmec.config.derive_constants
    thermal = 10.0 ** ((-174.0 - 30.0) / 10.0)
    noise_eff = thermal * 10.0 ** 0.9
    assert abs(d["noise_psd_eff_w_per_hz"] - noise_eff) / noise_eff < 1e-12
    assert abs(d["access_gain_threshold"] - noise_eff * 100.0 / 1e-7) / d["access_gain_threshold"] < 1e-12
    assert abs(d["max_per_device_bandwidth_hz"] - 1e6) < 1e-6
    assert abs(cfg["communication"]["access"]["pathloss"]["reference_loss_linear_at_1m"]
               - 10106.474906715503) < 1e-9


def test_mmwave_link_budget_key_points():
    """Spec section 6: horiz 0/1/2/2.3 km -> SNR 17.4/11.5/0.3/-3.2 dB."""
    cfg = load_scenario()
    expected = {0.0: 17.4, 1000.0: 11.5, 2000.0: 0.3, 2300.0: -3.2}
    for horiz, snr_expected in expected.items():
        snr = _snr_db_at_horizontal(cfg, horiz)
        assert abs(snr - snr_expected) < 0.15, (horiz, snr, snr_expected)
    # demod gate: 2.3 km is below threshold -> rate would be 0
    assert _snr_db_at_horizontal(cfg, 2300.0) < cfg["derived"]["bh_demod_snr_min_db"]
    radius = backhaul_service_radius_m(cfg)
    assert 2000.0 < radius < 2400.0, radius   # ~2.2 km service circle
    # region must exceed 2x service radius so the hub is forced to move (spec 5)
    assert cfg["env"]["region"]["lx_m"] > 2.0 * radius


def test_v6_continuous_workload_scenario_sanity():
    cfg = load_scenario("v6_continuous_workload")
    assert cfg["env"]["fleet_size_k"] == 16
    assert cfg["communication"]["backhaul"]["link_model"] == "continuous_mmwave"
    assert cfg["demand"]["workload_field"]["model"] == "normalized_background_gaussian"
    assert abs(cfg["communication"]["access"]["bandwidth_per_uav_hz"] - 40e6 / 16) < 1e-6
    assert abs(cfg["communication"]["backhaul"]["mmwave"]["beam_bandwidth_hz"] - 400e6 / 16) < 1e-6
    assert abs(cfg["derived"]["uav_compute_capacity_bits"] - 4.0e6) < 1e-6
    assert abs(cfg["derived"]["hap_compute_capacity_bits"] - 90.0e6) < 1e-6
    assert math.isinf(backhaul_service_radius_m(cfg))

    field = cfg["demand"]["workload_field"]
    a_tot = float(field["total_workload_bits_per_slot"])
    assert abs(a_tot - 150e6) < 1e-6
    assert 0.6 <= float(field["hotspot_fraction"]) <= 0.75
    f_total = (cfg["env"]["fleet_size_k"] * float(cfg["env"]["uav"]["cpu_frequency_hz"])
               + float(cfg["env"]["hap"]["cpu_frequency_hz"]))
    offered_compute_ratio = float(cfg["compute"]["cycles_per_bit"]) * a_tot / f_total
    assert 0.9 <= offered_compute_ratio <= 1.05


def test_v6_hap_loadbearing_link_budget():
    cfg = load_scenario("v6_hap_loadbearing")
    mmw = cfg["communication"]["backhaul"]["mmwave"]
    assert mmw["total_bandwidth_hz"] == 400e6
    assert mmw["beam_bandwidth_hz"] == 25e6
    assert mmw["tx_power_dbm"] == 23.0
    assert mmw["antenna_gain_total_db"] == 20.0
    assert mmw["link_margin_db"] == 7.0
    assert cfg["derived"]["bh_link_margin_db"] == 7.0


def test_validation_rejects_hardcoded_weights():
    cfg = load_scenario()
    cfg["cost"]["weights"]["omega_ovf_per_bit"] *= 1.5
    try:
        from onpolicy.envs.mec.config_loader import validate_scenario
        validate_scenario(cfg)
    except ValueError as e:
        assert "formula-derived" in str(e)
    else:
        raise AssertionError("tampered omega must be rejected (R2)")


if __name__ == "__main__":
    mod = sys.modules["__main__"]
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
