"""Comprehensive physics tests for the FSO beam propagation simulator.

Tests verify physical and mathematical correctness of Fourier transforms,
phase screens, vacuum propagation, atmospheric parameters, sampling
constraints, and screen r0 optimization.
"""

import pytest
import torch
import math
import numpy as np

# Use CPU for tests if no GPU available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestFTUtils:
    """Section 10.1 tests #1-3: Fourier transform correctness."""

    def test_ft2_ift2_roundtrip(self):
        """ft2/ift2 roundtrip: max error < 1e-10"""
        N = 256
        delta = 0.01
        g = torch.randn(N, N, dtype=torch.float64, device=DEVICE) + \
            1j * torch.randn(N, N, dtype=torch.float64, device=DEVICE)
        from kim2026.fso.ft_utils import ft2, ift2
        delta_f = 1 / (N * delta)
        g_recovered = ift2(ft2(g, delta), delta_f)
        assert torch.max(torch.abs(g - g_recovered)).item() < 1e-10

    def test_ft2_delta_function(self):
        """ft2(delta) = constant: max relative error < 1e-10"""
        N = 256
        delta = 0.01
        g = torch.zeros(N, N, dtype=torch.complex128, device=DEVICE)
        g[N // 2, N // 2] = 1.0
        from kim2026.fso.ft_utils import ft2
        G = ft2(g, delta)
        # FT of delta at center should be constant = delta^2
        expected = delta ** 2
        assert torch.max(torch.abs(G - expected)).item() / abs(expected) < 1e-10

    def test_parsevals_theorem(self):
        """Parseval: sum|g|^2*delta^2 ~ sum|G|^2*delta_f^2"""
        N = 256
        delta = 0.01
        delta_f = 1 / (N * delta)
        g = torch.randn(N, N, dtype=torch.complex128, device=DEVICE)
        from kim2026.fso.ft_utils import ft2
        G = ft2(g, delta)
        energy_spatial = torch.sum(torch.abs(g) ** 2).item() * delta ** 2
        energy_freq = torch.sum(torch.abs(G) ** 2).item() * delta_f ** 2
        assert abs(energy_spatial - energy_freq) / energy_spatial < 1e-10


class TestPhaseScreen:
    """Section 10.1 test #4: Phase screen statistical properties."""

    def test_phase_screen_mean_near_zero(self):
        """Phase screen mean ~ 0: |mean| < 0.1 rad"""
        from kim2026.fso.phase_screen import ft_sh_phase_screen
        r0 = 0.1  # 10 cm
        N = 256
        delta = 0.01  # 1 cm
        phz = ft_sh_phase_screen(r0, N, delta, device=DEVICE)
        assert abs(phz.mean().item()) < 0.1

    def test_phase_screen_shape_and_dtype(self):
        """Phase screen has correct shape and dtype."""
        from kim2026.fso.phase_screen import ft_sh_phase_screen
        r0 = 0.1
        N = 128
        delta = 0.01
        phz = ft_sh_phase_screen(r0, N, delta, device=DEVICE)
        assert phz.shape == (N, N)
        assert phz.dtype == torch.float64

    def test_phase_screen_variance_scales_with_r0(self):
        """Smaller r0 (stronger turbulence) produces larger phase variance."""
        from kim2026.fso.phase_screen import ft_sh_phase_screen
        N = 256
        delta = 0.01
        phz_strong = ft_sh_phase_screen(0.02, N, delta, device=DEVICE)
        phz_weak = ft_sh_phase_screen(0.5, N, delta, device=DEVICE)
        var_strong = phz_strong.var().item()
        var_weak = phz_weak.var().item()
        assert var_strong > var_weak, (
            f"Strong turbulence variance {var_strong} should exceed "
            f"weak turbulence variance {var_weak}"
        )

    def test_generate_phase_screens_count(self):
        """generate_phase_screens returns correct number of screens."""
        from kim2026.fso.phase_screen import generate_phase_screens
        r0_values = [0.1, 0.2, 0.15]
        N = 64
        delta_values = [0.01, 0.012, 0.014]
        screens = generate_phase_screens(r0_values, N, delta_values, device=DEVICE)
        assert len(screens) == 3
        for scr in screens:
            assert scr.shape == (N, N)


class TestVacuumPropagation:
    """Section 10.1 test #5 and Section 10.2 tests #8, #11."""

    def test_gaussian_vacuum_propagation(self):
        """Gaussian beam vacuum propagation vs analytic: irradiance relative error < 5% in ROI."""
        from kim2026.fso.propagation import (
            make_gaussian_source,
            ang_spec_multi_prop,
            compute_irradiance,
        )

        wvl = 1550e-9
        k = 2 * math.pi / wvl
        w0 = 0.01  # 1 cm beam waist
        Dz = 100.0  # 100 m
        z_R = math.pi * w0 ** 2 / wvl  # Rayleigh range

        N = 512
        delta1 = 4 * w0 / (N * 0.3)  # source grid spacing
        deltan = delta1  # approximately equal for short propagation

        n_planes = 5
        z_planes = [i * Dz / (n_planes - 1) for i in range(n_planes)]

        U_in = make_gaussian_source(N, delta1, w0, device=DEVICE)
        xn, yn, U_out = ang_spec_multi_prop(
            U_in, wvl, delta1, deltan, z_planes,
            phase_screens=None, device=DEVICE,
        )
        I_sim = compute_irradiance(U_out)

        # Analytic Gaussian beam propagation
        w_z = w0 * math.sqrt(1 + (Dz / z_R) ** 2)

        Xn, Yn = torch.meshgrid(xn, yn, indexing="ij")
        rn_sq = Xn ** 2 + Yn ** 2

        I_analytic = (w0 / w_z) ** 2 * torch.exp(-2 * rn_sq / w_z ** 2)

        # Normalize both to peak
        I_sim_norm = I_sim / I_sim.max()
        I_analytic_norm = I_analytic / I_analytic.max()

        # Compare in ROI (where irradiance > 1% of peak)
        roi = I_analytic_norm > 0.01
        assert roi.sum() > 0, "ROI is empty — grid may be too small"
        rel_err = torch.abs(I_sim_norm[roi] - I_analytic_norm[roi]) / I_analytic_norm[roi]
        mean_rel_err = rel_err.mean().item()
        assert mean_rel_err < 0.05, f"Mean relative error {mean_rel_err:.4f} > 5%"

    def test_vacuum_energy_conservation(self):
        """Vacuum propagation energy conservation: < 5% loss."""
        from kim2026.fso.propagation import make_gaussian_source, ang_spec_multi_prop

        wvl = 1550e-9
        w0 = 0.005
        Dz = 50.0
        N = 256
        delta1 = 4 * w0 / (N * 0.3)
        deltan = delta1
        n_planes = 3
        z_planes = [i * Dz / (n_planes - 1) for i in range(n_planes)]

        U_in = make_gaussian_source(N, delta1, w0, device=DEVICE)
        energy_in = torch.sum(torch.abs(U_in) ** 2).item() * delta1 ** 2

        xn, yn, U_out = ang_spec_multi_prop(
            U_in, wvl, delta1, deltan, z_planes,
            phase_screens=None, device=DEVICE,
        )
        energy_out = torch.sum(torch.abs(U_out) ** 2).item() * deltan ** 2

        loss = abs(energy_in - energy_out) / energy_in
        assert loss < 0.05, f"Energy loss {loss:.4f} > 5%"

    def test_vacuum_propagation_symmetry(self):
        """Vacuum-propagated Gaussian: x and y cross-sections should match."""
        from kim2026.fso.propagation import (
            make_gaussian_source,
            ang_spec_multi_prop,
            compute_irradiance,
        )

        wvl = 1550e-9
        w0 = 0.005
        Dz = 50.0
        N = 256
        delta1 = 4 * w0 / (N * 0.3)
        deltan = delta1
        z_planes = [0.0, Dz / 2, Dz]

        U_in = make_gaussian_source(N, delta1, w0, device=DEVICE)
        xn, yn, U_out = ang_spec_multi_prop(
            U_in, wvl, delta1, deltan, z_planes,
            phase_screens=None, device=DEVICE,
        )
        I_out = compute_irradiance(U_out)

        # For a Gaussian source the x and y cross-sections through center
        # should be identical (the beam is radially symmetric).
        c = N // 2
        I_x_slice = I_out[c, :]  # horizontal slice through center
        I_y_slice = I_out[:, c]  # vertical slice through center

        # Normalize and compare
        I_x_norm = I_x_slice / I_x_slice.max()
        I_y_norm = I_y_slice / I_y_slice.max()
        max_diff = torch.max(torch.abs(I_x_norm - I_y_norm)).item()
        assert max_diff < 1e-10, f"x/y cross-section mismatch {max_diff:.2e}"

    def test_make_gaussian_source_peak(self):
        """Gaussian source peak amplitude is 1.0 at grid center."""
        from kim2026.fso.propagation import make_gaussian_source

        N = 256
        delta1 = 0.001
        w0 = 0.01
        U = make_gaussian_source(N, delta1, w0, device=DEVICE)
        # Peak should be at center
        peak = torch.abs(U).max().item()
        assert abs(peak - 1.0) < 1e-12

    def test_make_gaussian_source_is_real(self):
        """Gaussian source (collimated) should have zero imaginary part."""
        from kim2026.fso.propagation import make_gaussian_source

        N = 128
        delta1 = 0.001
        w0 = 0.01
        U = make_gaussian_source(N, delta1, w0, device=DEVICE)
        assert torch.max(torch.abs(U.imag)).item() < 1e-15


class TestAtmosphere:
    """Test atmospheric parameter computations."""

    def test_r0_values(self):
        """r0_sw > r0_pw for same Cn2 and Dz (spherical wave sees less turbulence)."""
        from kim2026.fso.atmosphere import compute_atmospheric_params

        k = 2 * math.pi / 1550e-9
        params = compute_atmospheric_params(k, 1e-15, 10e3)
        assert params["r0_sw"] > params["r0_pw"]

    def test_r0_pw_known_value(self):
        """Check r0_pw against hand calculation."""
        from kim2026.fso.atmosphere import compute_atmospheric_params

        wvl = 1550e-9
        k = 2 * math.pi / wvl
        Cn2 = 1e-15
        Dz = 1000.0
        params = compute_atmospheric_params(k, Cn2, Dz)
        # r0_pw = (0.423 * k^2 * Cn2 * Dz)^(-3/5)
        r0_expected = (0.423 * k ** 2 * Cn2 * Dz) ** (-3 / 5)
        assert abs(params["r0_pw"] - r0_expected) / r0_expected < 1e-10

    def test_r0_sw_known_value(self):
        """Check r0_sw against hand calculation with 3/8 factor."""
        from kim2026.fso.atmosphere import compute_atmospheric_params

        wvl = 1550e-9
        k = 2 * math.pi / wvl
        Cn2 = 1e-15
        Dz = 1000.0
        params = compute_atmospheric_params(k, Cn2, Dz)
        r0_expected = (0.423 * k ** 2 * Cn2 * (3.0 / 8.0) * Dz) ** (-3 / 5)
        assert abs(params["r0_sw"] - r0_expected) / r0_expected < 1e-10

    def test_weak_fluctuation_flag(self):
        """Weak fluctuation flag correct for known weak and strong cases."""
        from kim2026.fso.atmosphere import compute_atmospheric_params

        k = 2 * math.pi / 1550e-9
        # Short path, weak turbulence => weak fluctuation
        params_weak = compute_atmospheric_params(k, 1e-17, 100.0)
        assert params_weak["weak_fluctuation"] == True

        # Long path, strong turbulence => strong fluctuation
        params_strong = compute_atmospheric_params(k, 1e-13, 50e3)
        assert params_strong["weak_fluctuation"] == False

    def test_r0_decreases_with_cn2(self):
        """Stronger turbulence (larger Cn2) gives smaller r0."""
        from kim2026.fso.atmosphere import compute_atmospheric_params

        k = 2 * math.pi / 1550e-9
        Dz = 5000.0
        params_weak = compute_atmospheric_params(k, 1e-16, Dz)
        params_strong = compute_atmospheric_params(k, 1e-14, Dz)
        assert params_strong["r0_pw"] < params_weak["r0_pw"]
        assert params_strong["r0_sw"] < params_weak["r0_sw"]

    def test_sigma2_chi_positive(self):
        """Rytov log-amplitude variance is always positive."""
        from kim2026.fso.atmosphere import compute_atmospheric_params

        k = 2 * math.pi / 1550e-9
        params = compute_atmospheric_params(k, 1e-15, 10e3)
        assert params["sigma2_chi_sw"] > 0


class TestSampling:
    """Test sampling constraint analysis."""

    def test_constraints_satisfied(self):
        """All sampling constraints must be satisfied."""
        from kim2026.fso.config import SimulationConfig
        from kim2026.fso.sampling import analyze_sampling

        cfg = SimulationConfig(
            Dz=10e3, Cn2=1e-15, theta_div=1e-3,
            D_roi=1.0, delta_n=5e-3, n_reals=5,
        )
        result = analyze_sampling(cfg)
        assert result.delta1 > 0
        assert result.N >= 64  # must be reasonable
        assert result.n_scr >= 2
        assert (result.N & (result.N - 1)) == 0  # power of 2

    def test_n_is_power_of_two(self):
        """Grid size N must be a power of 2 for FFT efficiency."""
        from kim2026.fso.config import SimulationConfig
        from kim2026.fso.sampling import analyze_sampling

        cfg = SimulationConfig(
            Dz=5e3, Cn2=1e-15, theta_div=2e-3,
            D_roi=0.5, delta_n=3e-3, n_reals=5,
        )
        result = analyze_sampling(cfg)
        N = result.N
        assert N > 0
        assert (N & (N - 1)) == 0, f"N={N} is not a power of 2"

    def test_z_planes_span_full_distance(self):
        """z_planes should start at 0 and end at Dz."""
        from kim2026.fso.config import SimulationConfig
        from kim2026.fso.sampling import analyze_sampling

        cfg = SimulationConfig(
            Dz=10e3, Cn2=1e-15, theta_div=1e-3,
            D_roi=1.0, delta_n=5e-3, n_reals=5,
        )
        result = analyze_sampling(cfg)
        assert abs(result.z_planes[0]) < 1e-10
        assert abs(result.z_planes[-1] - cfg.Dz) < 1e-10

    def test_delta_values_interpolate(self):
        """Grid spacings should interpolate from delta1 to delta_n."""
        from kim2026.fso.config import SimulationConfig
        from kim2026.fso.sampling import analyze_sampling

        cfg = SimulationConfig(
            Dz=10e3, Cn2=1e-15, theta_div=1e-3,
            D_roi=1.0, delta_n=5e-3, n_reals=5,
        )
        result = analyze_sampling(cfg)
        assert abs(result.delta_values[0] - result.delta1) < 1e-15
        assert abs(result.delta_values[-1] - result.delta_n) < 1e-15

    def test_invalid_delta_n_raises(self):
        """Too-large delta_n should raise ValueError."""
        from kim2026.fso.config import SimulationConfig
        from kim2026.fso.sampling import analyze_sampling

        cfg = SimulationConfig(
            Dz=10e3, Cn2=1e-15, theta_div=1e-3,
            D_roi=1.0, delta_n=1.0,  # way too large
            n_reals=5,
        )
        with pytest.raises(ValueError, match="No feasible delta1"):
            analyze_sampling(cfg)


class TestScreenR0Optimization:
    """Test phase screen r0 distribution."""

    def test_r0_optimization_converges(self):
        """Optimized r0 values reconstruct target r0_sw and sigma2_chi within tolerance."""
        from kim2026.fso.atmosphere import compute_atmospheric_params, optimize_screen_r0

        wvl = 1550e-9
        k = 2 * math.pi / wvl
        Cn2 = 1e-15
        Dz = 10e3
        params = compute_atmospheric_params(k, Cn2, Dz)
        r0_sw = params["r0_sw"]
        sigma2 = params["sigma2_chi_sw"]

        n_scr = 10
        r0_vals = optimize_screen_r0(r0_sw, sigma2, k, Dz, n_scr)

        # Reconstruct r0_sw and sigma2_chi using the same alpha convention
        # as optimize_screen_r0: equally spaced along [0, Dz), dropping endpoint
        z_planes = np.linspace(0, Dz, n_scr + 1)[:-1]
        alpha = z_planes / Dz

        x = r0_vals ** (-5 / 3)

        r0_recon = (np.sum(x * alpha ** (5 / 3))) ** (-3 / 5)
        kDz_56 = (k / Dz) ** (5 / 6)
        sigma2_recon = np.sum(
            x * alpha ** (5 / 6) * (1 - alpha) ** (5 / 6)
        ) * 1.33 / kDz_56

        assert abs(r0_recon - r0_sw) / r0_sw < 0.01 or abs(r0_recon - r0_sw) < 0.001
        # sigma2 might be harder to match exactly, allow 5%
        if sigma2 > 1e-6:
            assert abs(sigma2_recon - sigma2) / sigma2 < 0.05

    def test_r0_values_positive(self):
        """All optimized r0 values must be positive."""
        from kim2026.fso.atmosphere import compute_atmospheric_params, optimize_screen_r0

        wvl = 1550e-9
        k = 2 * math.pi / wvl
        Cn2 = 1e-15
        Dz = 5e3
        params = compute_atmospheric_params(k, Cn2, Dz)
        r0_vals = optimize_screen_r0(
            params["r0_sw"], params["sigma2_chi_sw"], k, Dz, 5,
        )
        assert np.all(r0_vals > 0)

    def test_r0_optimization_correct_count(self):
        """optimize_screen_r0 returns exactly n_scr values."""
        from kim2026.fso.atmosphere import compute_atmospheric_params, optimize_screen_r0

        wvl = 1550e-9
        k = 2 * math.pi / wvl
        Cn2 = 1e-15
        Dz = 10e3
        params = compute_atmospheric_params(k, Cn2, Dz)

        for n_scr in [5, 10, 15, 20]:
            r0_vals = optimize_screen_r0(
                params["r0_sw"], params["sigma2_chi_sw"], k, Dz, n_scr,
            )
            assert len(r0_vals) == n_scr

    def test_r0_optimization_rejects_single_screen(self):
        """optimize_screen_r0 should reject n_scr < 2."""
        from kim2026.fso.atmosphere import compute_atmospheric_params, optimize_screen_r0

        wvl = 1550e-9
        k = 2 * math.pi / wvl
        params = compute_atmospheric_params(k, 1e-15, 10e3)
        with pytest.raises(ValueError, match="n_scr must be >= 2"):
            optimize_screen_r0(params["r0_sw"], params["sigma2_chi_sw"], k, 10e3, 1)


class TestComputeIrradiance:
    """Test irradiance computation."""

    def test_irradiance_nonnegative(self):
        """Irradiance |U|^2 must be non-negative."""
        from kim2026.fso.propagation import compute_irradiance

        U = torch.randn(64, 64, dtype=torch.complex128, device=DEVICE)
        I = compute_irradiance(U)
        assert torch.all(I >= 0).item()

    def test_irradiance_value(self):
        """Irradiance of known field matches |U|^2."""
        from kim2026.fso.propagation import compute_irradiance

        U = torch.tensor([[1 + 2j, 3 + 4j]], dtype=torch.complex128, device=DEVICE)
        I = compute_irradiance(U)
        expected = torch.tensor([[5.0, 25.0]], dtype=torch.float64, device=DEVICE)
        assert torch.allclose(I, expected, atol=1e-12)


class TestMainPipeline:
    """Smoke test for the full simulation pipeline."""

    @pytest.mark.slow
    def test_run_simulation_smoke(self, tmp_path, monkeypatch):
        """run_simulation completes without error and produces expected outputs."""
        from kim2026.fso.config import SimulationConfig
        from kim2026.fso import main as main_module

        # Force CPU to avoid GPU OOM in CI / shared environments
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        # Chosen to give N=64, n_scr=5 for fast execution
        cfg = SimulationConfig(
            Dz=100.0, Cn2=1e-17, theta_div=1e-3,
            D_roi=0.1, delta_n=5e-3, n_reals=2,
        )
        output_dir = str(tmp_path / "sim_output")
        results = main_module.run_simulation(cfg, output_dir=output_dir)

        # Check output structure
        assert (tmp_path / "sim_output" / "config.json").exists()
        assert (tmp_path / "sim_output" / "sampling_analysis.json").exists()
        assert (tmp_path / "sim_output" / "screen_r0.json").exists()
        assert (tmp_path / "sim_output" / "coordinates.pt").exists()
        assert (tmp_path / "sim_output" / "vacuum" / "field.pt").exists()
        assert (tmp_path / "sim_output" / "vacuum" / "irradiance.pt").exists()
        assert (tmp_path / "sim_output" / "turbulence" / "field_0000.pt").exists()
        assert (tmp_path / "sim_output" / "turbulence" / "field_0001.pt").exists()
        assert (tmp_path / "sim_output" / "verification" / "structure_function_report.json").exists()
        assert (tmp_path / "sim_output" / "verification" / "coherence_factor_report.json").exists()

        # Check results dict has expected keys
        assert "atmospheric_params" in results
        assert "sampling" in results
        assert "verification" in results
        assert "output_dir" in results
