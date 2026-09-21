"""Fast checks of physical identities, grids, and equation signs."""

from dataclasses import FrozenInstanceError, replace
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numerical as num
from config import get_config, validate_config
from spectral_density import evaluate, get_spectral_density


pm = get_config()


class ModelTests(unittest.TestCase):
    def test_article_parameter_mapping(self):
        self.assertEqual(pm.SPINS[0].DELTA, 2.0)
        self.assertEqual(pm.SPINS[0].H_FIELD, 0.0)
        self.assertAlmostEqual(pm.TEMP, 0.2)
        omega = 2.7
        parameters = pm.spectral_parameters()
        expected = 0.5 * parameters["xi"] * omega * np.exp(-omega / parameters["wc"])
        self.assertAlmostEqual(float(num.J(omega)), expected)

    def test_spectral_density_factorization(self):
        omega = np.linspace(pm.OMEGA_MIN, pm.OMEGA_MAX, 37)
        np.testing.assert_allclose(
            num.J(omega, 0.4),
            num.mode_density(omega) * num.coupling_squared(omega, 0.4),
            rtol=2e-15,
            atol=1e-15,
        )

    def test_c_coefficients_match_operator_evolution(self):
        time = 0.37
        frequency = num.sqrt_Omega_j[0]
        generator = (
            -num.DELTA_j[0] * num.sx + num.H_FIELD_j[0] * num.sz
        ) / frequency
        U = (
            np.cos(0.5 * frequency * time) * np.eye(2)
            - 1j * np.sin(0.5 * frequency * time) * generator
        )
        evolved = U.conj().T @ num.sz @ U
        direct = np.array(
            [0.5 * np.trace(evolved @ operator).real for operator in num.pauli]
        )
        np.testing.assert_allclose(num.c_coef_spin(time, 0), direct, atol=1e-14)

    def test_nonthermal_peak_is_used(self):
        at_center = float(num.N_I(pm.omega_car))
        thermal = float(num.bose_occupation(pm.omega_car))
        self.assertAlmostEqual(at_center - thermal, pm.Delta_N)

    def test_configuration_is_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            pm.LAMBDA = 1.0

    def test_perturbative_lambda_is_strictly_below_one(self):
        self.assertGreater(pm.LAMBDA, 0.0)
        self.assertLess(pm.LAMBDA, 1.0)
        with self.assertRaises(ValueError):
            validate_config(replace(pm, LAMBDA=1.0))

    def test_metadata_contains_no_local_paths(self):
        import run

        payload = run._base_metadata("numeric", "dynamic")
        self.assertNotIn("config_path", payload)
        self.assertNotIn("python_executable", payload)
        self.assertEqual(payload["config_name"], "config.toml")

    def test_saved_bath_data_uses_the_selected_density(self):
        import run

        density = "drude"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            run._save_bath_data(path, num, density, occupations=None)
            with np.load(path / "bath_data.npz") as saved:
                np.testing.assert_allclose(
                    saved["spectral_density"], num.J(num.omega_arr, density=density)
                )
                np.testing.assert_allclose(
                    saved["coupling_squared"],
                    num.J(num.omega_arr, density=density)
                    / num.mode_density(num.omega_arr),
                )

    def test_custom_density_function(self):
        omega = np.array([0.4, 1.2, 2.5])
        values = evaluate(
            "custom",
            omega,
            0.3,
            pm.spectral_parameters(),
        )
        self.assertEqual(values.shape, omega.shape)
        self.assertTrue(np.all(values >= 0.0))

    def test_spectral_density_is_selected_by_name(self):
        density = get_spectral_density("custom")
        omega = np.array([0.2, 0.7, 1.4])
        values = density(omega, 0.5, pm.spectral_parameters())
        self.assertEqual(values.shape, omega.shape)
        with self.assertRaises(ValueError):
            get_spectral_density("missing")

    def test_adaptive_grid_resolves_the_largest_delay(self):
        spacing = np.diff(num.quad_omega)
        self.assertLessEqual(
            float(np.max(spacing) * pm.T_MAX),
            num.MAX_PHASE_INCREMENT * (1.0 + 1.0e-10),
        )
        self.assertGreater(float(np.max(spacing) / np.min(spacing)), 1.1)

    def test_physical_grid_contains_requested_probe_frequencies(self):
        for frequency in pm.PROBE_FREQUENCIES:
            self.assertTrue(np.any(num.omega_arr == frequency))

    def test_adaptive_correlation_matches_dense_reference(self):
        dense_omega = np.linspace(pm.OMEGA_MIN, pm.OMEGA_MAX, 100_001)
        dense_n0 = num.N_I(dense_omega)
        dense_j = num.J(dense_omega)
        for delay in (0.0, 0.3, 2.0, 10.0):
            phase = dense_omega * delay
            integrand = dense_j * (
                (2.0 * dense_n0 + 1.0) * np.cos(phase)
                - 1j * np.sin(phase)
            )
            integrate = getattr(np, "trapezoid", np.trapz)
            reference = integrate(integrand, dense_omega)
            np.testing.assert_allclose(
                num.C_corr(delay, num.N0),
                reference,
                rtol=2.0e-4,
                atol=1.0e-4,
            )

    def test_one_spin_rhs_n_has_minus_sine_term(self):
        time = 0.4
        t1 = np.array([0.0, time])
        c_hist = np.asarray([num.c_coef_all(value) for value in t1])
        c_now = c_hist[-1]
        sigma = np.array([[0.2, -0.3, 0.4]])
        occupation = np.full_like(num.omega_arr, 0.15)
        actual = num.RHS_N(
            occupation, sigma, time, t1, c_hist, c_now, "ohmic"
        )

        weights = num.trapezoid_weights(t1)
        w_cos = np.einsum("hl,l->h", c_hist[:, 0, :], c_now[0])
        w_sin = np.array(
            [
                sum(
                    c_hist[h, 0, l]
                    * c_now[0, k]
                    * num.eps[l, k, gamma]
                    * sigma[0, gamma]
                    for l in range(3)
                    for k in range(3)
                    for gamma in range(3)
                )
                for h in range(len(t1))
            ]
        )
        tau = time - t1
        phase = np.outer(num.omega_arr, tau)
        manual = 2.0 * pm.LAMBDA**2 * num.coupling_squared(
            num.omega_arr, time
        ) * (
            np.cos(phase) @ (weights * w_cos)
            - (2.0 * occupation + 1.0) * (np.sin(phase) @ (weights * w_sin))
        )
        np.testing.assert_allclose(actual, manual, rtol=2e-14, atol=1e-14)

    def test_one_spin_rhs_sigma_has_plus_sine_term(self):
        time = 0.4
        t1 = np.array([0.0, time])
        c_hist = np.asarray([num.c_coef_all(value) for value in t1])
        c_now = c_hist[-1]
        sigma = np.zeros((1, 3))
        actual = num.RHS_sigma(
            sigma, num.N0, time, t1, c_hist, c_now, "ohmic"
        )
        weights = num.trapezoid_weights(t1)
        _, sine_integral = num.correlation_parts(time - t1, num.N0, time, "ohmic")
        integrated = np.einsum("h,h,hsl->sl", weights, sine_integral, c_hist)
        manual = 4.0 * pm.LAMBDA**2 * num.ETA2_j[:, None] * np.einsum(
            "kal,pk,pl->pa", num.eps, c_now, integrated
        )
        np.testing.assert_allclose(actual, manual, rtol=2e-14, atol=1e-14)

    def test_rhs_supports_multiple_spins(self):
        time = 0.4
        t1 = np.array([0.0, time])
        one_spin_history = np.asarray([num.c_coef_all(value)[0] for value in t1])
        c_history = np.stack((one_spin_history, one_spin_history), axis=1)
        c_now = c_history[-1]
        sigma = np.array([[0.2, -0.3, 0.4], [-0.1, 0.25, 0.35]])
        occupation = np.full_like(num.omega_arr, 0.15)
        eta = np.array([1.0, 0.7])
        with patch.object(num, "ETA_j", eta), patch.object(num, "ETA2_j", eta**2):
            rhs_sigma = num.RHS_sigma(
                sigma, occupation, time, t1, c_history, c_now, "ohmic"
            )
            rhs_n = num.RHS_N(
                occupation, sigma, time, t1, c_history, c_now, "ohmic"
            )
        self.assertEqual(rhs_sigma.shape, (2, 3))
        self.assertEqual(rhs_n.shape, num.omega_arr.shape)
        self.assertTrue(np.all(np.isfinite(rhs_sigma)))
        self.assertTrue(np.all(np.isfinite(rhs_n)))

    def test_correlation_at_zero_has_no_imaginary_part(self):
        value = num.C_corr(0.0, num.N0)
        self.assertAlmostEqual(value.imag, 0.0, places=14)
        self.assertGreater(value.real, 0.0)

    def test_stationary_correlation_has_conjugate_symmetry(self):
        for delay in (0.1, 0.7, 3.0):
            positive = num.C_corr(delay, num.N0)
            negative = num.C_corr(-delay, num.N0)
            self.assertAlmostEqual(negative.real, positive.real, places=12)
            self.assertAlmostEqual(negative.imag, -positive.imag, places=12)

    def test_kernel_lookup_matches_direct_quadrature(self):
        delays = np.array([0.123, 0.707, 3.141, 9.876, 19.999])
        real_part, sine_part = num.correlation_parts(delays, num.N0, 0.0, "ohmic")
        direct = np.asarray([num.C_corr(delay, num.N0) for delay in delays])
        np.testing.assert_allclose(real_part, direct.real, rtol=2.0e-4, atol=2.0e-4)
        np.testing.assert_allclose(sine_part, -direct.imag, rtol=2.0e-4, atol=2.0e-4)

    def test_frozen_step_preserves_occupation(self):
        diagnostics = num.SolverDiagnostics(
            spectral_density="ohmic",
            occupation_mode="frozen",
        )
        _, occupation = num.trbdf2_coupled_step(
            num.BETA0.copy(),
            num.N0.copy(),
            step=0,
            density="ohmic",
            diagnostics=diagnostics,
            evolve_occupations=False,
        )
        np.testing.assert_array_equal(occupation, num.N0)

    @unittest.skipUnless(importlib.util.find_spec("qutip"), "qutip is not installed")
    def test_heom_default_run_is_physical(self):
        import heom

        result = heom.solve_heom()
        self.assertGreater(abs(result.terminator_delta), 0.0)
        self.assertLessEqual(float(np.max(np.linalg.norm(result.sigma, axis=1))), 1.0 + 1.0e-7)


if __name__ == "__main__":
    unittest.main()
