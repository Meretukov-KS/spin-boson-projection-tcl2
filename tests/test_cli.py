"""Short end-to-end runs, including output files and both HEOM modes."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
from config import get_config, load_config, set_active_config
from spectral_density import evaluate, time_factor


def small_config():
    return (CODE_DIR / "config.toml").read_text(encoding="utf-8").replace(
        "maximum = 20.0", "maximum = 0.2"
    ).replace("points = 801", "points = 9").replace(
        "quadrature_points = 5200", "quadrature_points = 160"
    ).replace("physical_grid_points = 200", "physical_grid_points = 20")


class ConfigurationTests(unittest.TestCase):
    def test_internal_settings_are_not_configuration_fields(self):
        config = get_config()
        for name in ("N_DEPTH", "TRBDF2_ATOL", "TRBDF2_MAX_ITER", "ADAPTIVE_STRENGTH"):
            self.assertFalse(hasattr(config, name))

    def test_legacy_solver_section_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(small_config() + "\n[solver]\nabsolute_tolerance = 0.1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "obsolete"):
                load_config(path)

    def test_comparison_selects_drude_before_solver_import(self):
        import config
        with patch.object(config, "_active", None):
            actual = set_active_config(comparison=True)
            self.assertEqual(actual.SPECTRAL_DENSITY, "drude")
            self.assertNotIn("xi", actual.spectral_parameters())

    def test_comparison_rejects_nonthermal_or_modulated_bath(self):
        for source in (
            small_config().replace("excess_amplitude = 0.0", "excess_amplitude = 0.1"),
            small_config().replace("[spectral_parameters.drude]", "[spectral_parameters.drude]\ntime_amplitude = 0.1"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.toml"
                path.write_text(source, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "thermal initial bath"):
                    set_active_config(path, comparison=True)

    def test_custom_has_no_unverified_separability_assumption(self):
        self.assertIsNone(time_factor("custom", 0.1, {}))

    def test_spectral_sections_are_independent(self):
        source = small_config().replace("wc = 7.5", "wc = 3.0", 1)
        source = source.replace("drude_gamma = 7.5", "drude_gamma = 4.5")
        source = source.replace("[spectral_parameters.ohmic]", "[spectral_parameters.ohmic]\ntime_amplitude = 0.2")
        import config
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "_active", None):
            path = Path(directory) / "config.toml"
            path.write_text(source, encoding="utf-8")
            actual = load_config(path)
            self.assertEqual(actual.spectral_parameters("ohmic")["wc"], 3.0)
            self.assertEqual(actual.spectral_parameters("ohmic_lorentz")["wc"], 7.5)
            self.assertEqual(actual.spectral_parameters("custom")["cutoff"], 7.5)
            compared = set_active_config(path, comparison=True)
            self.assertEqual(compared.spectral_parameters()["drude_gamma"], 4.5)
            self.assertEqual(compared.spectral_parameters()["time_amplitude"], 0.0)

    def test_flat_spectral_section_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(small_config().replace("[spectral_parameters.ohmic]", "[spectral_parameters]"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "named sections"):
                load_config(path)

    def test_omitted_initial_section_is_thermal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(small_config().split("[initial_occupation]")[0], encoding="utf-8")
            self.assertEqual(load_config(path).Delta_N, 0.0)

    def test_complex_and_negative_density_are_rejected(self):
        import spectral_density as sd
        for result, message in ((1 + 1j, "real-valued"), (-1.0, "negative")):
            with patch.dict(sd.SPECTRAL_DENSITIES, custom=lambda w, t, p: result):
                with self.assertRaisesRegex(ValueError, message):
                    evaluate("custom", np.array([1.0, 2.0]), 0.0, {})

    @unittest.skipUnless(importlib.util.find_spec("qutip"), "qutip is not installed")
    def test_heom_includes_individual_spin_coupling(self):
        import heom
        config = get_config()
        spin = replace(config.SPINS[0], COUPLING=0.7)
        first = replace(config, SPINS=(spin,), T_MAX=0.2, T_POINTS=9)
        second = replace(first, SPINS=(replace(spin, COUPLING=1.0),), LAMBDA=config.LAMBDA * 0.7)
        with patch.object(heom, "pm", first):
            a = heom.solve_heom()
        with patch.object(heom, "pm", second):
            b = heom.solve_heom()
        np.testing.assert_allclose(a.sigma, b.sigma, rtol=1e-9, atol=1e-10)


class CLITests(unittest.TestCase):
    def run_case(self, mode, source):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "input.toml"
            config_path.write_text(source, encoding="utf-8")
            script = f"from pathlib import Path; import run; run.CODE_DIR=Path({str(root)!r}); run.main()"
            result = subprocess.run(
                [sys.executable, "-B", "-c", script, mode, "--config", str(config_path)],
                cwd=CODE_DIR, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = next((root / mode).iterdir())
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertNotIn(str(root), json.dumps(metadata))
            self.assertGreater(metadata["timing_seconds"]["tcl2_solver"], 0)
            self.assertTrue((output / "summary.txt").exists())
            self.assertTrue((output / "config.toml").exists())
            self.assertTrue((output / "bath_initial.csv").exists())
            self.assertGreater(next(output.glob("*.png")).stat().st_size, 10000)
            with np.load(output / "bath_data.npz") as bath:
                np.testing.assert_allclose(bath["spectral_density"], bath["mode_density"] * bath["coupling_squared"])
                name = "solution.npz" if mode == "numeric" else "compare_solution.npz"
                with np.load(output / name) as data:
                    self.assertTrue(np.all(np.isfinite(data["occupations"])))
                    if mode == "compare-frozen":
                        np.testing.assert_array_equal(data["occupations"], np.broadcast_to(bath["initial_occupation"], data["occupations"].shape))
                    else:
                        self.assertGreater(np.max(np.abs(data["occupations"][-1] - bath["initial_occupation"])), 0)
                    if mode == "numeric":
                        self.assertEqual(data["sigma"].shape[1], source.count("[[spins]]"))
            if mode != "numeric":
                self.assertEqual(metadata["spectral_density"]["name"], "drude")
                self.assertGreater(metadata["timing_seconds"]["heom_solver"], 0)

    def test_numeric(self):
        self.run_case("numeric", small_config())

    def test_custom_modulated_nonthermal_two_spin(self):
        source = small_config().replace('spectral_density = "ohmic"', 'spectral_density = "custom"')
        source = source.replace("[spectral_parameters.custom]", "[spectral_parameters.custom]\ntime_amplitude = 0.2\ntime_frequency = 0.01")
        source = source.replace("excess_amplitude = 0.0", "excess_amplitude = 0.1")
        source += '\n[[spins]]\ndelta=2.3\nfield=0.1\ncoupling=0.7\ninitial_bloch=[0.0,0.0,1.0]\n'
        self.run_case("numeric", source)

    @unittest.skipUnless(importlib.util.find_spec("qutip"), "qutip is not installed")
    def test_compare(self):
        self.run_case("compare", small_config())

    @unittest.skipUnless(importlib.util.find_spec("qutip"), "qutip is not installed")
    def test_compare_frozen(self):
        self.run_case("compare-frozen", small_config())
