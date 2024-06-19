import unittest

import torch

from core.hemodynamics import (
    blood_volume_prime,
    d_bv_dt,
    blood_pressure_unadjusted_prime,
    blood_pressure,
)


class TestHemodynamics(unittest.TestCase):

    def setUp(self):
        """Set up common test parameters."""
        self.config = {
            "HEMODYNAMICS": {
                "S_BLOOD_VOLUME": 5000,
                "S_BLOOD_PRESSURE": 65,
                "K_BASELINE_BLOOD_PRESSURE": 0.001,
                "K_BLOOD_PRESSURE_NO": 1.0e-07,
                "K_URINATE": 0.01,
            },
            "INFLAMMATION": {
                "S_NO": 1e-6,  # Example value, adjust as needed
            },
        }

    def test_blood_volume_prime(self):
        """Test the blood_volume_prime function."""
        test_cases = [
            # (sum_plasma_infusions, sum_rbc_infusions, sum_platelet_infusions, sum_fluid_infusions, k_bleed, blood_volume, expected_output)
            (
                torch.tensor(100.0),
                torch.tensor(50.0),
                torch.tensor(25.0),
                torch.tensor(200.0),
                torch.tensor(0.05),
                torch.tensor(4500.0),
                torch.tensor(270.0),
            ),
            (
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.1),
                torch.tensor(5000.0),
                torch.tensor(-500.0),
            ),
        ]
        for case in test_cases:
            (
                sum_plasma_infusions,
                sum_rbc_infusions,
                sum_platelet_infusions,
                sum_fluid_infusions,
                k_bleed,
                blood_volume,
                expected_output,
            ) = case
            output = blood_volume_prime(
                sum_plasma_infusions,
                sum_rbc_infusions,
                sum_platelet_infusions,
                sum_fluid_infusions,
                k_bleed,
                blood_volume,
                self.config,
            )
            self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_d_bv_dt(self):
        """Test the d_bv_dt function (alias for blood_volume_prime)."""
        # Since d_bv_dt is an alias, we can reuse the test cases from blood_volume_prime.
        test_cases = [
            # (sum_plasma_infusions, sum_rbc_infusions, sum_platelet_infusions, sum_fluid_infusions, k_bleed, blood_volume, expected_output)
            (
                torch.tensor(100.0),
                torch.tensor(50.0),
                torch.tensor(25.0),
                torch.tensor(200.0),
                torch.tensor(0.05),
                torch.tensor(4500.0),
                torch.tensor(270.0),
            ),
            (
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.1),
                torch.tensor(5000.0),
                torch.tensor(-500.0),
            ),
        ]
        for case in test_cases:
            (
                sum_plasma_infusions,
                sum_rbc_infusions,
                sum_platelet_infusions,
                sum_fluid_infusions,
                k_bleed,
                blood_volume,
                expected_output,
            ) = case
            output = d_bv_dt(
                sum_plasma_infusions,
                sum_rbc_infusions,
                sum_platelet_infusions,
                sum_fluid_infusions,
                k_bleed,
                blood_volume,
                self.config,
            )
            self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_blood_pressure_unadjusted_prime(self):
        """Test the blood_pressure_unadjusted_prime function."""
        test_cases = [
            # (_blood_pressure, no, expected_output)
            (torch.tensor(60.0), torch.tensor(1e-7), torch.tensor(5.000001)),
            (torch.tensor(70.0), torch.tensor(5e-7), torch.tensor(-4.999995)),
        ]
        for case in test_cases:
            _blood_pressure, no, expected_output = case
            output = blood_pressure_unadjusted_prime(_blood_pressure, no, self.config)
            self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_blood_pressure(self):
        """Test the blood_pressure function."""
        test_cases = [
            # (blood_pressure_unadjusted, blood_volume, expected_output)
            (torch.tensor(65.0), torch.tensor(5000.0), torch.tensor(65.0)),
            (torch.tensor(60.0), torch.tensor(4000.0), torch.tensor(41.6)),
        ]
        for case in test_cases:
            blood_pressure_unadjusted, blood_volume, expected_output = case
            output = blood_pressure(
                blood_pressure_unadjusted, blood_volume, self.config
            )
            self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
