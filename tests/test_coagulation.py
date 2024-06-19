import unittest

import torch

from core.coagulation import (
    k_bleed,
    rbc_prime,
    inactive_coag_factor_prime,
    active_coag_factor_prime,
    pro_coag_total,
    inactive_anti_coag_prime,
    active_anti_coag_prime,
    anti_coag_total,
    platelets_prime,
    platelet_source_enhance,
    clot_prime,
)


class TestCoagulation(unittest.TestCase):

    def setUp(self):
        """Set up common test parameters."""
        self.config = {
            "HEMODYNAMICS": {
                "S_BLOOD_VOLUME": 5000,
                "S_BLOOD_PRESSURE": 65,
            },
            "COAGULATION": {
                "S_RBC": 5.0e9,
                "K_BASELINE_RBC": 4.01127e-6,
                "S_INACTIVE_COAG_FACTOR": 1.0e8,
                "S_INACTIVE_ANTI_COAG": 4.0e6,
                "S_ACTIVE_ANTI_COAG": 1.0e3,
                "S_PLATELETS": 3,
                "K_BASELINE_PLATELETS": 0.000120338,
                "RBC_INFUSION_BONUS_CONC": 1.44,
                "PLASMA_INFUSION_COAG_BONUS": 1.55,
                "PLATELET_INFUSION_BONUS_CONC": 3.67,
                "D_ACTIVE_COAG_FACTOR": 0.693314718,
                "D_CLOT": 1.0e-5,
                "K_ACTIVE_COAG_FACTOR_ACTIVE_ANTI_COAG_ENHANCE": 0.001,
                "K_ACTIVE_COAG_FACTOR_ACTIVE_ANTI_COAG_COLLIDE": 0.001,
                "K_PLATELET_SOURCE_ENHANCE": 0.001,
                "K_COAG_FACTOR_ACTIVATION_TRAUMA": 1.0e-7,
                "K_INACTIVE_ANTI_COAG_ACTIVE_COAG_FACTOR": 1.0e-7,
                "K_CLOT": 1.0e-9,
                "K_CLOT_RBC": 1.0e-7,
                "K_BLEED_TRAUMA": 12,
                "K_BLEED_CONTROL": 5.0e-4,
                "K_FIBRINOLYSIS": 0.001,
                "ACTIVECOAGFACTOR_PORTION_OF_CLOT_CONVERTER": 1,
                "PLATELETS_PORTION_OF_CLOT_CONVERTER": 1,
            },
            "INFLAMMATION": {
                "MAX_ISS": 75,
                "K_COAG_FACTOR_ACTIVATION_IL6": 1.0e-7,
            },
        }

    def test_k_bleed(self):
        """Test the k_bleed function."""
        test_cases = [
            # (_trauma, clot, blood_pressure, expected_output)
            (
                torch.tensor(20.0),
                torch.tensor(5.0),
                torch.tensor(60.0),
                torch.tensor(0.46153846),
            ),
            (
                torch.tensor(10.0),
                torch.tensor(15.0),
                torch.tensor(70.0),
                torch.tensor(0.0),
            ),
        ]
        for case in test_cases:
            trauma, clot, blood_pressure, expected_output = case
            output = k_bleed(trauma, clot, blood_pressure, self.config)
            self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_rbc_prime(self):
        """Test the rbc_prime function."""
        test_cases = [
            # (rbc, blood_volume, sum_rbc_infusions, k_bleed, d_bv_dt, expected_output)
            (
                torch.tensor(4.5e9),
                torch.tensor(4800.0),
                torch.tensor(100.0),
                torch.tensor(0.05),
                torch.tensor(10.0),
                torch.tensor(192.30769231),
            ),
            (
                torch.tensor(5.5e9),
                torch.tensor(5200.0),
                torch.tensor(0.0),
                torch.tensor(0.1),
                torch.tensor(-20.0),
                torch.tensor(-1057.69230769),
            ),
        ]
        for case in test_cases:
            rbc, blood_volume, sum_rbc_infusions, k_bleed, d_bv_dt, expected_output = (
                case
            )
            output = rbc_prime(
                rbc, blood_volume, sum_rbc_infusions, k_bleed, d_bv_dt, self.config
            )
            self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_inactive_coag_factor_prime(self):
        """Test the inactive_coag_factor_prime function."""
        test_cases = [
            # (inactive_coag_factor, blood_volume, sum_plasma_infusions, sum_platelet_infusions, k_bleed, _trauma, il6, d_bv_dt, expected_output)
            (
                torch.tensor(9e7),
                torch.tensor(4900.0),
                torch.tensor(50.0),
                torch.tensor(25.0),
                torch.tensor(0.04),
                torch.tensor(15.0),
                torch.tensor(100.0),
                torch.tensor(5.0),
                torch.tensor(118.57142857),
            ),
            (
                torch.tensor(1.1e8),
                torch.tensor(5100.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.08),
                torch.tensor(25.0),
                torch.tensor(200.0),
                torch.tensor(-10.0),
                torch.tensor(-2637.25490196),
            ),
        ]
        for case in test_cases:
            (
                inactive_coag_factor,
                blood_volume,
                sum_plasma_infusions,
                sum_platelet_infusions,
                k_bleed,
                trauma,
                il6,
                d_bv_dt,
                expected_output,
            ) = case
            output = inactive_coag_factor_prime(
                inactive_coag_factor,
                blood_volume,
                sum_plasma_infusions,
                sum_platelet_infusions,
                k_bleed,
                trauma,
                il6,
                d_bv_dt,
                self.config,
            )
            self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_active_coag_factor_prime(self):
        """Test the active_coag_factor_prime function."""
        test_cases = [
            # (inactive_coag_factor, blood_volume, _trauma, il6, k_bleed, active_anti_coag, inactive_anti_coag, platelets, active_coag_factor, d_bv_dt, expected_output)
            (
                torch.tensor(1e8),
                torch.tensor(5000.0),
                torch.tensor(20.0),
                torch.tensor(150.0),
                torch.tensor(0.06),
                torch.tensor(800.0),
                torch.tensor(3.8e6),
                torch.tensor(2.5),
                torch.tensor(1000.0),
                torch.tensor(8.0),
                torch.tensor(-69314.718),
            ),
            (
                torch.tensor(9e7),
                torch.tensor(4800.0),
                torch.tensor(10.0),
                torch.tensor(100.0),
                torch.tensor(0.02),
                torch.tensor(1200.0),
                torch.tensor(4.2e6),
                torch.tensor(3.0),
                torch.tensor(1500.0),
                torch.tensor(-5.0),
                torch.tensor(-103972.077),
            ),
        ]
        for case in test_cases:
            (
                inactive_coag_factor,
                blood_volume,
                trauma,
                il6,
                k_bleed,
                active_anti_coag,
                inactive_anti_coag,
                platelets,
                active_coag_factor,
                d_bv_dt,
                expected_output,
            ) = case
            output = active_coag_factor_prime(
                inactive_coag_factor,
                blood_volume,
                trauma,
                il6,
                k_bleed,
                active_anti_coag,
                inactive_anti_coag,
                platelets,
                active_coag_factor,
                d_bv_dt,
                self.config,
            )
            self.assertTrue(torch.allclose(output, expected_output, atol=1e-3))

    def test_pro_coag_total(self):
        """Test the pro_coag_total function."""
        test_cases = [
            # (inactive_coag_factor, active_coag_factor, expected_output)
            (torch.tensor(9e7), torch.tensor(1e7), torch.tensor(1e8)),
            (torch.tensor(1.1e8), torch.tensor(1.5e7), torch.tensor(1.25e8)),
        ]
        for case in test_cases:
            inactive_coag_factor, active_coag_factor, expected_output = case
            output = pro_coag_total(inactive_coag_factor, active_coag_factor)
            self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_inactive_anti_coag_prime(self):
        """Test the inactive_anti_coag_prime function."""
        test_cases = [
            # (inactive_anti_coag, blood_volume, sum_plasma_infusions, sum_platelet_infusions, k_bleed, active_coag_factor, d_bv_dt, expected_output)
            (
                torch.tensor(3.9e6),
                torch.tensor(5100.0),
                torch.tensor(100.0),
                torch.tensor(50.0),
                torch.tensor(0.03),
                torch.tensor(1200.0),
                torch.tensor(-15.0),
                torch.tensor(139.80392157),
            ),
            (
                torch.tensor(4.1e6),
                torch.tensor(4900.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.07),
                torch.tensor(800.0),
                torch.tensor(10.0),
                torch.tensor(-2893.61702128),
            ),
        ]
        for case in test_cases:
            (
                inactive_anti_coag,
                blood_volume,
                sum_plasma_infusions,
                sum_platelet_infusions,
                k_bleed,
                active_coag_factor,
                d_bv_dt,
                expected_output,
            ) = case
            output = inactive_anti_coag_prime(
                inactive_anti_coag,
                blood_volume,
                sum_plasma_infusions,
                sum_platelet_infusions,
                k_bleed,
                active_coag_factor,
                d_bv_dt,
                self.config,
            )
            self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_active_anti_coag_prime(self):
        """Test the active_anti_coag_prime function."""
        test_cases = [
            # (active_anti_coag, inactive_anti_coag, blood_volume, active_coag_factor, sum_plasma_infusions, sum_platelet_infusions, k_bleed, d_bv_dt, expected_output)
            (
                torch.tensor(900.0),
                torch.tensor(4e6),
                torch.tensor(5000.0),
                torch.tensor(1000.0),
                torch.tensor(50.0),
                torch.tensor(25.0),
                torch.tensor(0.05),
                torch.tensor(10.0),
                torch.tensor(227.5),
            ),
            (
                torch.tensor(1100.0),
                torch.tensor(3.8e6),
                torch.tensor(4800.0),
                torch.tensor(1500.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.09),
                torch.tensor(-12.0),
                torch.tensor(-292.6),
            ),
        ]
        for case in test_cases:
            (
                active_anti_coag,
                inactive_anti_coag,
                blood_volume,
                active_coag_factor,
                sum_plasma_infusions,
                sum_platelet_infusions,
                k_bleed,
                d_bv_dt,
                expected_output,
            ) = case
            output = active_anti_coag_prime(
                active_anti_coag,
                inactive_anti_coag,
                blood_volume,
                active_coag_factor,
                sum_plasma_infusions,
                sum_platelet_infusions,
                k_bleed,
                d_bv_dt,
                self.config,
            )
            self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_anti_coag_total(self):
        """Test the anti_coag_total function."""
        test_cases = [
            # (inactive_anti_coag, active_anti_coag, expected_output)
            (torch.tensor(3.9e6), torch.tensor(900.0), torch.tensor(3.9009e6)),
            (torch.tensor(4.1e6), torch.tensor(1100.0), torch.tensor(4.1011e6)),
        ]
        for case in test_cases:
            inactive_anti_coag, active_anti_coag, expected_output = case
            output = anti_coag_total(inactive_anti_coag, active_anti_coag)
            self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_platelets_prime(self):
        """Test the platelets_prime function."""
        test_cases = [
            # (platelets, blood_volume, sum_platelet_infusions, platelet_source_enhance, active_coag_factor, k_bleed, d_bv_dt, expected_output)
            (
                torch.tensor(2.8),
                torch.tensor(4900.0),
                torch.tensor(75.0),
                torch.tensor(0.2),
                torch.tensor(950.0),
                torch.tensor(0.04),
                torch.tensor(6.0),
                torch.tensor(1.07755102),
            ),
            (
                torch.tensor(3.2),
                torch.tensor(5100.0),
                torch.tensor(0.0),
                torch.tensor(0.1),
                torch.tensor(1050.0),
                torch.tensor(0.08),
                torch.tensor(-8.0),
                torch.tensor(-0.39607843),
            ),
        ]
        for case in test_cases:
            (
                platelets,
                blood_volume,
                sum_platelet_infusions,
                platelet_source_enhance,
                active_coag_factor,
                k_bleed,
                d_bv_dt,
                expected_output,
            ) = case
            output = platelets_prime(
                platelets,
                blood_volume,
                sum_platelet_infusions,
                platelet_source_enhance,
                active_coag_factor,
                k_bleed,
                d_bv_dt,
                self.config,
            )
            self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_platelet_source_enhance(self):
        """Test the platelet_source_enhance function."""
        test_cases = [
            # (_trauma, expected_output)
            (torch.tensor(30.0), torch.tensor(0.16)),
            (torch.tensor(15.0), torch.tensor(0.04)),
        ]
        for case in test_cases:
            trauma, expected_output = case
            output = platelet_source_enhance(trauma, self.config)
            print(output, expected_output)
            self.assertTrue(torch.allclose(output, expected_output, atol=1e-1))

    def test_clot_prime(self):
        """Test the clot_prime function."""
        test_cases = [
            # (platelets, active_coag_factor, rbc, _trauma, blood_volume, clot, expected_output)
            (
                torch.tensor(3.0),
                torch.tensor(1000.0),
                torch.tensor(5e9),
                torch.tensor(25.0),
                torch.tensor(5000.0),
                torch.tensor(8.0),
                torch.tensor(14.9976),
            ),
            (
                torch.tensor(2.5),
                torch.tensor(800.0),
                torch.tensor(4.8e9),
                torch.tensor(15.0),
                torch.tensor(4800.0),
                torch.tensor(5.0),
                torch.tensor(9.5984),
            ),
        ]
        for case in test_cases:
            (
                platelets,
                active_coag_factor,
                rbc,
                trauma,
                blood_volume,
                clot,
                expected_output,
            ) = case
            output = clot_prime(
                platelets,
                active_coag_factor,
                rbc,
                trauma,
                blood_volume,
                clot,
                self.config,
            )
            self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
