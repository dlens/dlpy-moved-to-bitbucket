from unittest import TestCase
import numpy.testing as npt
import dlpy.percentile as dlpr


class Test(TestCase):
    def test_std_perc(self):
        X = (10, 20, 30, 40, 50)
        npt.assert_almost_equal(dlpr.std_perc(X, 0), 0)
        npt.assert_almost_equal(dlpr.std_perc(X, 10), 0.10)
        npt.assert_almost_equal(dlpr.std_perc(X, 15), 0.20)
        npt.assert_almost_equal(dlpr.std_perc(X, 20), 0.30)
        npt.assert_almost_equal(dlpr.std_perc(X, 30), 0.50)
        npt.assert_almost_equal(dlpr.std_perc(X, 40), 0.70)
        npt.assert_almost_equal(dlpr.std_perc(X, 50), 0.90)
        npt.assert_almost_equal(dlpr.std_perc(X, 150), 1.00)

    def test_gcp_sorted_deduped(self):
        X = (10, 20, 30, 40, 50)
        epsilon = 0.01
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped(X, 10, epsilon), epsilon)
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped(X, 20), epsilon, 0.25)
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped(X, 25, epsilon), 0.375)
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped(X, 30, epsilon), 0.50)
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped(X, 35, epsilon), 0.625)
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped(X, 40, epsilon), 0.75)
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped(X, 45, epsilon), (0.75 + 1 - epsilon) / 2)
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped(X, 50, epsilon), 1 - epsilon)
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped(X, 55, epsilon), 0.9992307692307693)
        (pts, LHS, RHS) = dlpr.gcp(X, 0, return_params=True)
        self.assertEqual(
            pts, [(10, epsilon), (20, 0.25), (30, 0.50), (40, 0.75), (50, 1 - epsilon)]
        )
        npt.assert_array_almost_equal(RHS, (-0.0041666666666666, 49.58333333333333, 1))
        npt.assert_array_almost_equal(LHS, (-0.0041666666666666, 10.41666666666666, 0))

    def test_sort_dedupe(self):
        X = [50, 30, 20, 10, 40, 20, 50, 10]
        Y = dlpr.sort_dedupe(X)
        npt.assert_array_equal(Y, [10, 20, 30, 40, 50])

    def test_gcp_linear(self):
        X = [50, 30, 20, 10, 40, 20, 50, 10]
        epsilon = 0.01
        npt.assert_almost_equal(dlpr.gcp(X, 10, epsilon), epsilon)
        npt.assert_almost_equal(dlpr.gcp(X, 20, epsilon), epsilon, 0.25)
        npt.assert_almost_equal(dlpr.gcp(X, 25, epsilon), 0.375)
        npt.assert_almost_equal(dlpr.gcp(X, 30, epsilon), 0.50)
        npt.assert_almost_equal(dlpr.gcp(X, 35, epsilon), 0.625)
        npt.assert_almost_equal(dlpr.gcp(X, 40, epsilon), 0.75)
        npt.assert_almost_equal(dlpr.gcp(X, 45, epsilon), (0.75 + 1 - epsilon) / 2)
        npt.assert_almost_equal(dlpr.gcp(X, 50, epsilon), 1 - epsilon)
        npt.assert_almost_equal(dlpr.gcp(X, 55, epsilon), 0.9992307692307693)

    def test_gcp_exponential(self):
        X = [50, 30, 20, 10, 40, 20, 50, 10]
        epsilon = 0.01
        npt.assert_almost_equal(dlpr.gcp(X, 10, epsilon, decay_type=dlpr.DecayType.EXPONENTIAL), epsilon)
        npt.assert_almost_equal(dlpr.gcp(X, 20, epsilon, decay_type=dlpr.DecayType.EXPONENTIAL), epsilon, 0.25)
        npt.assert_almost_equal(dlpr.gcp(X, 25, epsilon, decay_type=dlpr.DecayType.EXPONENTIAL), 0.375)
        npt.assert_almost_equal(dlpr.gcp(X, 30, epsilon, decay_type=dlpr.DecayType.EXPONENTIAL), 0.50)
        npt.assert_almost_equal(dlpr.gcp(X, 35, epsilon, decay_type=dlpr.DecayType.EXPONENTIAL), 0.625)
        npt.assert_almost_equal(dlpr.gcp(X, 40, epsilon, decay_type=dlpr.DecayType.EXPONENTIAL), 0.75)
        npt.assert_almost_equal(dlpr.gcp(X, 45, epsilon, decay_type=dlpr.DecayType.EXPONENTIAL),
                                (0.75 + 1 - epsilon) / 2)
        npt.assert_almost_equal(dlpr.gcp(X, 50, epsilon, decay_type=dlpr.DecayType.EXPONENTIAL), 1 - epsilon)
        npt.assert_almost_equal(dlpr.gcp(X, 55, epsilon, decay_type=dlpr.DecayType.EXPONENTIAL), 0.9999999385578765)

    def test_gcp_sorted_deduped_inverse(self):
        X = [10, 20, 30, 40, 50]
        epsilon = 0.01
        # Whole thing is just inverse of tests above
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped_inverse(X, epsilon, epsilon), 10)
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped_inverse(X, 0.25, epsilon), 20)
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped_inverse(X, 0.375, epsilon), 25)
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped_inverse(X, 0.5, epsilon), 30)
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped_inverse(X, 0.625, epsilon), 35)
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped_inverse(X, 0.75, epsilon), 40)
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped_inverse(X, (0.75 + 1 - epsilon) / 2, epsilon), 45)
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped_inverse(X, 1 - epsilon, epsilon), 50)
        npt.assert_almost_equal(dlpr.gcp_sorted_deduped_inverse(X, 0.9992307692307693, epsilon), 55)
        npt.assert_almost_equal(
            dlpr.gcp_sorted_deduped_inverse(X, 0.9999999385578765, epsilon, decay_type=dlpr.DecayType.EXPONENTIAL), 55)

    def test_gcp_inverse(self):
        # Just copy of test above
        X = [50, 40, 20, 30, 10, 50, 10, 20, 30, 40, 50]
        epsilon = 0.01
        # Whole thing is just inverse of tests above
        npt.assert_almost_equal(dlpr.gcp_inverse(X, epsilon, epsilon), 10)
        npt.assert_almost_equal(dlpr.gcp_inverse(X, 0.25, epsilon), 20)
        npt.assert_almost_equal(dlpr.gcp_inverse(X, 0.375, epsilon), 25)
        npt.assert_almost_equal(dlpr.gcp_inverse(X, 0.5, epsilon), 30)
        npt.assert_almost_equal(dlpr.gcp_inverse(X, 0.625, epsilon), 35)
        npt.assert_almost_equal(dlpr.gcp_inverse(X, 0.75, epsilon), 40)
        npt.assert_almost_equal(dlpr.gcp_inverse(X, (0.75 + 1 - epsilon) / 2, epsilon), 45)
        npt.assert_almost_equal(dlpr.gcp_inverse(X, 1 - epsilon, epsilon), 50)
        npt.assert_almost_equal(dlpr.gcp_inverse(X, 0.9992307692307693, epsilon), 55)
        npt.assert_almost_equal(dlpr.gcp_inverse(X, 0.9999999385578765, epsilon, decay_type=dlpr.DecayType.EXPONENTIAL),
                                55)

    def test_gcp_approx_pts(self):
        X = [50, 40, 20, 30, 10, 50, 10, 20, 30, 40, 50]
        epsilon = 0.01
        pts = dlpr.gcp_approx_pts(X, epsilon)
        npt.assert_allclose(
            pts,
            [(10.0, 0.01), (17.916666666666668, 0.2), (26.0, 0.4), (30.0, 0.5), (34.0, 0.6), (42.083333333333336, 0.8), (50.0, 0.99)]
        )