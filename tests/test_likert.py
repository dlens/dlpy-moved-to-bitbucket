from unittest import TestCase
import dlpy.likert as lk
import numpy.testing as npt


class TestStandardLikert(TestCase):
    def test_standards(self):
        a = lk.StandardLikert.H
        self.assertEqual(str(a), "Very High")
        self.assertEqual(a.value(), 1.0)
        self.assertEqual(lk.likert_from_01_grade(0.1), lk.StandardLikert.L)
        self.assertEqual(lk.likert_from_01_grade(0.2), lk.StandardLikert.l)
        self.assertEqual(lk.likert_from_01_grade(0.5), lk.StandardLikert.m)
        self.assertEqual(lk.likert_from_01_grade(0.7), lk.StandardLikert.h)
        self.assertEqual(lk.likert_from_01_grade(0.9), lk.StandardLikert.H)


L = lk.StandardLikert.L
l = lk.StandardLikert.l
m = lk.StandardLikert.m
h = lk.StandardLikert.h
H = lk.StandardLikert.H


class Test(TestCase):
    def test_likert_using_zscores(self):
        values = [1, 2, 3, 4, 5]
        self.assertEqual(lk.likert_using_zscores(values), [L, l, m, h, H])
        self.assertEqual(lk.likert_using_zscores([1, 3, 2, 5, 4]), [L, m, l, H, h])

    def test_likert_using_small_count(self):
        self.assertEqual(lk.likert_using_small_count([1, 2, 3]), [l, m, h])
        self.assertEqual(lk.likert_using_small_count([3, 2, 1]), [h, m, l])
        self.assertEqual(lk.likert_using_small_count([1, 2, 3, 3]), [l, m, h, h])
        self.assertEqual(lk.likert_using_small_count([3, 3, 3, 1, 1, 1, 2, 2, 2, 4, 4, 5]),
                         [m, m, m, L, L, L, l, l, l, h, h, H])

    def test_cluster_simple(self):
        values = [1, 0.99999, 1.00001, 1.5, 2, 1.99999, 2.00001]
        clusters = lk.cluster_simple(values, 0.001, 0.1)
        means = [cluster.center() for cluster in clusters]
        npt.assert_almost_equal(means, [1, 1.5, 2])

    def test_cluster_means_indexed(self):
        values = [1, 0.99999, 1.00001, 1.5, 2, 1.99999, 2.00001]
        clusters = lk.cluster_simple(values, 0.001, 0.1)
        centers = lk.ClusterOfNumber.clustered_values_from_clusters(clusters)
        npt.assert_almost_equal(centers, [1, 1, 1, 1.5, 2, 2, 2])

    def test_small_count_clusters(self):
        values = [0.9999, 0.9989, 1.0003, 0, 0.00001, -0.001, 1, 2, 1.001]
        likerts = lk.likert_using_small_count(values)
        self.assertEqual(likerts, [m, m, m, l, l, l, m, h, m])

    def test_zscore_likert_with_clusters(self):
        values = [1.0001, 1.002, 1.003, 1.004, 2, 3, 3.001, 3.002]
        likerts = lk.likert_using_zscores(values, do_cluster=True)
        self.assertEqual(likerts, [L, L, L, L, m, H, H, H])

    def test_likert_using_percentile(self):
        values = [1.0001, 1.002, 1.003, 1.004, 2, 3, 3.001, 3.002]
        likerts = lk.likert_using_percentile(values, do_cluster=True)
        self.assertEqual(likerts, [L, L, L, L, m, H, H, H])

    def test_clustering(self):
        values = [101, 102, 103, 104, 105, 106, 107, 108, 110, 112, 150, 151, 100]
        vs = lk.cluster_values_simple(values)
        expected = [101, 102, 103, 104, 105, 106, 107, 108, 110, 112, 150.5, 150.5, 100]
        npt.assert_almost_equal(vs, expected)