from unittest import TestCase

from dlpy.ap.scoring import RankScoringV1, rank_interpolate
import numpy.testing as npt

rks = RankScoringV1(2, 4, 3, 8, 3, 12, 3, 15)
rksPower = RankScoringV1(2, 4, 3, 8, 3, 12, 3, 15, True)
scores = [i for i in range(1, 20)]
planAP = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
planAA = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
planA = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
planB = [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
planC = [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
planD = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]


class TestRankScoringV1(TestCase):
    def test_grade_on(self):
        val = rks.grade_on(scores, planA, 1, 2, [1,1], 0.8, 1.0)
        npt.assert_almost_equal(val, 0.8)
        val = rks.grade_on(scores, planAA, 1, 2, [1,1], 0.8, 1.0)
        npt.assert_almost_equal(val, 1.0)
        val = rks.grade_on(scores, planB, 1, 3, [1,1,1],0.8, 1.0)
        npt.assert_almost_equal(val, 0.8)
        val = rks.grade_on(scores, planB, 2, 3, [0,0,1,1], 0.8, 1.0)
        npt.assert_equal(val, None)

    def test_grade(self):
        val = rks.grade(scores, planAP)
        npt.assert_almost_equal(val, 1.0)
        val = rks.grade(scores, planAA)
        npt.assert_almost_equal(val, 0.9)
        val = rks.grade(scores, planA)
        npt.assert_almost_equal(val, 0.8)
        val = rks.grade(scores, planB)
        npt.assert_almost_equal(val, 0.6)
        val = rks.grade(scores, planC)
        npt.assert_almost_equal(val, 0.4)
        val = rks.grade(scores, planD)
        npt.assert_almost_equal(val, 0.2)

    def test_grade_rank(self):
        val = rksPower.grade(scores, planAP)
        npt.assert_almost_equal(val, 1.0)
        val = rksPower.grade(scores, planAA)
        npt.assert_almost_equal(val, 0.9414353837141184)
        val = rksPower.grade(scores, planA)
        npt.assert_almost_equal(val, 0.8757594325717634)
        val = rksPower.grade(scores, planB)
        npt.assert_almost_equal(val, 0.6858529746212939)
        val = rksPower.grade(scores, planC)
        npt.assert_almost_equal(val, 0.4568828092827464)
        val = rksPower.grade(scores, planD)
        npt.assert_almost_equal(val, 0.20551112597542848)

    def test_percent(self):
        npt.assert_almost_equal(rks.percent(scores, planAP, 5), 0.8)
        npt.assert_almost_equal(rks.percent(scores, planAA, 5), 0.6)
        npt.assert_almost_equal(rks.percent(scores, planA, 5), 0.4)

    def test_percents(self):
        pers = rks.percents(scores, planAA)
        npt.assert_almost_equal(pers, (3 / 4, 3 / 8, 1 / 4, 1 / 5))

    def test_target_percents(self):
        npt.assert_almost_equal(rks.target_percents(), [2 / 4, 3 / 8, 3 / 12, 3 / 15])

    def test_letter_of_grade(self):
        self.assertEqual(rks.letter_of_grade(1.0), "A")
        self.assertEqual(rks.letter_of_grade(0.9), "A")
        self.assertEqual(rks.letter_of_grade(0.8), "A")
        self.assertEqual(rks.letter_of_grade(0.799), "B")
        self.assertEqual(rks.letter_of_grade(0.7), "B")
        self.assertEqual(rks.letter_of_grade(0.6), "B")
        self.assertEqual(rks.letter_of_grade(0.5999), "C")
        self.assertEqual(rks.letter_of_grade(0.5), "C")
        self.assertEqual(rks.letter_of_grade(0.4), "C")
        self.assertEqual(rks.letter_of_grade(0.3999), "D")
        self.assertEqual(rks.letter_of_grade(0.3), "D")
        self.assertEqual(rks.letter_of_grade(0.2), "D")
        self.assertEqual(rks.letter_of_grade(0.1999), "F")
        self.assertEqual(rks.letter_of_grade(0.1), "F")

    def test_rank_interpolate3A(self):
        plan_subset = [1, 1, 1]
        score = rank_interpolate(plan_subset, 1, 0.8, 1.0)
        npt.assert_almost_equal(score, 1)
        plan_subset = [1,1,0]
        score = rank_interpolate(plan_subset, 1, 0.8, 1.0)
        npt.assert_almost_equal(score, 0.9414353837141184)
        plan_subset = [1,0,1]
        score = rank_interpolate(plan_subset, 1, 0.8, 1.0)
        npt.assert_almost_equal(score, 0.9171646162858818)
        plan_subset = [1,0,0]
        score = rank_interpolate(plan_subset, 1, 0.8, 1.0)
        npt.assert_almost_equal(score, 0.8586)
        plan_subset = [0,1,0]
        score = rank_interpolate(plan_subset, 1, 0.8, 1.0)
        npt.assert_almost_equal(score, 0.8242707674282367)
        plan_subset = [0,0,1]
        score = rank_interpolate(plan_subset, 1, 0.8, 1.0)
        npt.assert_almost_equal(score, 0.8)

    def test_rank_interpolate5B(self):
        prev_grade_min=2
        prev_grade_out_of=3
        best_plan = [1, 0, 0, 1, 1]
        plan_subset = [1, 0, 0, 1, 1]
        score = rank_interpolate(plan_subset, 2, 0.6, 0.8, best_plan)
        npt.assert_almost_equal(score, 0.8)
        plan_subset = [0,1,0,1,1]
        score = rank_interpolate(plan_subset, 2, 0.6, 0.8, best_plan)
        npt.assert_almost_equal(score, 0.7414)
        plan_subset = [0,0,1,1,1]
        score = rank_interpolate(plan_subset, 2, 0.6, 0.8, best_plan)
        npt.assert_almost_equal(score, 0.6999698)
        plan_subset = [1,0,0,1,0]
        score = rank_interpolate(plan_subset, 2, 0.6, 0.8, best_plan)
        npt.assert_almost_equal(score, 0.7500301954398001)
        plan_subset = [0,1,0,1,0]
        score = rank_interpolate(plan_subset, 2, 0.6, 0.8, best_plan)
        npt.assert_almost_equal(score, 0.6914301954398001)
        plan_subset = [1,0,0,0,1]
        score = rank_interpolate(plan_subset, 2, 0.6, 0.8, best_plan)
        npt.assert_almost_equal(score, 0.7293213514000001)
        plan_subset = [0,1,0,0,1]
        score = rank_interpolate(plan_subset, 2, 0.6, 0.8, best_plan)
        npt.assert_almost_equal(score, 0.6707213514)
        plan_subset = [0,0,1,0,1]
        score = rank_interpolate(plan_subset, 2, 0.6, 0.8, best_plan)
        npt.assert_almost_equal(score, 0.6292911514)
        plan_subset = [0,0,0,1,1]
        score = rank_interpolate(plan_subset, 2, 0.6, 0.8, best_plan)
        npt.assert_almost_equal(score, 0.6)
