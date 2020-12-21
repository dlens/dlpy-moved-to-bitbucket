from unittest import TestCase

from dlpy.ap.scoring import RankScoringV1
import numpy.testing as npt

rks = RankScoringV1(2,4, 3,8, 3,12, 3,15)
scores = [i for i in range(1,20)]
planAP= [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
planAA= [1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
planA = [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
planB = [0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
planC = [0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
planD = [0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0]


class TestRankScoringV1(TestCase):
    def test_grade_on(self):
        val=rks.grade_on(scores, planA, 1, 2, 0.8, 1.0)
        npt.assert_almost_equal(val, 0.8)
        val=rks.grade_on(scores, planAA, 1, 2, 0.8, 1.0)
        npt.assert_almost_equal(val, 1.0)
        val=rks.grade_on(scores, planB, 1, 3, 0.8, 1.0)
        npt.assert_almost_equal(val, 0.8)
        val=rks.grade_on(scores, planB, 2, 3, 0.8, 1.0)
        npt.assert_equal(val, None)


    def test_grade(self):
        val=rks.grade(scores, planAP)
        npt.assert_almost_equal(val, 1.0)
        val=rks.grade(scores, planAA)
        npt.assert_almost_equal(val, 0.9)
        val=rks.grade(scores, planA)
        npt.assert_almost_equal(val, 0.8)
        val=rks.grade(scores, planB)
        npt.assert_almost_equal(val, 0.6)
        val=rks.grade(scores, planC)
        npt.assert_almost_equal(val, 0.4)
        val=rks.grade(scores, planD)
        npt.assert_almost_equal(val, 0.2)

    def test_percent(self):
        npt.assert_almost_equal(rks.percent(scores, planAP, 5), 0.8)
        npt.assert_almost_equal(rks.percent(scores, planAA, 5), 0.6)
        npt.assert_almost_equal(rks.percent(scores, planA, 5), 0.4)

    def test_percents(self):
        pers = rks.percents(scores, planAA)
        npt.assert_almost_equal(pers, (3/4, 3/8, 1/4, 1/5))

    def test_target_percents(self):
        npt.assert_almost_equal(rks.target_percents(), [2/4, 3/8, 3/12, 3/15])

    def test_letter_of_grade(self):
        self.assertEqual(rks.letter_of_grade(1.0), "A")
        self.assertEqual(rks.letter_of_grade(0.9), "A")
        self.assertEqual(rks.letter_of_grade(0.8), "B")
        self.assertEqual(rks.letter_of_grade(0.7), "B")
        self.assertEqual(rks.letter_of_grade(0.6), "C")
        self.assertEqual(rks.letter_of_grade(0.5), "C")
        self.assertEqual(rks.letter_of_grade(0.4), "D")
        self.assertEqual(rks.letter_of_grade(0.3), "D")
        self.assertEqual(rks.letter_of_grade(0.2), "F")
        self.assertEqual(rks.letter_of_grade(0.1), "F")
