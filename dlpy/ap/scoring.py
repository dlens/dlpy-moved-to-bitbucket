'''
Here for different scoring mechanisms
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D


def decay_between(stepsToDecay, max_steps, lower, upper):
    diff = upper - lower
    slope = (upper - lower)/max_steps
    return lower + stepsToDecay*slope


class RankScoringV1:
    '''
    This class scores using the Rank Scoring algorithm, where you set a target/out_of for each
    level A,B,C,D and the first target/out_of you hit is the letter grade.  The gradations come
    from how much above that level you get.
    '''
    def __init__(self, a_target, a_out_of, b_target, b_out_of, c_target, c_out_of, d_target, d_out_of):
        '''
        Constructor
        :param a_target: the target number for the A target.  For instance if 2 of top 5 is an A, this would be 2
        :param a_out_of: the out_of number for the A target. For instance if 2 of top 5 is an A, this would be 2
        :param b_target: the target number for the B target.  For instance if 3 of top 10 is an B, this would be 3
        :param b_out_of: the out_of number for the B target. For instance if 3 of top 10 is an B, this would be 10
        :param c_target: the target number for the C target.  For instance if 4 of top 15 is an C, this would be 4
        :param c_out_of: the out_of number for the C target. For instance if 4 of top 15 is an C, this would be 15
        :param d_target: the target number for the D target.  For instance if 5 of top 25 is an D, this would be 5
        :param d_out_of: the out_of number for the D target. For instance if 5 of top 25 is an D, this would be 25
        '''
        self.a_target = a_target
        self.b_target = b_target
        self.c_target = c_target
        self.d_target = d_target
        self.a_out_of = a_out_of
        self.b_out_of = b_out_of
        self.c_out_of = c_out_of
        self.d_out_of = d_out_of
        self.grade_decay_rate = 0.707

    @staticmethod
    def standard(scores):
        '''
        Creates a standard ranking scoring for a list of scores
        :param scores:
        :return: A RankTargets object with cut-offs that are standard
        '''
        nitems = len(scores)
        if nitems <= 1:
            return RankScoringV1(1, 1, 1, 1, 1, 1, 1, 1, 1)
        elif nitems == 2:
            return RankScoringV1(1, 1, 1, 2, 1, 2, 1, 2)
        elif nitems == 3:
            return RankScoringV1(1, 1, 1, 2, 1, 3, 1, 3)
        elif nitems <=5:
            return RankScoringV1(1, 1, 1, 2, 1, 3, 1, 4)
        elif nitems <=7:
            return RankScoringV1(1, 2, 1, 3, 1, 4, 1, 5)
        elif nitems <=10:
            return RankScoringV1(1, 2, 2, 4, 2, 6, 1, 6)
        elif nitems <=20:
            return RankScoringV1(1, 2, 2, 4, 3, 6, 4, 8)
        elif nitems <=40:
            return RankScoringV1(1, 4, 2, 8, 3, 12, 4, 16)
        elif nitems <=100:
            return RankScoringV1(2, 5, 3, 10, 4, 15, 4, 20)
        else:
            return RankScoringV1(2, 10, 2, 15, 3, 20, 4, 25)

    def __str__(self):
        return "A={} of top {}\nB={} of top {}\nC={} of top {}\nD={} of top {}".format(
            self.a_target, self.a_out_of,
            self.b_target, self.b_out_of,
            self.c_target, self.c_out_of,
            self.d_target, self.d_out_of
        )

    @staticmethod
    def grade_on(scores, plan, target, out_of, min_grade, max_grade, return_none=True):
        '''
        A simple function to get the grade of a plan ON a particular target.  Used by the grade() method only.
        :param plan: The plan to grade
        :param scores: The scores to grade with
        :param target: The target you are trying to hit
        :param out_of: The out_of the top ______ you are trying to hit the target on
        :param min_grade: The minimum grade this would possibly get if we hit the target
        :param max_grade: The maximum grade this would possibly get if we hit the target
        :param return_none: If True and we don't hit the target None is returned, otherwise the percentage of
        the top out_of divided by target/out_of percent.  In that case it returns 0 if none of the top
        out_of items were in the plan up to slightly less than 1 if target-1 of the top out_of items are
        in the plan.
        :return: If the target is hit, the grade is returned. Otherwise, if return_none=True it returns None,
        otherwise it returns (nitems_in_plan_of_top_out_of / out_of) / (target / out_of)
        '''
        sort_ix = np.argsort(scores)
        total = 0
        for i in range(out_of):
            if plan[sort_ix[i]]:
                total += 1
        if total >= target:
            # We have an A, how much over the A are we
            overage = total - target
            # Exponentially decay between upper and lower
            return decay_between(overage, out_of - target, min_grade, max_grade)
        else:
            if return_none:
                return None
            else:
                percentage = (total / out_of) / (target / out_of)
                diff = max_grade - min_grade
                return min_grade + percentage * diff

    def grade(self, scores, plan):
        '''
        Grades a plan on a set of scores, returning 0->Worst F to 1->Best A.
        :param scores: The scores to grade upon
        :param plan: The plan to score
        :return: The grade, a number between 0 and 1.
        '''
        # Check for A
        score = self.grade_on(scores, plan, self.a_target, self.a_out_of, 0.8, 1.0)
        if score is not None:
            return score
        # Check for B
        score = self.grade_on(scores, plan, self.b_target, self.b_out_of, 0.6, 0.8)
        if score is not None:
            return score
        # Check for C
        score = self.grade_on(scores, plan, self.c_target, self.c_out_of, 0.4, 0.6)
        if score is not None:
            return score
        # Check for D
        score = self.grade_on(scores, plan, self.d_target, self.d_out_of, 0.2, 0.4)
        if score is not None:
            return score
        # We have an F
        return self.grade_on(scores, plan, self.d_target, self.d_out_of, 0.0, 0.2, return_none=False)

    @staticmethod
    def percent(scores, plan, out_of):
        '''
        Used for reporting purposes, returns the percentage of each A, B, C, D out_ofs for a given plan.
        :param scores: The scores to grade on
        :param plan: The plan to grade
        :param out_of: The out_of number to use
        :return:
        '''
        sort_ix = np.argsort(scores)
        total = 0
        for i in range(out_of):
            if plan[sort_ix[i]]:
                total += 1
        return total / out_of

    def percents(self, scores, plan, return_targets_out_ofs=False):
        '''
        Used for reporting purposes, returns the A, B, C, D percentages out of their out_of values
        :param scores: The scores to calculate percentages using
        :param plan: The plan to calculate percentages
        :param return_targets_out_ofs: If False we just return the percentages for A,B,C,D in a list.
        Other we return a list of 4 tuples of the form (percent, target, out), for each of A,B,C,D
        respectively.
        :return:
        '''
        if return_targets_out_ofs:
            return [
                (self.percent(scores, plan, self.a_out_of), self.a_target, self.a_out_of),
                (self.percent(scores, plan, self.b_out_of), self.b_target, self.b_out_of),
                (self.percent(scores, plan, self.c_out_of), self.c_target, self.c_out_of),
                (self.percent(scores, plan, self.d_out_of), self.d_target, self.d_out_of)
            ]
        return [
            self.percent(scores, plan, self.a_out_of),
            self.percent(scores, plan, self.b_out_of),
            self.percent(scores, plan, self.c_out_of),
            self.percent(scores, plan, self.d_out_of),
        ]

    def target_percents(self):
        '''
        Returns the targets are percentages for A,B,C,D respectively as a list.
        :return:
        '''
        return [
            self.a_target / self.a_out_of,
            self.b_target / self.b_out_of,
            self.c_target / self.c_out_of,
            self.d_target / self.d_out_of
        ]

    def letter_of_grade(self, score):
        '''
        In case you ever need it, this converts the 0-1 grade to a letter
        :param score:
        :return:
        '''
        if score > 0.8:
            return "A"
        elif score > 0.6:
            return "B"
        elif score > 0.4:
            return "C"
        elif score > 0.2:
            return "D"
        else:
            return "F"

    def plot(self, scores, plan,
             unselected_bar_color='#0000ff33',
             selected_bar_color='#0000ffff'):
        '''
        Creates a cool matplotlib based plot of the A/B/C/D targets and how close the plan is to reacing them.
        It has a nice title that shows the grade, generally very useful for visually playing.
        :param scores: The scores to grade against.  Anything that can be sorted.
        :param plan: The plan, which is either a boolean list, or list of 0-1 scores.
        :param unselected_bar_color: The color for the bars A/B/C/D that are not the ones used for grading
        :param selected_bar_color: The color for the bar A/B/C/D that is used for grading
        :return: Nothing
        '''
        ps = self.percents(scores, plan)
        raw_grade = self.grade(scores, plan)
        xs = [1, 2, 3, 4]
        targets = self.target_percents()
        if raw_grade >= 0.8:
            grade_index = 0
        elif raw_grade >= 0.6:
            grade_index = 1
        elif raw_grade >= 0.4:
            grade_index = 2
        elif raw_grade >= 0.2:
            grade_index = 3
        else:
            grade_index = None

        if grade_index is None:
            # We have an F and all bars should be the same color
            plt.bar(xs, ps, zorder=1, color=unselected_bar_color)
        else:
            plt.bar(xs[grade_index:(grade_index + 1)], ps[grade_index:(grade_index + 1)],
                    zorder=1, color=selected_bar_color)
            xs.pop(grade_index)
            ps.pop(grade_index)
            plt.bar(xs, ps, zorder=1, color=unselected_bar_color)

        # plt.bar(xs, ps, zorder=1)
        plt.scatter([1, 2, 3, 4], targets, c="red", marker="d", s=200, zorder=2)
        xtick_labels = ["{}\n% of top {}".format(letter, out_of) for letter, out_of in
                        zip(
                            ("A", "B", "C", "D"),
                            (self.a_out_of, self.b_out_of, self.c_out_of, self.d_out_of)
                        )]
        plt.xticks([1, 2, 3, 4], xtick_labels)

        # Let's annotate our targets
        count = 1
        for info in self.percents(scores, plan, return_targets_out_ofs=True):
            percent, target, out_of = info
            plt.annotate("{} of top {}".format(target, out_of), (count, target / out_of), textcoords="offset points",
                         xytext=(15, 0), ha="left")
            count += 1
        plt.ylim(0, 1)
        ax = plt.gca()
        legend_elements = [
            Line2D([0], [0], marker='d', color='w', label='Target',
                   markerfacecolor='red', markersize=15)
        ]
        ax.legend(handles=legend_elements)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        grade = "{:.1f}".format(raw_grade * 100)
        plt.title("Our plan grade is {}, {}th percentile".format(self.letter_of_grade(raw_grade), grade))
