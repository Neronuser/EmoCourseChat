import unittest

from src.models.courses.recommender import Recommender


class RecommenderTests(unittest.TestCase):

    def test_courses_loading(self):
        recommender = Recommender()
        print(recommender.recommend("Hi, I want to study marketing"))

