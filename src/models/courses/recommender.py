import json
import logging
import os
import re
from collections import defaultdict, namedtuple

import numpy as np
from nltk.corpus import stopwords

from src.models.conversational.utils import APP_NAME
from src.utils import save_object, load_object, load_word2vec, closest_to_vector

Course = namedtuple('Course', ['title', 'link', 'short_description', 'text'])
PUNCTUATION_PATTERN = re.compile("[^\w\s]+")


class Recommender(object):
    """Recommend courses by embedding similarity search."""

    def __init__(self, w2v_path, save_dir='models/recommender', courses_path='data/processed/grouped_courses.json'):
        """Load, preprocess and precompute word and course vectors. Load vectors if precomputed are available.

        Args:
            w2v_path (str): Google News Word2Vec word embeddings path.
            save_dir (str): Directory path to persist precomputed vectors.
            courses_path (str): Courses json path, grouped by category {category: Course}.

        """
        self.logger = logging.getLogger(APP_NAME + ".Recommender")
        self.logger.info("Loading word2vec embeddings")
        self.word2id, self.id2word, self.word_embeddings = load_word2vec(w2v_path, 1000000)
        self.logger.info("Loaded %d word vectors with dim=%d" % self.word_embeddings.shape)
        self.stopwords = stopwords.words("english")

        course2id_path = os.path.join(save_dir, 'course2id.pkl')
        id2course_path = os.path.join(save_dir, 'id2course.pkl')
        category2id_path = os.path.join(save_dir, 'category2id.pkl')
        id2category_path = os.path.join(save_dir, 'id2category.pkl')
        category2courses_path = os.path.join(save_dir, 'category2courses.pkl')
        course_embeddings_path = os.path.join(save_dir, 'course_embeddings.pkl')
        category_embeddings_path = os.path.join(save_dir, 'category_embeddings.pkl')
        if os.path.exists(save_dir):
            self.logger.info("Loading course embeddings")
            self.course2id = load_object(course2id_path)
            self.id2course = load_object(id2course_path)
            self.category2id = load_object(category2id_path)
            self.id2category = load_object(id2category_path)
            self.category2courses = load_object(category2courses_path)
            self.course_embeddings = load_object(course_embeddings_path)
            self.category_embeddings = load_object(category_embeddings_path)
        else:
            os.makedirs(save_dir)
            self.logger.info("Course embeddings not found, building")
            self.course2id, self.id2course, self.category2id, self.id2category, self.category2courses, \
                self.course_embeddings, self.category_embeddings = self._prepare_courses(courses_path)
            save_object(self.course2id, course2id_path)
            save_object(self.id2course, id2course_path)
            save_object(self.category2id, category2id_path)
            save_object(self.id2category, id2category_path)
            save_object(self.category2courses, category2courses_path)
            save_object(self.course_embeddings, course_embeddings_path)
            save_object(self.category_embeddings, category_embeddings_path)

    def _prepare_courses(self, courses_path):
        """Extract and encode courses and their categories.

        Args:
            courses_path (str): Courses json path, grouped by category {category: Course}.

        Returns:
            {Course: int}, [Course], {str: int}, [str], {int: int}, np.array((n_courses, 300)),
            np.array((n_categories, 300)): Course->id mapping, id->Course mapping, Category->id mapping,
            id->Category mapping, course embeddings, category embeddings.

        """
        with open(courses_path) as courses_handle:
            grouped_courses = json.load(courses_handle)
        courses_number = 0
        course2id = {}
        id2course = []
        category2id = {}
        id2category = []
        category2courses = defaultdict(list)
        course_embeddings = []
        category_embeddings = []
        for i, (category, courses_json) in enumerate(grouped_courses.items()):
            category2id[category] = i
            id2category.append(category)
            preprocessed_category = self._preprocess_text(category)
            category_embeddings.append(self._encode_word_list(preprocessed_category))
            courses_json = json.loads(courses_json)
            for course_json in courses_json:
                course = Course(course_json["title"], course_json["mooc_list_url"], course_json["short_description"],
                                course_json["text"])
                if course not in course2id:
                    course2id[course] = courses_number
                    id2course.append(course)
                    preprocessed_course = self._preprocess_course(course)
                    course_embeddings.append(self._encode_word_list(preprocessed_course))
                    courses_number += 1
                category2courses[i].append(course2id[course])
        return course2id, id2course, category2id, id2category, category2courses, np.array(course_embeddings), \
            np.array(category_embeddings)

    def _preprocess_course(self, course):
        """Extract, join and preprocess all text information about the course: title, short and full description."""
        return self._preprocess_text(" ".join([course.title,
                                               course.short_description if course.short_description else '',
                                               course.text if course.text else '']))

    def _preprocess_text(self, text):
        """Remove punctuation, stopwords and split into a list of words."""
        text_no_punctuation = re.sub(PUNCTUATION_PATTERN, " ", text)
        text_no_punctuation_no_stopwords = [word for word in text_no_punctuation.lower().split(" ") if
                                            word not in self.stopwords]
        return text_no_punctuation_no_stopwords

    def _encode_word_list(self, word_list):
        """Average word vectors from the given list.

        Args:
            word_list ([str]): List of target words - subsentence/sentence/paragraph.

        Returns:
            np.array(300): Mean vector of embedded words in the list.
            None: If no embeddings were found for given words.

        """
        vectors = [self.word_embeddings[self.word2id[word]] for word in word_list if word in self.word2id]
        if vectors:
            return np.mean(np.array(vectors), axis=0)
        return None

    def recommend(self, text, threshold=0.5):
        """Return semantically closest course to given text.

        Args:
            text (str): Arbitrary string, related to the field of learning.
            threshold (float): Maximum appropriate cosine distance to be considered close.

        Returns:
            Course: Closest course along with title, link and description.
            None: If no courses passed the threshold.

        """
        preprocessed_text = self._preprocess_text(text)
        text_embedding = self._encode_word_list(preprocessed_text)
        if text_embedding is None:
            self.logger.debug("Embedding nan for " + text)
            return None
        closest_category_ids = closest_to_vector(text_embedding, self.category_embeddings, 1, threshold=threshold)
        if closest_category_ids is None:
            self.logger.debug(text + ": " + str(threshold))
            return None
        category_course_ids = self.category2courses[closest_category_ids[0]]
        category_course_embeddings = self.course_embeddings[category_course_ids]
        closest_course_id = closest_to_vector(text_embedding, category_course_embeddings, 1)[0]
        return self.id2course[category_course_ids[closest_course_id]]
