import json

import bottle
from bottle import request, response

from src.models.courses.recommender import Recommender

app = application = bottle.default_app()
recommender = Recommender()


def cors(func):
    def wrapper(*args, **kwargs):
        bottle.response.set_header("Access-Control-Allow-Origin", "*")
        bottle.response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        bottle.response.set_header("Access-Control-Allow-Headers", "Origin, Content-Type")

        # skip the function if it is not needed
        if bottle.request.method == 'OPTIONS':
            return

        return func(*args, **kwargs)

    return wrapper


@app.route('/utterance', method=['OPTIONS', 'POST'])
@cors
def utterance_handler():
    data = request.json
    utterance = data["content"]

    response.headers['Content-Type'] = 'application/json'
    text_response = recommender.recommend(utterance)
    return json.dumps({'response': text_response.link})


if __name__ == '__main__':
    bottle.run(host='127.0.0.1', port=8000)
