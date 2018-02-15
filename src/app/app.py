import json
import logging

import bottle
from bottle import request, response

from src.app.bot import EmoCourseChat
from src.models.conversational.utils import APP_NAME
from src.utils import parse_config, LOG_FORMAT

config = parse_config('app')
logger = logging.getLogger(APP_NAME)
logger.setLevel(config['LogLevel'])
handler = logging.FileHandler(config['LogPath'])
handler.setLevel(config['LogLevel'])
formatter = logging.Formatter(LOG_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info(dict(config.items()))
app = application = bottle.default_app()
bot = EmoCourseChat(config["Checkpoint"], config["Vocabulary"], config["EmotionVocabulary"], config["Word2Vec"],
                    config.getint("BeamSize"), config.getfloat("Threshold"))


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
    text_response = bot.respond(utterance)
    response.headers['Content-Type'] = 'application/json'
    return json.dumps({'response': text_response})


if __name__ == '__main__':
    bottle.run(host='127.0.0.1', port=8000)
