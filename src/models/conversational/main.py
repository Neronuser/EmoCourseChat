from src.models.conversational.evaluate import run_test
from src.models.conversational.train import train_iterations
from src.utils import parse_config


def parse_filename(filename, test=False):
    filename = filename.split('/')
    dataType = filename[-1][:-4]  # remove '.tar'
    parse = dataType.split('_')
    reverse = 'reverse' in parse
    layers, hidden = filename[-2].split('_')
    n_layers = int(layers.split('-')[0])
    hidden_size = int(hidden)
    return n_layers, hidden_size, reverse


def run(config):
    reverse, fil, n_iteration, print_every, save_every, learning_rate, n_layers, hidden_size, batch_size, beam_size, input, max_length, max_words = \
        config.getboolean('Reverse'), config.getboolean('Filter'), config.getint('Iteration'), config.getint('Print'), config.getfloat('SaveEvery'), config.getfloat('LearningRate'), \
        config.getint('Layer'), config.getint('Hidden'), config.getint('Batch_size'), config.getint('Beam'), config.getboolean('Input'), config.getint('MaxLength'), config.getint('MaxWords')
    if config['Train'] and not config['Load']:
        train_iterations(config['Train'], reverse, n_iteration, learning_rate, batch_size,
                   n_layers, hidden_size, print_every, save_every, max_length, max_words)
    elif config['Load']:
        n_layers, hidden_size, reverse = parse_filename(config['Load'])
        train_iterations(config['Train'], reverse, n_iteration, learning_rate, batch_size,
                   n_layers, hidden_size, print_every, save_every, max_length, max_words, load_filename=config['Load'])
    elif config['Test']:
        n_layers, hidden_size, reverse = parse_filename(config['Test'], True)
        run_test(n_layers, hidden_size, reverse, config['Test'], beam_size, input, config['Corpus'])


if __name__ == '__main__':
    config = parse_config('dialogue')
    run(config)
