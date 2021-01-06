from wrapper import NetworkWrapper
from network import Network
import flagreader
import datareader


def train(flags):
    train_loader, test_loader = datareader.read_data(x_range=flags.x_range,
                                                     y_range=flags.y_range,
                                                     geoboundary=flags.geoboundary,
                                                     batch_size=flags.batch_size,
                                                     normalize_input=flags.normalize_input,
                                                     data_dir=flags.data_dir,
                                                     test_ratio=flags.test_ratio)
    if flags.normalize_input:
        flags.geoboundary_norm = [-1, 1, -1, 1]
    wrapper = NetworkWrapper(flags, train_loader, test_loader)
    print("training")
    return wrapper.train_network()


def train_ga(flags):
    train_loader, test_loader = datareader.read_data(x_range=flags.x_range,
                                                     y_range=flags.y_range,
                                                     geoboundary=flags.geoboundary,
                                                     batch_size=flags.batch_size,
                                                     normalize_input=flags.normalize_input,
                                                     data_dir=flags.data_dir,
                                                     test_ratio=flags.test_ratio)
    if flags.normalize_input:
        flags.geoboundary_norm = [-1, 1, -1, 1]
    wrapper = NetworkWrapper(flags, train_loader, test_loader)
    print("training")
    return wrapper.train_network_ascent()


if __name__ == '__main__':
    flags = flagreader.read_flag()
    train(flags)
