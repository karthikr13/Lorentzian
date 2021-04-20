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
    wrapper.train_network()


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
    print("training using gradient ascent")
    return wrapper.train_network_3()


if __name__ == '__main__':
    flags = flagreader.read_flag()
    flags.model_name = '1_lorentzian_gd'
    flags.strength = 0.01
    train(flags)
    #train_ga(flags)
