import train
import flagreader

if __name__ == '__main__':

    flags = flagreader.read_flag()  # setting the base case
    flags.model_name = 'default_10x'
    losses = []
    for i in range(10):
        losses.append(train.train(flags))
    print(losses)
