"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
# os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import time
import flagreader

if __name__ == '__main__':
    for i in range(3):
        flags = flagreader.read_flag()  # setting the base case
        flags.optim='SGD'
        flags.model_name = 'sgd_gd_1L_run{}'.format(i)
        train.train(flags)

    for i in range(3):
        flags = flagreader.read_flag()  # setting the base case
        flags.optim='Adam'
        flags.model_name = 'adam_gd_1L_run{}'.format(i)
        train.train(flags)