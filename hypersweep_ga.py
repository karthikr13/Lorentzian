"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import time
import flagreader

if __name__ == '__main__':

    flags = flagreader.read_flag()  # setting the base case
    # flags.linear = [8, 100, 100, 12]
    model_name = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # model_description = "Smooth_L1_Loss_Warm_Restart"
    # model_description = "MSE_Loss_Grad_Clip"
    model_description = "layers{}_nodes{}_reg{}"
    model_strength_desc = "strength{}_run"
    # for restart in [200, 500]:
    #     for exp in [4,8]:
    #         for clip in [20]:
    #             # flags.lr_warm_restart = restart
    #             # flags.use_warm_restart = True
    #             flags.grad_clip = clip
    #             for i in range(5):
    #                 flags.linear = [8, 100, 100, 12]
    #                 flags.model_name = model_name + model_description +str(exp)  + '_WRst_' + str(restart) + "_GC_" + \
    #                                    str(clip) + "_run" + str(i + 1)
    #                 # flags.model_name = model_name + model_description + "_L" + str(exp) +"_GC_" + \
    #                 #                    str(clip) + "_run" + str(i + 1)
    #                 train_network.training_from_flag(flags)
    # for i in range(3):
    #     flags.linear = [8, 100, 100, 100]
    #     flags.model_name = model_name + '_' + model_description + "_run" + str(i + 1)
    #     train.training_from_flag(flags)
    """
    for lr in [1e-5,1e-4, 1e-3, 1e-2, 1e-1]:
        flags.lr = lr
        flags.model_name = model_name + '_' + model_description.format(len(flags.linear),flags.linear[1],
                                                                       flags.reg_scale,lr)+"_run"
        train.training_from_flag(flags)"""
    best_losses = {}
    '''
    for l in [1,3,5]:
        for reg in [1e-5,1e-4, 1e-3, 1e-2, 1e-1]:
            for n in [10, 30, 50, 70,100, 150, 200,500]:
                    flags.reg_scale = reg
                    flags.linear = [n for j in range(l+2)]
                    flags.linear[0] = 2
                    flags.linear[-1] = 300
                    flags.model_name = model_name + '_' + model_description.format(l,n,reg,)+"_run"
                    best_losses[flags.model_name] = train.train_ga(flags)
    '''
    for s in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        flags.linear[0] = 2
        flags.linear[-1] = 100
        flags.model_name = model_strength_desc.format(s)
        flags.strength = s
        best_losses[flags.model_name] = train.train_ga(flags)
    for i in best_losses:
        print(i + ": " + str(best_losses[i]))
