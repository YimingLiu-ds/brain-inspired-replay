#!/usr/bin/env python3
import os
import numpy as np
from param_stamp import get_param_stamp_from_args
import options
from visual import plt as my_plt
from matplotlib.pyplot import get_cmap
import main_cl



## Parameter-values to compare
lamda_list = [1., 10., 100., 1000., 10000., 100000., 1000000., 10000000., 100000000., 1000000000., 10000000000.,
              100000000000., 1000000000000.]
c_list = [0.1, 1., 10., 100., 1000., 10000., 100000., 1000000., 10000000., 100000000., 1000000000.,
                  10000000000.]
dg_prop_list = [0., 0.2, 0.4, 0.6, 0.8]

## Repulsion hyperparameters...
lamda_diff_list = [1., 10., 100., 1000., 10000., 100000., 1000000., 10000000., 100000000., 1000000000., 10000000000.,
              100000000000.]
# Whether to use KL or JS divergence...
#kl_js_list = ['kl', 'js']
# Selectrion factors...
#f_list = [1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5]


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'single_task': False, 'only_MNIST': False, 'generative': True, 'compare_code': 'hyper'}
    # Define input options
    parser = options.define_args(filename="_compare_CIFAR100_hyperParams",
                                 description='Compare hyperparameters fo split CIFAR-100.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_task_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_allocation_options(parser, **kwargs)
    parser = options.add_replay_options(parser, **kwargs)
    parser = options.add_bir_options(parser, **kwargs)
    parser.add_argument('--per-bir-comp', action='store_true', help="also do gridsearch for individual BI-R components")
    # Parse and process (i.e., set defaults for unselected options) options
    args = parser.parse_args()
    options.set_defaults(args, **kwargs)
    return args


def get_result(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run, and if not do so
    if os.path.isfile('{}/prec-{}.txt'.format(args.r_dir, param_stamp)):
        print("{}: already run".format(param_stamp))
    else:
        print("{}: ...running...".format(param_stamp))
        main_cl.run(args)
    # -get average precision
    fileName = '{}/prec-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -return it
    return ave


if __name__ == '__main__':

    ## Load input-arguments & set default values
    args = handle_inputs()

    ## Add default arguments (will be different for different runs)
    args.repulsion = True
    args.ewc = False
    args.online = False
    args.si = True
    args.xdg = False
    args.freeze_convE = False

    ## Use pre-trained convolutional layers for all compared methods
    args.pre_convE = True

    ## If needed, create plotting directory
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    ## Brain-inspired Replay with SI
    args.replay = "generative"
    args.hidden = True
    args.distill = True
    args.prior = "GMM"
    args.per_class = True
    args.dg_gates = True
    args.feedback = True
    args.freeze_convE = True
    BIR_SI = {}
    for dg_prop in dg_prop_list:
        BIR_SI[dg_prop] = {}
        args.dg_prop = dg_prop
        for si_c in [0]+c_list:
            args.si_c = si_c
            args.si = True if si_c>0 else False
            BIR_SI[dg_prop][si_c] = {}
            for lamda_diff in lamda_diff_list:
                args.lamda_diff = lamda_diff
                BIR_SI[dg_prop][si_c][lamda_diff] = get_result(args) 


    #-------------------------------------------------------------------------------------------------#

    #--------------------------------------------#
    #----- COLLECT DATA AND PRINT ON SCREEN -----#
    #--------------------------------------------#

    ext_c_list = [0] + c_list
    ext_lambda_list = [0] + lamda_list
    print("\n")


    ###---BI-R + SI---###

    # -collect data
    ave_prec_bir_per_c = []
    for dg_prop in dg_prop_list:
        ave_prec_bir_per_c.append([BIR_SI[dg_prop][c] for c in ext_c_list])
    # -print on screen
    print("\n\nBI-R & SI")
    print(" param-list (si_c): {}".format(ext_c_list))
    curr_max = 0
    for dg_prop in dg_prop_list:
        ave_prec_temp = [BIR_SI[dg_prop][c] for c in ext_c_list]
        print("  (dg-prop={}):   {}".format(dg_prop, ave_prec_temp))
        if np.max(ave_prec_temp)>curr_max:
            dg_prop_max = dg_prop
            si_max = ext_c_list[np.argmax(ave_prec_temp)]
            curr_max = np.max(ave_prec_temp)
    print("--->  dg_prop = {}  -  si_c = {}     --    {}".format(dg_prop_max, si_max, curr_max))


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- PLOTTING -----#
    #--------------------#

    # name for plot
    plot_name = "hyperParams-{}{}-{}".format(args.experiment, args.tasks, args.scenario)
    scheme = "incremental {} learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)
    ylabel = "Average accuracy (after all tasks)"

    # calculate limits y-axes (to have equal for all graphs)
    full_list = [item for sublist in ave_prec_per_lambda for item in sublist] + ave_prec_si + ave_prec_bir + \
                [item for sublist in ave_prec_bir_per_c for item in sublist]

    miny = np.min(full_list)
    maxy = np.max(full_list)
    marginy = 0.1*(maxy-miny)

    # open pdf
    pp = my_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []


    ###---Brain-Inspired Replay + SI---###
    # - select colors
    colors = get_cmap('Blues_r')(np.linspace(0.6, 0., len(dg_prop_list))).tolist()
    # - make plot (line plot - only average)
    figure = my_plt.plot_lines(ave_prec_bir_per_c, x_axes=ext_c_list, ylabel=ylabel,
                               line_names=["BI-R, gate-prop = {}".format(dg_prop) for dg_prop in dg_prop_list],
                               title=title, x_log=True, xlabel="BI-R + SI: c (log-scale)",
                               ylim=(miny-marginy, maxy+marginy),
                               with_dots=True, colors=colors, h_line=BASE, h_label="None")
    figure_list.append(figure)


    # add figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))
