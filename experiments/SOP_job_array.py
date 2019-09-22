from spinup.utils.run_utils import ExperimentGrid
from spinup.algos.sop_pytorch.SOP import sop
import time

if __name__ == '__main__':
    import argparse
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--setting', type=int, default=0)
    args = parser.parse_args()
    ## MAKE SURE ALPHA IS ADDED, MAKE SURE EACH SETTING IS ADDED
    ## MAKE SURE exp name is change, make sure used correct sac function

    setting_names = ['env_name', 'seed', 'beta', 'sigma']
    settings = [['Humanoid-v2','Ant-v2','HalfCheetah-v2','Hopper-v2','Walker2d-v2'],
               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
               [1.1, 1.3, 1.4],
               [0.1, 0.2, 0.4]]

##########################################DON'T NEED TO MODIFY#######################################
    ## basically, maps a set of settings to a hpc job array id
    total = 1
    for sett in settings:
        total *= len(sett)

    print("total: ", total)

    def get_setting(setting_number, total, settings, setting_names):
        indexes = []  ## this says which hyperparameter we use
        remainder = setting_number
        for setting in settings:
            division = int(total / len(setting))
            index = int(remainder / division)
            remainder = remainder % division
            indexes.append(index)
            total = division
        actual_setting = {}
        for j in range(len(indexes)):
            actual_setting[setting_names[j]] = settings[j][indexes[j]]
        return indexes, actual_setting

    indexes, actual_setting = get_setting(args.setting, total, settings, setting_names)
####################################################################################################

    ## use eg.add to add parameters in the settings or add parameters tha apply to all jobs
    eg = ExperimentGrid(name='SOP')
    #if actual_setting['env_name'] == 'Ant-v2' :
    #    eg.add('epochs', 600)
    #else:
    eg.add('epochs', 200)
    eg.add('steps_per_epoch', 5000)
    eg.add('env_name', actual_setting['env_name'], '', True)
    eg.add('seed', actual_setting['seed'])
    eg.add('beta', actual_setting['beta'],'beta_', True)
    eg.add('sigma', actual_setting['sigma'], 'sigma_', True)
    #if actual_setting['env_name'] == 'Humanoid-v2':
    #    eg.add('alpha',0.05)

    eg.run(sop, num_cpu=args.cpu)

    print('\n###################################### GRID EXP END ######################################')
    print('total time for grid experiment:',time.time()-start_time)
