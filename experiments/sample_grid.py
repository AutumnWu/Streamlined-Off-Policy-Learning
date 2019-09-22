from spinup.utils.run_utils import ExperimentGrid
from spinup.algos.sac_pytorch.sac_pytorch import sac_pytorch
import time

if __name__ == '__main__':
    import argparse
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()

    eg = ExperimentGrid(name='sac-baseline')
    eg.add('env_name', ['Pendulum-v0'], '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 200)
    eg.add('steps_per_epoch', 5000)
    eg.run(sac_pytorch, num_cpu=args.cpu)

    print('\n###################################### GRID EXP END ######################################')
    print('total time for grid experiment:',time.time()-start_time)
