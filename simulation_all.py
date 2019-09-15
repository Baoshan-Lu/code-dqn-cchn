import os
import torch
import numpy as np
import argparse
import time
from datetime import datetime
from cchn_model import CCHN
from multi_agent_train import Model_train

from numerical_evaluation import Performace

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# path = '../experiment/'+ datetime.now().strftime('%Y-%m-%d') + '-1' + '/'
path = 'experiment/'+ datetime.now().strftime('%Y-%m-%d') + '-1' + '/'


figure_format = '.fig'

if not os.path.exists(path + 'pdf/'):
    os.makedirs(path + 'pdf/')
if not os.path.exists(path + 'png/'):
    os.makedirs(path + 'png/')
if not os.path.exists(path + 'data/'):
    os.makedirs(path + 'data/')

if __name__ == '__main__':

    '''DQN_model'''
    parameters = argparse.ArgumentParser()
    parameters.add_argument('--epsion', type=int, default=0.8, help="epsion")
    parameters.add_argument('--targetnet_update_rate', type=int, default=100, help="targetnet_update_rate")
    parameters.add_argument('--memory_capacity', type=int, default=2000, help="memory_capacity")
    parameters.add_argument('--batchsize', type=int, default=256)
    parameters.add_argument('--gamma', type=float, default=0.9)
    parameters.add_argument('--epoch', type=int, default=10000)
    parameters.add_argument('--save_path', type=str, default='../models/')
    parameters.add_argument('--data_path', type=str, default='../results/')
    parameters.add_argument('--picture_path', type=str, default='../figures/')
    parameters.add_argument('--simulation_results_path', type=str, default=path)
    parameters.add_argument('--figure_format', type=str, default=figure_format)

    parameters.add_argument('--test_number', type=int, default=100)
    parameters.add_argument('--start_train', type=int, default=1000)

    '''CCHN_network'''
    parameters.add_argument('--radius', type=int, default=200)
    parameters.add_argument('--primary_number', type=int, default=1)
    parameters.add_argument('--secondary_number', type=int, default=1)
    parameters.add_argument('--CR_router_number', type=int, default=3)
    parameters.add_argument('--power_set_number', type=int, default=10)
    parameters.add_argument('--user_power_max', type=float, default=0.1)
    parameters.add_argument('--user_power_min', type=float, default=0.02)
    parameters.add_argument('--reward', type=float, default=100)
    parameters.add_argument('--sigma_factor', type=float, default=10)
    parameters.add_argument('--noise_power', type=float, default=pow(3.9811, -15))  # 3.9811e-15
    parameters.add_argument('--mu', type=float, default=0.0)
    parameters.add_argument('--sigma', type=float, default=0.0005)
    parameters.add_argument('--channel_gain', type=int, default=-4)
    parameters.add_argument('--bandwidth', type=int, default=pow(10, 6))


    parameters.add_argument('--primary_rate_min', type=float, default=0.01)

    parameters.add_argument('--secodary_rate_min', type=float, default=0.7)
    parameters.add_argument('--primary_init_power', type=float, default=0.2)
    parameters.add_argument('--learning_rate', type=float, default=1 * 1e-3)
    parameters.add_argument('--pu_power_mode', type=int, default=2)
    parameters.add_argument('--transition_steps', type=int, default=20)
    parameters.add_argument('--iteration', type=int, default=2000)


    parameters.add_argument('--gpu_type', type=bool, default=False, choices=[True, False])
    parameters.add_argument('--pretrain', type=bool, default=False, choices=[True, False])

    parameters.add_argument('--double_dqn', type=bool, default=True, choices=[True, False])
    parameters.add_argument('--dueling_dqn', type=bool, default=True, choices=[True, False])
    parameters.add_argument('--noisy_dqn', type=bool, default=False, choices=[True, False])

    parameters.add_argument('--distance', type=int, default=[200, 300, 50, 100])
    parameters.add_argument('--weight_reward', type=float, default=[0, 1, 0,0])  # pu,su,pu_sinr

    parameters.add_argument('--sense_coverage', type=int, default=800)


    '''初始参数'''
    parameters = parameters.parse_args([
        '--radius', '500', '--primary_number', '50', '--secondary_number', '80',
        '--CR_router_number', '100',
        '--power_set_number', '4',
        '--memory_capacity', '2000', '--start_train', '64', '--batchsize', '256',
        '--primary_rate_min', '20000',
        '--iteration', '80', '--epoch', '50', '--learning_rate', '0.01',
        '--sense_coverage','500'
    ])

    '''==One:  随机种子'''
    np.random.seed(1)
    torch.manual_seed(1)

    '''==Two:  CCHN网络'''
    cchn = CCHN(parameters,show_model=False)

    '''==Three: 模型训练'''
    model_traing = Model_train(parameters)

    '''==Four: 性能仿真'''
    Performace = Performace(parameters, cchn, model_traing)

    model = ['nature', 'double', 'dueling', 'double_dueling', 'noisy', 'noisy_double']

    default_model = 'double_dueling'
    batchsize_default = 32

    DQN_epoch = 300
    pu_num = 5
    su_num = 3
    crr_num = 30

    DQN_CG_RA_epoch = 50

    su_num_compare = 6
    pu_num_vary = [1,2, 3,4,5,6]

    pu_num_compare = 3
    su_num_vary = [3,4,5,6,7,8]

    game_fail_num = 100

    sum_rate_type = 'su'
    show_picture=True

    Simulate_mode_availeble = {'Fig1_deburg_model': 0,
                               'Fig2_training_with_different_batchsize': 1,
                               'Fig3_compare_sum_rate_fariness_reward_with_DQN_RA_CG': 2,
                               'Fig6_training_with_different_DQN_model': 4,
                               'Fig7_training_with_different_CR_routers': 5,
                               'Fig8_training_with_different_learning_rate': 6,
                               'Fig9_training_with_different_greedy_epsion': 7,
                               'Fig10_training_with_different_PU_MRQ_with_DQN_RA_CG': 33,
                               'Fig11_optimal_allocation': 8,
                               }
    '''仿真模式选择'''
    Simulate_mode = Simulate_mode_availeble['Fig11_optimal_allocation']


    print('Start simulation....')
    print('\nSimulate_mode=1: Fig1  DQN-DoubleDQN-DuelingDQN——对比 \n')
    model=['nature','double']
    #,#, 'double' 'dueling', 'double_dueling']
    Performace.compare_different_dqn_model1(pu_num=pu_num, su_num=su_num, crr_num=crr_num,
                                           EPOCH=DQN_epoch, ITERATION=80, learningrate=0.01,
                                           batchsize=32, model=model,
                                            sum_rate_type=sum_rate_type,show=show_picture)


    print('\nSimulate_mode=2: Fig2 不同batch size的训练过程')
    Performace.sum_rate_varying_batchsize(pu_num=pu_num, su_num=su_num, crr_num=crr_num,
                                          EPOCH=DQN_epoch, ITERATION=80, learningrate=0.01,
                                          batchsize=[8, 16, 32, 64], model=default_model,
                                          sum_rate_type=sum_rate_type,show=show_picture)


    print('\nSimulate_mode=3: Fig3 CR-router对模型的影响  \n')
    Performace.compare_crr_effect(pu_num=pu_num, su_num=su_num, crr_num=[10, 25, 40, 55],
                                  EPOCH=DQN_epoch, ITERATION=80, learningrate=0.01, batchsize=32,
                                  model=default_model,sum_rate_type=sum_rate_type,show=show_picture)


    print('\nSimulate_mode=4: Fig4 Learning_rate的影响  \n')
    Performace.compare_different_learning_rate(pu_num=pu_num, su_num=su_num, crr_num=crr_num,
                                               EPOCH=DQN_epoch, ITERATION=80, learningrate=[0.1, 0.01, 0.001, 0.0001],
                                               batchsize=32, model=default_model,sum_rate_type=sum_rate_type,show=show_picture)


    print('\nSimulate_mode=5: Fig5 e-greedy的影响  \n')
    Performace.compare_different_e_greedy_epsion(pu_num=pu_num, su_num=su_num, crr_num=crr_num,
                                                 EPOCH=DQN_epoch, ITERATION=80, learningrate=0.01, batchsize=32,
                                                 model=default_model,
                                                 epsion=[0.7, 0.8, 0.9],sum_rate_type=sum_rate_type,show=show_picture)


    print('\nSimulate_mode=61: Fig3 DQN-RA_CG  SU sum_rate fairness  reward \n')
    Performace.compare_dqn_random_game_sum_rate_fairness_reward(pu_num=pu_num_compare, su_num=su_num_vary, crr_num=crr_num,
                                                                EPOCH=DQN_CG_RA_epoch, ITERATION=80, learningrate=0.01,
                                                                batchsize=batchsize_default,
                                                                mode='su', model=default_model
                                                                ,game_fail_num=game_fail_num,
                                                                sum_rate_type=sum_rate_type,show=show_picture)


    print('\nSimulate_mode=62: Fig3 DQN-RA_CG  PU sum_rate fairness  reward  \n')
    Performace.compare_dqn_random_game_sum_rate_fairness_reward(pu_num=pu_num_vary, su_num=su_num_compare,
                                                                crr_num=crr_num,
                                                                EPOCH=DQN_CG_RA_epoch, ITERATION=80, learningrate=0.01,
                                                                batchsize=32,
                                                                mode='pu', model=default_model,
                                                                game_fail_num=game_fail_num,
                                                                sum_rate_type=sum_rate_type, show=show_picture)


    print('\nSimulate_mode=63: Fig3 DQN-RA_CG   CRR sum_rate fairness  reward  \n')
    Performace.compare_dqn_random_game_sum_rate_fairness_reward(pu_num=pu_num_compare, su_num=su_num_vary, crr_num=[10, 25, 40],
                                                                EPOCH=DQN_CG_RA_epoch, ITERATION=80, learningrate=0.01,
                                                                batchsize=32,
                                                                mode='crr', model=default_model,game_fail_num=game_fail_num,
                                                                sum_rate_type=sum_rate_type,show=show_picture)


    print('Simulate_mode=64: Fig3 DQN-RA_CG   CRR sum_rate fairness  reward  \n')
    Performace.compare_dqn_random_game_sum_rate_fairness_reward(pu_num=pu_num_compare, su_num=su_num_vary, crr_num=crr_num,
                                                                EPOCH=DQN_CG_RA_epoch, ITERATION=80, learningrate=0.01,
                                                                batchsize=32,
                                                                mode='power', model=default_model,power_num=[5,10,15],
                                                                game_fail_num=game_fail_num,
                                                                sum_rate_type=sum_rate_type,show=show_picture)

    #
    print('Simulate_mode=7: Fig7 DQN-RA_CG reward different MRQ 对比  \n')
    Performace.compare_dqn_random_game_reward_with_different_mrq(min_rate_mat=[0.005, 0.010, 0.015, 0.020, 0.025,0.030,0.035,0.04],
                                                                 pu_num=pu_num_compare, su_num=su_num_compare, crr_num=crr_num,
                                                                 EPOCH=DQN_CG_RA_epoch, ITERATION=80, learningrate=0.01,
                                                                 batchsize=32, mode='su', model=default_model,
                                                                 game_fail_num=game_fail_num,
                                                                 sum_rate_type=sum_rate_type,show=show_picture)

    print('=====================End===========================')