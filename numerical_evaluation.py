import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.io as scio


class Performace(object):
    def __init__(self, parameters, cchn, Model_train):
        self.simulation_result_path = parameters.simulation_results_path
        try:
            os.mkdir(self.simulation_result_path+'png/')
            os.mkdir(self.simulation_result_path + 'pdf/')
        except Exception:
            pass
        self.figure_format=parameters.figure_format
        self.weight_reward = parameters.weight_reward
        self.primary_rate_min = parameters.primary_rate_min
        self.model_train = Model_train
        self.cchn = cchn
        self.display = ['-or', '->b', '-sg', '-<y', '-dm', '-pc', '--hb']
        self.model = ['nature', 'double', 'dueling', 'double_dueling', 'noisy', 'noisy_double']


    def sum_rate_varying_su(self, su_num, pu_num=3, crr_num=5, EPOCH=2, ITERATION=80, learningrate=0.01, batchsize=8,
                            model='double_dueling'):
        sum_rate_su1 = []
        sum_rate_su2 = []
        iter = []
        loop = su_num[1] - su_num[0] + 1
        for i in range(loop):
            SU_NUM = su_num[0] + i
            self.cchn.reset_network()
            batchsize = 8
            metrics, Sum_rate_PU1, Sum_rate_SU1, Fairness = self.model_train.multi_agent(cchn=self.cchn, pu_num=pu_num,
                                                                                         su_num=SU_NUM, crr_num=crr_num,
                                                                                         EPOCH=EPOCH,
                                                                                         ITERATION=ITERATION,
                                                                                         learningrate=learningrate,
                                                                                         batchsize=batchsize,
                                                                                         inform='-1', model=model)
            batchsize = 16
            metrics, Sum_rate_PU2, Sum_rate_SU2, Fairness = self.model_train.multi_agent(cchn=self.cchn, pu_num=pu_num,
                                                                                         su_num=SU_NUM, crr_num=crr_num,
                                                                                         EPOCH=EPOCH,
                                                                                         ITERATION=ITERATION,
                                                                                         learningrate=learningrate,
                                                                                         batchsize=batchsize,
                                                                                         inform='-1', model=model)

            sum_rate_su1.append(Sum_rate_SU1 / pow(10, 6))
            sum_rate_su2.append(Sum_rate_SU2 / pow(10, 6))
            iter.append(SU_NUM)

        plt.figure(figsize=(8, 5))
        plt.xlabel(r'The number of SUs', fontsize=15)
        plt.ylabel(r'Sum-rate of SUs (Mnats/s)', fontsize=15)
        plt.plot(iter, sum_rate_su1, '->b', MarkerSize=1, label=u'batch size= 8')
        plt.plot(iter, sum_rate_su2, '-or', MarkerSize=1, label=u'batch size= 16')
        # plt.xticks(iter)
        plt.grid()  # 生成网格
        # plt.plot(metrics['epoch'], metrics['sum_rate_su'], '-or', MarkerSize=1, label=u'SU')
        plt.legend(fontsize=12)
        plt.savefig(self.simulation_result_path +
                    'Fig  differ_batch' + ' SU-' + str(su_num[0]) + ' --' + str(su_num[1]) + ' model-' + model + '.png',
                    dpi=400, bbox_inches='tight')

        # plt.show()

    def compare_dqn_random_game_sum_rate_fairness_reward(self, pu_num=3, su_num=3, crr_num=5, EPOCH=2, ITERATION=80,
                                learningrate=0.01,
                                batchsize=8, mode='su', model='double_dueling',power_num=[2,4,6],game_fail_num=200,sum_rate_type='su',show=False):

        results={'user_number':[],'sum_rate_su_DQN':[],'sum_rate_su_RA':[],'sum_rate_su_CG':[],
                 'sum_rate_pu_DQN':[],'sum_rate_pu_RA':[],'sum_rate_pu_CG':[],
                 'sum_rate_DQN':[],'sum_rate_RA':[],'sum_rate_CG':[],
                 'fairness_DQN':[],'fairness_RA':[],'fairness_CG':[],
                 'reward_DQN':[],'reward_RA':[],'reward_CG':[],
                 }

        # loop = user_range[1] - user_range[0] + 1
        pu_num_range = pu_num
        su_num_range = su_num
        crr_num_range = crr_num
        # user_range = su_num_range

        if mode == 'su':
            loop = len(su_num_range)
            user_range=su_num_range

        if mode == 'pu':
            loop = len(pu_num_range)
            user_range=pu_num_range

        if mode == 'crr':
            loop=len(su_num_range)
            user_range=su_num_range

        if mode == 'power':
            loop=len(su_num_range)
            user_range=su_num_range

        for i in range(loop):
            # usr = user_range[0] + i
            # if mode == 'su':
            #     su_num = user_range[0] + i
            # if mode == 'pu':
            #     pu_num = user_range[0] + i
            # if mode == 'crr':
            #     crr_num = user_range[0] + i
            if mode == 'su':
                su_num = np.array(su_num_range)[i]
                usr=su_num

            if mode == 'pu':
                pu_num = np.array(pu_num_range)[i]
                usr=pu_num

            if mode == 'crr':
                su_num = np.array(su_num_range)[i]
                usr=su_num

            if mode == 'power':
                su_num = np.array(su_num_range)[i]
                usr=su_num

            self.cchn.reset_network()


            if mode == 'su' or mode == 'pu' :
                metrics2, Sum_rate_PU1, Sum_rate_SU1, Fairness1, reward_old1 = \
                    self.model_train.multi_agent_vary_model(cchn=self.cchn, pu_num=pu_num, su_num=su_num,
                                                            crr_num=crr_num_range,
                                                            EPOCH=EPOCH, ITERATION=ITERATION, learningrate=learningrate,
                                                            batchsize=batchsize, inform='-1', model=model)

                Sum_rate_PU2, Sum_rate_SU2, Fairness2, random_decision, reward_old2 = self.random_allocation(pu_num=pu_num,
                                                                                                           su_num=su_num,
                                                                                                           crr_num=crr_num)
                Sum_rate_PU3, Sum_rate_SU3, Fairness3, reward_old3 = self.game_allocation(initdecision=random_decision,
                                                                                        pu_num=pu_num, su_num=su_num,
                                                                                        crr_num=crr_num,
                                                                                        max_failnum=game_fail_num, iteration=10)

                print('reward_old_DQN:', reward_old1,'reward_old_RA:', reward_old2, 'reward_old_CG:', reward_old3)


            if mode=='crr':
                self.cchn.reset_network()
                metrics2, Sum_rate_PU1, Sum_rate_SU1, Fairness1, reward_old1 = \
                    self.model_train.multi_agent_vary_model(cchn=self.cchn, pu_num=pu_num, su_num=su_num,
                                                            crr_num=crr_num_range[0],
                                                            EPOCH=EPOCH, ITERATION=ITERATION, learningrate=learningrate,
                                                            batchsize=batchsize, inform='-1', model=model)
                self.cchn.reset_network()
                metrics2, Sum_rate_PU2, Sum_rate_SU2, Fairness2, reward_old2 = \
                    self.model_train.multi_agent_vary_model(cchn=self.cchn, pu_num=pu_num, su_num=su_num,
                                                            crr_num=crr_num_range[1],
                                                            EPOCH=EPOCH, ITERATION=ITERATION, learningrate=learningrate,
                                                            batchsize=batchsize, inform='-1', model=model)
                self.cchn.reset_network()
                metrics2, Sum_rate_PU3, Sum_rate_SU3, Fairness3, reward_old3 = \
                    self.model_train.multi_agent_vary_model(cchn=self.cchn, pu_num=pu_num, su_num=su_num,
                                                            crr_num=crr_num_range[2],
                                                            EPOCH=EPOCH, ITERATION=ITERATION, learningrate=learningrate,
                                                            batchsize=batchsize, inform='-1', model=model)



            if mode=='power':
                self.cchn.reset_network()
                self.cchn.reset_network_powerset(power_num[0])
                metrics2, Sum_rate_PU1, Sum_rate_SU1, Fairness1, reward_old1 = \
                    self.model_train.multi_agent_vary_model(cchn=self.cchn, pu_num=pu_num, su_num=su_num,
                                                            crr_num=crr_num,
                                                            EPOCH=EPOCH, ITERATION=ITERATION, learningrate=learningrate,
                                                            batchsize=batchsize, inform='-1', model=model)
                self.cchn.reset_network()
                self.cchn.reset_network_powerset(power_num[1])
                metrics2, Sum_rate_PU2, Sum_rate_SU2, Fairness2, reward_old2 = \
                    self.model_train.multi_agent_vary_model(cchn=self.cchn, pu_num=pu_num, su_num=su_num,
                                                            crr_num=crr_num,
                                                            EPOCH=EPOCH, ITERATION=ITERATION, learningrate=learningrate,
                                                            batchsize=batchsize, inform='-1', model=model)
                self.cchn.reset_network()
                self.cchn.reset_network_powerset(power_num[2])
                metrics2, Sum_rate_PU3, Sum_rate_SU3, Fairness3, reward_old3 = \
                    self.model_train.multi_agent_vary_model(cchn=self.cchn, pu_num=pu_num, su_num=su_num,
                                                            crr_num=crr_num,
                                                            EPOCH=EPOCH, ITERATION=ITERATION, learningrate=learningrate,
                                                            batchsize=batchsize, inform='-1', model=model)

            results['user_number'].append(usr)

            '''1:sum-rate of SUs'''
            results['sum_rate_su_DQN'].append(Sum_rate_SU1/ pow(10, 6))
            results['sum_rate_su_RA'].append(Sum_rate_SU2/ pow(10, 6))
            results['sum_rate_su_CG'].append(Sum_rate_SU3/ pow(10, 6))


            '''2:sum-rate of PUs'''
            results['sum_rate_pu_DQN'].append(Sum_rate_PU1/ pow(10, 6))
            results['sum_rate_pu_RA'].append(Sum_rate_PU2/ pow(10, 6))
            results['sum_rate_pu_CG'].append(Sum_rate_PU3/ pow(10, 6))

            '''3:sum-rate of PUs and SUs'''
            results['sum_rate_DQN'].append(Sum_rate_SU1/ pow(10, 6)+Sum_rate_PU1/ pow(10, 6))
            results['sum_rate_RA'].append(Sum_rate_SU2/ pow(10, 6)+Sum_rate_PU2/ pow(10, 6))
            results['sum_rate_CG'].append(Sum_rate_SU3/ pow(10, 6)+Sum_rate_PU3/ pow(10, 6))

            '''4:fairness'''
            results['fairness_DQN'].append(Fairness1)
            results['fairness_RA'].append(Fairness2)
            results['fairness_CG'].append(Fairness3)

            '''5:reward'''
            results['reward_DQN'].append(reward_old1)
            results['reward_RA'].append(reward_old2)
            results['reward_CG'].append(reward_old3)

            scio.savemat(self.simulation_result_path+'data/Fig6_'+mode+'.mat',
         {'user_number': results['user_number'],
          'sum_rate_su_DQN': results['sum_rate_su_DQN'], 'sum_rate_su_RA': results['sum_rate_su_RA'], 'sum_rate_su_CG': results['sum_rate_su_CG'],
          'sum_rate_pu_DQN': results['sum_rate_pu_DQN'], 'sum_rate_pu_RA': results['sum_rate_pu_RA'],  'sum_rate_pu_CG': results['sum_rate_pu_CG'],
          'sum_rate_DQN': results['sum_rate_DQN'], 'sum_rate_RA': results['sum_rate_RA'], 'sum_rate_CG': results['sum_rate_CG'],
          'fairness_DQN': results['fairness_DQN'], 'fairness_RA': results['fairness_RA'], 'fairness_CG': results['fairness_CG'],
          'reward_DQN': results['reward_DQN'], 'reward_RA': results['reward_RA'], 'reward_CG': results['reward_CG'],
          })

        # print('results[reward_DQN]:',results['reward_DQN'])
        # print('results[reward_RA]:', results['reward_RA'])
        # print('results[reward_CG]:', results['reward_CG'])
        # print('results[reward]', results['reward'][0])
        # print('results[reward]', results['reward'][0])
        if show==True:
            if  mode == 'su' or mode == 'pu':
                lable = [u'DQN allocation', u'Random allocation', u'Coalitional game']
            if mode=='crr':
                lable = [u'CR-routers='+str(crr_num_range[0]), u'CR-routers='+str(crr_num_range[1]), u'CR-routers='+str(crr_num_range[2])]

            if mode == 'power':
                lable = [u'Power actions=' + str(power_num[0]), u'Power actions=' + str(power_num[1]),
                         u'Power actions=' + str(power_num[2])]

            if 'su' in sum_rate_type:
                self.plot_simulation_result(user_range,model=model,results=results,user_mode=mode,plot_mode='sum_rate_su',
                                            MarkerSize=8,lable=lable,crr_num_range = crr_num,
                                           )
                self.plot_simulation_result(user_range, model=model,results=results, user_mode=mode, plot_mode='fairness',
                                            MarkerSize=8,
                                            lable=lable,crr_num_range = crr_num,
                                            figure_format='.png')

            if 'pu' in sum_rate_type:
                self.plot_simulation_result(user_range,model=model,results=results,user_mode=mode,plot_mode='sum_rate_pu',
                                            MarkerSize=8,lable=lable,crr_num_range = crr_num,
                                            figure_format='.png')

            if 'sum' in sum_rate_type:
                self.plot_simulation_result(user_range,model=model,results=results,user_mode=mode,plot_mode='sum_rate',
                                            MarkerSize=8,lable=lable,crr_num_range = crr_num,
                                            figure_format='.png')


            self.plot_simulation_result(user_range, model=model,results=results, user_mode=mode, plot_mode='reward',
                                        MarkerSize=8,
                                        lable=lable,crr_num_range = crr_num,
                                        figure_format='.png')

    def plot_simulation_result(self,user_range,model,results,user_mode,plot_mode,MarkerSize,lable,
                               figure_format='.png',figure_name='Fig6 DQN_CG_RD ',crr_num_range = [10,15,20],):

        # results={'user_number':[],'sum_rate_su':[],'sum_rate_pu':[],'sum_rate':[],'fairness':[],'reward':[],}
        loop = len(user_range)#user_range#[1] - user_range[0] + 1
        plt.figure(figsize=(8, 5))
        if user_mode == 'su':
            plt.xlabel(r'The number of IoT links', fontsize=15)
            save_inform = ' SU-' + str(user_range[0]) + '--' + str(user_range[-1]) +' '+plot_mode +'-' + model

        if user_mode == 'pu':
            plt.xlabel(r'The number of PU links', fontsize=15)
            save_inform = ' PU-' + str(user_range[0]) + '--' + str(user_range[-1]) + ' '+plot_mode +'-' + model

        if user_mode == 'crr':
            plt.xlabel(r'The number of IoT links', fontsize=15)
            save_inform = ' CRR-' + str(user_range[0]) + '--' + str(user_range[-1]) + ' '+plot_mode +'-' + model

        if user_mode == 'power':
            plt.xlabel(r'The number of IoT links', fontsize=15)
            save_inform = ' Power-SU-' + str(user_range[0]) + '--' + str(user_range[-1]) +' '+plot_mode + '-' + model

        if plot_mode=='sum_rate_su':
            plt.ylabel(r'Sum-rate of IoT links (Mnats/s)', fontsize=15)
            plt.plot(results['user_number'], results['sum_rate_su_DQN'],self.display[0], MarkerSize=MarkerSize, label=lable[0])
            plt.plot(results['user_number'], results['sum_rate_su_RA'],self.display[1], MarkerSize=MarkerSize, label=lable[1])
            plt.plot(results['user_number'], results['sum_rate_su_CG'],self.display[2], MarkerSize=MarkerSize, label=lable[2])

        if plot_mode=='sum_rate_pu':
            plt.ylabel(r'Sum-rate of PU links (Mnats/s)', fontsize=15)
            plt.plot(results['user_number'], results['sum_rate_pu_DQN'],self.display[0], MarkerSize=MarkerSize, label=lable[0])
            plt.plot(results['user_number'], results['sum_rate_pu_RA'],self.display[1], MarkerSize=MarkerSize, label=lable[1])
            plt.plot(results['user_number'], results['sum_rate_pu_CG'],self.display[2], MarkerSize=MarkerSize, label=lable[2])

        if plot_mode=='sum_rate':
            plt.ylabel(r'Sum-rate (Mnats/s)', fontsize=15)
            # plt.ylabel(r'Sum-rate of PU links (Mnats/s)', fontsize=15)
            plt.plot(results['user_number'], results['sum_rate_DQN'],self.display[0], MarkerSize=MarkerSize, label=lable[0])
            plt.plot(results['user_number'], results['sum_rate_RA'],self.display[1], MarkerSize=MarkerSize, label=lable[1])
            plt.plot(results['user_number'], results['sum_rate_CG'],self.display[2], MarkerSize=MarkerSize, label=lable[2])

        if plot_mode == 'fairness':
            plt.ylabel(r'Fairness', fontsize=15)
            # plt.ylabel(r'Sum-rate of PU links (Mnats/s)', fontsize=15)
            plt.plot(results['user_number'], results['fairness_DQN'],self.display[0], MarkerSize=MarkerSize, label=lable[0])
            plt.plot(results['user_number'], results['fairness_RA'],self.display[1], MarkerSize=MarkerSize, label=lable[1])
            plt.plot(results['user_number'], results['fairness_CG'],self.display[2], MarkerSize=MarkerSize, label=lable[2])

        if plot_mode == 'reward':
            plt.ylabel(r'Reward', fontsize=15)
            plt.plot(results['user_number'], results['reward_DQN'],self.display[0], MarkerSize=MarkerSize, label=lable[0])
            plt.plot(results['user_number'], results['reward_RA'],self.display[1], MarkerSize=MarkerSize, label=lable[1])
            plt.plot(results['user_number'], results['reward_CG'],self.display[2], MarkerSize=MarkerSize, label=lable[2])

        plt.xticks(results['user_number'])
        plt.grid()  # 生成网格
        plt.legend(fontsize=12)
        plt.savefig(self.simulation_result_path + 'png/'+ figure_name+ save_inform +figure_format, dpi=600, bbox_inches='tight')
        plt.savefig(self.simulation_result_path + 'pdf/ '+figure_name+ save_inform+self.figure_format , dpi=600, bbox_inches='tight')


    def compare_dqn_random_game(self, user_range, pu_num=3, su_num=3, crr_num=5, EPOCH=2, ITERATION=80,
                                learningrate=0.01,
                                batchsize=8, mode='su', model='double_dueling'):
        sum_rate_su1 = []
        sum_rate_su2 = []
        sum_rate_su3 = []
        iter = []
        loop = user_range[1] - user_range[0] + 1

        for i in range(loop):
            usr = user_range[0] + i
            if mode == 'su':
                su_num = user_range[0] + i
            else:
                pu_num = user_range[0] + i
            self.cchn.reset_network()

            metrics2, Sum_rate_PU1, Sum_rate_SU1, Fairness, reward_old = self.model_train.multi_agent_vary_model(
                                                                                        cchn=self.cchn,
                                                                                        pu_num=pu_num,
                                                                                        su_num=su_num,
                                                                                        crr_num=crr_num,
                                                                                        EPOCH=EPOCH,
                                                                                        ITERATION=ITERATION,
                                                                                        learningrate=learningrate,
                                                                                        batchsize=batchsize,
                                                                                        inform='-1',
                                                                                        model=model)

            Sum_rate_PU2, Sum_rate_SU2, Fairness, random_decision, reward_old = self.random_allocation(pu_num=pu_num,su_num=su_num,crr_num=crr_num)
            Sum_rate_PU3, Sum_rate_SU3, Fairness, reward_old = self.game_allocation(initdecision=random_decision,
                                                                                    pu_num=pu_num, su_num=su_num,crr_num=crr_num,
                                                                                    max_failnum=2000, iteration=10)

            sum_rate_su1.append(Sum_rate_SU1 / pow(10, 6))
            sum_rate_su2.append(Sum_rate_SU2 / pow(10, 6))
            sum_rate_su3.append(Sum_rate_SU3 / pow(10, 6))
            iter.append(usr)

            lable = [u'DQN allocation', u'Random allocation', u'Coalitional game']
            MarkerSize = 8
            plt.figure(figsize=(8, 5))
            if mode == 'su':
                plt.xlabel(r'The number of SUs', fontsize=15)
                save_inform = ' SU-' + str(user_range[0]) + '--' + str(user_range[1]) + ' model-' + model
            else:
                plt.xlabel(r'The number of PUs', fontsize=15)
                save_inform = ' PU-' + str(user_range[0]) + '--' + str(user_range[1]) + ' model-' + model

            plt.ylabel(r'Sum-rate of SUs (Mnats/s)', fontsize=15)

            plt.plot(iter, sum_rate_su1, self.display[0], MarkerSize=MarkerSize, label=lable[0])
            plt.plot(iter, sum_rate_su2, self.display[1], MarkerSize=MarkerSize, label=lable[1])
            plt.plot(iter, sum_rate_su3, self.display[2], MarkerSize=MarkerSize, label=lable[2])
            plt.xticks(iter)
            plt.grid()  # 生成网格

            # plt.plot(metrics['epoch'], metrics['sum_rate_su'], '-or', MarkerSize=1, label=u'SU')
            plt.legend(fontsize=12)

            plt.savefig(self.simulation_result_path + 'Fig3 DQN_RA_CG sum-rate ' + save_inform + '.pdf', dpi=600,
                        bbox_inches='tight')

    def compare_dqn_random_game_fairness(self, user_range, pu_num=3, su_num=3, crr_num=5, EPOCH=2, ITERATION=80,
                                         learningrate=0.01,
                                         batchsize=8, mode='su', model='double_dueling'):
        sum_rate_su1 = []
        sum_rate_su2 = []
        sum_rate_su3 = []
        iter = []
        loop = user_range[1] - user_range[0] + 1

        for i in range(loop):
            usr = user_range[0] + i
            if mode == 'su':
                su_num = user_range[0] + i
            else:
                pu_num = user_range[0] + i
            self.cchn.reset_network()
            # model = ['nature', 'double', 'dueling', 'double_dueling', 'noisy', 'noisy_double']
            # m=3
            metrics, Sum_rate_PU1, Sum_rate_SU1, Fairness1, reward_old = self.model_train.multi_agent_vary_model(
                cchn=self.cchn,
                pu_num=pu_num,
                su_num=su_num,
                crr_num=crr_num,
                EPOCH=EPOCH,
                ITERATION=ITERATION,
                learningrate=learningrate,
                batchsize=batchsize,
                inform='-1',
                model=model)

            Sum_rate_PU2, Sum_rate_SU2, Fairness2, random_decision, reward_old = self.random_allocation(pu_num=pu_num,
                                                                                                        su_num=su_num,
                                                                                                        crr_num=crr_num)
            Sum_rate_PU3, Sum_rate_SU3, Fairness3, reward_old = self.game_allocation(initdecision=random_decision,
                                                                                     pu_num=pu_num, su_num=su_num,
                                                                                     crr_num=crr_num,
                                                                                     max_failnum=2000, iteration=10)

            # print(Sum_rate_SU1+Sum_rate_PU1,Sum_rate_SU2+Sum_rate_PU2,Sum_rate_SU3+Sum_rate_PU3)
            # print('Fairness_DQN:', Fairness1,'Fairness_RA:', Fairness2,'Fairness_CG:', Fairness3)
            sum_rate_su1.append(Fairness1)
            sum_rate_su2.append(Fairness2)
            sum_rate_su3.append(Fairness3)
            iter.append(usr)

            lable = [u'DRL-based allocation', u'Random allocation', u'Game allocation']
            MarkerSize = 8
            plt.figure(figsize=(8, 5))
            if mode == 'su':
                plt.xlabel(r'The number of SUs', fontsize=15)
                save_inform = ' SU-' + str(user_range[0]) + '--' + str(user_range[1]) + ' model-' + model
            else:
                plt.xlabel(r'The number of PUs', fontsize=15)
                save_inform = ' PU-' + str(user_range[0]) + '--' + str(user_range[1]) + ' model' + model

            plt.ylabel(r'Fairness', fontsize=15)

            plt.plot(iter, sum_rate_su1, self.display[0], MarkerSize=MarkerSize, label=lable[0])
            plt.plot(iter, sum_rate_su2, self.display[1], MarkerSize=MarkerSize, label=lable[1])
            plt.plot(iter, sum_rate_su3, self.display[2], MarkerSize=MarkerSize, label=lable[2])
            plt.xticks(iter)
            plt.grid()  # 生成网格

            # plt.plot(metrics['epoch'], metrics['sum_rate_su'], '-or', MarkerSize=1, label=u'SU')
            plt.legend(fontsize=12)

            plt.savefig(self.simulation_result_path + 'Fig4 DQN_RA_CG Fairness ' + save_inform + '.pdf', dpi=600,
                        bbox_inches='tight')

    def compare_dqn_random_game_reward(self, user_range, pu_num=3, su_num=3, crr_num=5, EPOCH=2, ITERATION=80,
                                       learningrate=0.01,
                                       batchsize=8, mode='su', model='double_dueling'):
        sum_rate_su1 = []
        sum_rate_su2 = []
        sum_rate_su3 = []
        iter = []
        loop = user_range[1] - user_range[0] + 1

        for i in range(loop):
            usr = user_range[0] + i
            if mode == 'su':
                su_num = user_range[0] + i
            else:
                pu_num = user_range[0] + i
            self.cchn.reset_network()
            # model = ['nature', 'double', 'dueling', 'double_dueling', 'noisy', 'noisy_double']
            # m=3
            metrics, Sum_rate_PU1, Sum_rate_SU1, Fairness1, reward1 = self.model_train.multi_agent_vary_model(
                cchn=self.cchn,
                pu_num=pu_num,
                su_num=su_num,
                crr_num=crr_num,
                EPOCH=EPOCH,
                ITERATION=ITERATION,
                learningrate=learningrate,
                batchsize=batchsize,
                inform='-1',
                model=model, pu_min_rate=20000)

            Sum_rate_PU2, Sum_rate_SU2, Fairness2, random_decision, reward2 = self.random_allocation(pu_num=pu_num,
                                                                                                     su_num=su_num,
                                                                                                     crr_num=crr_num,
                                                                                                     pu_min_rate=20000)

            Sum_rate_PU3, Sum_rate_SU3, Fairness3, reward3 = self.game_allocation(initdecision=random_decision,
                                                                                  pu_num=pu_num,
                                                                                  su_num=su_num, crr_num=crr_num,
                                                                                  max_failnum=2000, iteration=10,
                                                                                  pu_min_rate=20000)

            # print(Sum_rate_SU1+Sum_rate_PU1,Sum_rate_SU2+Sum_rate_PU2,Sum_rate_SU3+Sum_rate_PU3)
            # print('Fairness_DQN:', Fairness1,'Fairness_RA:', Fairness2,'Fairness_CG:', Fairness3)
            sum_rate_su1.append(reward1)
            sum_rate_su2.append(reward2)
            sum_rate_su3.append(reward3)
            iter.append(usr)

            lable = [u'DRL-based allocation', u'Random allocation', u'Game allocation']
            MarkerSize = 8
            plt.figure(figsize=(8, 5))
            if mode == 'su':
                plt.xlabel(r'The number of SUs', fontsize=15)
                save_inform = ' SU-' + str(user_range[0]) + '--' + str(user_range[1]) + ' model-' + model
            else:
                plt.xlabel(r'The number of PUs', fontsize=15)
                save_inform = ' PU-' + str(user_range[0]) + '--' + str(user_range[1]) + ' model' + model

            plt.ylabel(r'Reward ', fontsize=15)

            plt.plot(iter, sum_rate_su1, self.display[0], MarkerSize=MarkerSize, label=lable[0])
            plt.plot(iter, sum_rate_su2, self.display[1], MarkerSize=MarkerSize, label=lable[1])
            plt.plot(iter, sum_rate_su3, self.display[2], MarkerSize=MarkerSize, label=lable[2])
            plt.xticks(iter)
            plt.grid()  # 生成网格

            # plt.plot(metrics['epoch'], metrics['sum_rate_su'], '-or', MarkerSize=1, label=u'SU')
            plt.legend(fontsize=12)

            plt.savefig(self.simulation_result_path + 'Fig5 DQN_RA_CG reward ' + save_inform + '.pdf', dpi=600,
                        bbox_inches='tight')

    def compare_dqn_random_game_reward_with_different_mrq(self, min_rate_mat, pu_num=3, su_num=3, crr_num=5, EPOCH=2,
                                                          ITERATION=80,
                                                          learningrate=0.01,
                                                          batchsize=8, mode='su', model='double_dueling',game_fail_num=200,sum_rate_type='su',show=False):
        sum_rate_su1 = []
        sum_rate_su2 = []
        sum_rate_su3 = []

        sum_rate_pu1 = []
        sum_rate_pu2 = []
        sum_rate_pu3 = []

        sum_rate_supu1 = []
        sum_rate_supu2 = []
        sum_rate_supu3 = []

        fairness1 = []
        fairness2 = []
        fairness3 = []

        reward1 = []
        reward2 = []
        reward3 = []


        iter = []

        for i in range(len(min_rate_mat)):
            pu_min_rate = min_rate_mat[i] * pow(10, 6)

            self.cchn.reset_network()

            metrics, Sum_rate_PU1, Sum_rate_SU1, Fairness1, reward_1 = self.model_train.multi_agent_vary_model(
                                                                                                cchn=self.cchn,
                                                                                                pu_num=pu_num,
                                                                                                su_num=su_num,
                                                                                                crr_num=crr_num,
                                                                                                EPOCH=EPOCH,
                                                                                                ITERATION=ITERATION,
                                                                                                learningrate=learningrate,
                                                                                                batchsize=batchsize,
                                                                                                inform='-1',
                                                                                                model=model, pu_min_rate=pu_min_rate)

            Sum_rate_PU2, Sum_rate_SU2, Fairness2, random_decision, reward_2 = self.random_allocation(pu_num=pu_num,
                                                                                                     su_num=su_num,
                                                                                                     crr_num=crr_num,
                                                                                                     pu_min_rate=pu_min_rate)

            Sum_rate_PU3, Sum_rate_SU3, Fairness3, reward_3 = self.game_allocation(initdecision=random_decision,
                                                                                  pu_num=pu_num,
                                                                                  su_num=su_num, crr_num=crr_num,
                                                                                  max_failnum=game_fail_num, iteration=10,
                                                                                  pu_min_rate=pu_min_rate)

            sum_rate_su1.append(Sum_rate_SU1/pow(10,6))
            sum_rate_su2.append(Sum_rate_SU2/pow(10,6))
            sum_rate_su3.append(Sum_rate_SU3/pow(10,6))

            sum_rate_pu1.append(Sum_rate_PU1/pow(10,6))
            sum_rate_pu2.append(Sum_rate_PU2/pow(10,6))
            sum_rate_pu3.append(Sum_rate_PU3/pow(10,6))

            sum_rate_supu1.append(Sum_rate_SU1/pow(10,6)+Sum_rate_PU1/pow(10,6))
            sum_rate_supu2.append(Sum_rate_SU2/pow(10,6)+Sum_rate_PU2/pow(10,6))
            sum_rate_supu3.append(Sum_rate_SU3/pow(10,6)+Sum_rate_PU3/pow(10,6))

            fairness1.append(Fairness1)
            fairness2.append(Fairness2)
            fairness3.append(Fairness3)

            reward1.append(reward_1)
            reward2.append(reward_2)
            reward3.append(reward_3)


            iter.append(pu_min_rate / pow(10, 6))

        scio.savemat(self.simulation_result_path+'data/Fig7_'+'MRQ'+'.mat',
     {'user_number': iter,
      'sum_rate_su_DQN': sum_rate_su1, 'sum_rate_su_RA': sum_rate_su2, 'sum_rate_su_CG': sum_rate_su3,
      'sum_rate_pu_DQN': sum_rate_pu1, 'sum_rate_pu_RA': sum_rate_pu2,  'sum_rate_pu_CG': sum_rate_pu3,
      'sum_rate_DQN': sum_rate_supu1, 'sum_rate_RA': sum_rate_supu2, 'sum_rate_CG': sum_rate_supu3,
      'fairness_DQN': fairness1, 'fairness_RA': fairness2, 'fairness_CG': fairness3,
      'reward_DQN': reward1, 'reward_RA': reward2, 'reward_CG': reward3,
      })

        if show==True:

            lable = [u'DRL-based allocation', u'Random allocation', u'Game allocation']

            MarkerSize = 8

            if 'su' in sum_rate_type:
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'The MRQ of PU links  (Mnat/s) ', fontsize=15)
                plt.ylabel(r'Sum-rate of IoT links (Mnat/s) ', fontsize=15)
                plt.plot(iter, sum_rate_su1, self.display[0], MarkerSize=MarkerSize, label=lable[0])
                plt.plot(iter, sum_rate_su2, self.display[1], MarkerSize=MarkerSize, label=lable[1])
                plt.plot(iter, sum_rate_su3, self.display[2], MarkerSize=MarkerSize, label=lable[2])
                plt.xticks(iter)
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' + 'Fig7 MRQ_sum_rate_su_' +'epoch=' + str(EPOCH) + '-' + model + '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/' + 'Fig7 MRQ_sum_rate_su_' + 'epoch=' + str(EPOCH)  + '-' + model + '.pdf',
                            dpi=600, bbox_inches='tight')

                plt.figure(figsize=(8, 5))
                plt.xlabel(r'The MRQ of PU links  (Mnat/s) ', fontsize=15)
                plt.ylabel(r'Fairness ', fontsize=15)
                plt.plot(iter, fairness1, self.display[0], MarkerSize=MarkerSize, label=lable[0])
                plt.plot(iter, fairness2, self.display[1], MarkerSize=MarkerSize, label=lable[1])
                plt.plot(iter, fairness3, self.display[2], MarkerSize=MarkerSize, label=lable[2])
                plt.xticks(iter)
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' + 'Fig7 MRQ_fairness_' + 'epoch=' + str(
                    EPOCH) + '-' + model + '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/' + 'Fig7 MRQ_fairness_' + 'epoch=' + str(
                    EPOCH) + '-' + model + '.pdf',
                            dpi=600, bbox_inches='tight')

            if 'pu' in sum_rate_type:
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'The MRQ of PU links (Mnat/s) ', fontsize=15)
                plt.ylabel(r'Sum-rate of PU links (Mnat/s) ', fontsize=15)
                plt.plot(iter, sum_rate_pu1, self.display[0], MarkerSize=MarkerSize, label=lable[0])
                plt.plot(iter, sum_rate_pu2, self.display[1], MarkerSize=MarkerSize, label=lable[1])
                plt.plot(iter, sum_rate_pu3, self.display[2], MarkerSize=MarkerSize, label=lable[2])
                plt.xticks(iter)
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' + 'Fig7 MRQ_sum_rate_pu_' +'epoch=' + str(EPOCH) + '-' + model + '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/' + 'Fig7 MRQ_sum_rate_pu_' + 'epoch=' + str(EPOCH)  + '-' + model + '.pdf',
                            dpi=600, bbox_inches='tight')

            if 'sum' in sum_rate_type:
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'The MRQ of PU links (Mnat/s) ', fontsize=15)
                plt.ylabel(r'Sum-rate (Mnat/s) ', fontsize=15)
                plt.plot(iter, sum_rate_supu1, self.display[0], MarkerSize=MarkerSize, label=lable[0])
                plt.plot(iter, sum_rate_supu2, self.display[1], MarkerSize=MarkerSize, label=lable[1])
                plt.plot(iter, sum_rate_supu3, self.display[2], MarkerSize=MarkerSize, label=lable[2])
                plt.xticks(iter)
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' + 'Fig7 MRQ_sum_rate_supu_' +'epoch=' + str(EPOCH) + '-' + model + '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/' + 'Fig7 MRQ_sum_rate_supu_' + 'epoch=' + str(EPOCH)  + '-' + model + '.pdf',
                            dpi=600, bbox_inches='tight')


            plt.figure(figsize=(8, 5))
            plt.xlabel(r'The MRQ of PU links (Mnat/s) ', fontsize=15)
            plt.ylabel(r'Reward ', fontsize=15)
            plt.plot(iter, reward1, self.display[0], MarkerSize=MarkerSize, label=lable[0])
            plt.plot(iter, reward2, self.display[1], MarkerSize=MarkerSize, label=lable[1])
            plt.plot(iter, reward3, self.display[2], MarkerSize=MarkerSize, label=lable[2])
            plt.xticks(iter)
            plt.grid()  # 生成网格
            plt.legend(fontsize=12)
            plt.savefig(self.simulation_result_path + 'png/' + 'Fig7 MRQ_reward_' + 'epoch=' + str(
                EPOCH) + '-' + model + '.png',
                        dpi=600, bbox_inches='tight')
            plt.savefig(self.simulation_result_path + 'pdf/' + 'Fig7 MRQ_reward_' + 'epoch=' + str(
                EPOCH) + '-' + model + '.pdf',
                        dpi=600, bbox_inches='tight')

    def compare_different_dqn_model(self, pu_num=3, su_num=4, crr_num=5,
                                    EPOCH=2, ITERATION=80, learningrate=0.01, batchsize=8, model='double_dueling'):

        self.cchn.reset_network()

        result = []
        for i in range(len(model)):
            metrics2, Sum_rate_PU1, Sum_rate_SU1, Fairness, reward = \
                self.model_train.multi_agent_vary_model(
                                                    cchn=self.cchn,
                                                    pu_num=pu_num,
                                                    su_num=su_num,
                                                    crr_num=crr_num,
                                                    EPOCH=EPOCH,
                                                    ITERATION=ITERATION,
                                                    learningrate=learningrate,
                                                    batchsize=batchsize,
                                                    inform='-1',
                                                    model=model[i])
            result.append(metrics2)
        save_inform = ' epoch-' + str(EPOCH) + ' '
        # print(result[0]['epoch'], np.array(result[0]['sum_rate_su']))

        '''Sum-rate'''
        MarkerSize = 1
        plt.figure(figsize=(8, 5))
        plt.xlabel(r'Epoches', fontsize=15)
        plt.ylabel(r'Sum-rate of IoT links (Mnats/s)', fontsize=15)

        plt.plot(result[0]['epoch'], np.array(result[0]['sum_rate_su']) / pow(10, 6), '->b', MarkerSize=MarkerSize,
                 label=u'Nature DQN')
        plt.plot(result[1]['epoch'], np.array(result[1]['sum_rate_su']) / pow(10, 6), '-sg', MarkerSize=MarkerSize,
                 label=u'Double DQN')
        plt.plot(result[2]['epoch'], np.array(result[2]['sum_rate_su']) / pow(10, 6), '-or', MarkerSize=MarkerSize,
                 label=u'Dueling DQN')
        plt.plot(result[3]['epoch'], np.array(result[3]['sum_rate_su']) / pow(10, 6), '-<y', MarkerSize=MarkerSize,
                 label=u'Double+dueling DQN')
        plt.grid()  # 生成网格
        plt.legend(fontsize=12)
        plt.savefig(self.simulation_result_path + 'png/' + 'Fig1 model-sum_rate' + save_inform + 'png',
                    dpi=600, bbox_inches='tight')
        plt.savefig(self.simulation_result_path + 'pdf/ ' +'Fig1 model-sum_rate ' +  save_inform + '.pdf',
                    dpi=600, bbox_inches='tight')

        '''Reward'''
        plt.figure(figsize=(8, 5))
        plt.xlabel(r'Epoches', fontsize=15)
        plt.ylabel(r'Reward', fontsize=15)
        plt.plot(result[0]['epoch'], np.array(result[0]['reward']), '->b', MarkerSize=MarkerSize,
                 label=u'Nature DQN')
        plt.plot(result[1]['epoch'], np.array(result[1]['reward']) , '-sg', MarkerSize=MarkerSize,
                 label=u'Double DQN')
        plt.plot(result[2]['epoch'], np.array(result[2]['reward']) , '-or', MarkerSize=MarkerSize,
                 label=u'Dueling DQN')
        plt.plot(result[3]['epoch'], np.array(result[3]['reward']) , '-<y', MarkerSize=MarkerSize,
                 label=u'Double+dueling DQN')
        plt.grid()  # 生成网格
        plt.legend(fontsize=12)
        plt.savefig(self.simulation_result_path + 'png/' + 'Fig1 model-reward ' + save_inform + 'png',
                    dpi=600, bbox_inches='tight')
        plt.savefig(self.simulation_result_path + 'pdf/ ' +'Fig1 model-reward ' +  save_inform + '.pdf',
                    dpi=600, bbox_inches='tight')

        '''Fairness'''
        plt.figure(figsize=(8, 5))
        plt.xlabel(r'Epoches', fontsize=15)
        plt.ylabel(r'Fairness', fontsize=15)
        plt.plot(result[0]['epoch'], np.array(result[0]['fairness']) , '->b', MarkerSize=MarkerSize,
                 label=u'Nature DQN')
        plt.plot(result[1]['epoch'], np.array(result[1]['fairness']) , '-sg', MarkerSize=MarkerSize,
                 label=u'Double DQN')
        plt.plot(result[2]['epoch'], np.array(result[2]['fairness']) , '-or', MarkerSize=MarkerSize,
                 label=u'Dueling DQN')
        plt.plot(result[3]['epoch'], np.array(result[3]['fairness']) , '-<y', MarkerSize=MarkerSize,
                 label=u'Double+dueling DQN')
        plt.grid()  # 生成网格
        plt.legend(fontsize=12)
        plt.savefig(self.simulation_result_path + 'png/' + 'Fig1 model-fairness ' + save_inform + 'png',
                    dpi=600, bbox_inches='tight')
        plt.savefig(self.simulation_result_path + 'pdf/ ' +'Fig1 model-fairness ' +  save_inform + '.pdf',
                    dpi=600, bbox_inches='tight')


        '''loss'''
        plt.figure(figsize=(8, 5))
        plt.xlabel(r'Epoches', fontsize=15)
        plt.ylabel(r'Loss', fontsize=15)
        plt.plot(result[0]['epoch'], np.array(result[0]['total_loss']) , '->b', MarkerSize=MarkerSize,
                 label=u'Nature DQN')
        plt.plot(result[1]['epoch'], np.array(result[1]['total_loss']) , '-sg', MarkerSize=MarkerSize,
                 label=u'Double DQN')
        plt.plot(result[2]['epoch'], np.array(result[2]['total_loss']) , '-or', MarkerSize=MarkerSize,
                 label=u'Dueling DQN')
        plt.plot(result[3]['epoch'], np.array(result[3]['total_loss']) , '-<y', MarkerSize=MarkerSize,
                 label=u'Double+dueling DQN')
        plt.grid()  # 生成网格
        plt.legend(fontsize=12)
        plt.savefig(self.simulation_result_path + 'png/' + 'Fig1 model-total_loss ' + save_inform + 'png',
                    dpi=600, bbox_inches='tight')
        plt.savefig(self.simulation_result_path + 'pdf/ ' +'Fig1 model-total_loss ' +  save_inform + '.pdf',
                    dpi=600, bbox_inches='tight')

    def compare_different_dqn_model1(self, pu_num=3, su_num=4, crr_num=5,
                                    EPOCH=2, ITERATION=80, learningrate=0.01, batchsize=8, model='double_dueling',sum_rate_type='su',show=False):

        self.cchn.reset_network()

        result = []
        for i in range(len(model)):
            metrics2, Sum_rate_PU1, Sum_rate_SU1, Fairness, reward = \
                self.model_train.multi_agent_vary_model(
                    cchn=self.cchn,
                    pu_num=pu_num,
                    su_num=su_num,
                    crr_num=crr_num,
                    EPOCH=EPOCH,
                    ITERATION=ITERATION,
                    learningrate=learningrate,
                    batchsize=batchsize,
                    inform='-1',
                    model=model[i])
            result.append(metrics2)

        scio.savemat(self.simulation_result_path + 'data/Fig1_differ_model'  + '.mat',{'result':result})
        # for i in range(len(model)):
        #     scio.savemat(self.simulation_result_path+'data/Fig1_'+model[i]+'.mat',
        #                  {'epoch':result[i]['epoch'],'sum_rate_su':result[i]['sum_rate_su'],
        #                   'sum_rate_pu':result[i]['sum_rate_pu'],'sum_rate':result[i]['sum_rate'],
        #                   'total_loss':result[i]['total_loss'],
        #                   'fairness': result[i]['fairness'],'reward':result[i]['reward'],}
        #                  
        #                  )

        if show==True:

            save_inform = ' epoch-' + str(EPOCH) + ' '+ model[0]
            # print(result[0]['epoch'], np.array(result[0]['sum_rate_su']))
            MarkerSize = 1


            # lable = [u'Nature DQN', u'Double DQN', u'Dueling DQN', u'Double+dueling DQN',u'Noisy DQN',u'Noisy+double DQN']

            if 'su' in sum_rate_type:
                '''Sum-rate_su'''
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Sum-rate of IoT links (Mnats/s)', fontsize=15)

                if len(model)==1:
                    plt.plot(result[0]['epoch'], np.array(result[0]['sum_rate_su']) / pow(10, 6), self.display[0],
                             MarkerSize=MarkerSize,
                             label=model[0])
                else:
                    for i in range(len(model)):
                        plt.plot(result[i]['epoch'], np.array(result[i]['sum_rate_su']) / pow(10, 6), self.display[i], MarkerSize=MarkerSize,
                                 label=model[i])

                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' + 'Fig1 model-sum_rate_su' + save_inform +'.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/ ' + 'Fig1 model-sum_rate_su ' + save_inform + '.pdf',
                            dpi=600, bbox_inches='tight')

                '''Fairness'''
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Fairness', fontsize=15)
                if len(model)==1:
                    plt.plot(result[0]['epoch'], np.array(result[0]['fairness']), self.display[0],
                             MarkerSize=MarkerSize,
                             label=model[0])
                else:
                    for i in range(len(model)):
                        plt.plot(result[i]['epoch'], np.array(result[i]['fairness']), self.display[i], MarkerSize=MarkerSize,
                                 label=model[i])
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' + 'Fig1 model-fairness ' + save_inform + '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/ ' + 'Fig1 model-fairness ' + save_inform + '.pdf',
                            dpi=600, bbox_inches='tight')




            if 'pu' in sum_rate_type:
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Sum-rate of PU links (Mnats/s)', fontsize=15)
                if len(model)==1:
                    plt.plot(result[0]['epoch'], np.array(result[0]['sum_rate_pu']) / pow(10, 6), self.display[0],
                             MarkerSize=MarkerSize,
                             label=model[0])
                else:
                    for i in range(len(model)):
                        plt.plot(result[i]['epoch'], np.array(result[i]['sum_rate_pu']) / pow(10, 6), self.display[i], MarkerSize=MarkerSize,
                                 label=model[i])
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' + 'Fig1 model-sum_rate_pu' + save_inform + '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/ ' + 'Fig1 model-sum_rate_pu ' + save_inform + '.pdf',
                            dpi=600, bbox_inches='tight')

            if 'sum' in sum_rate_type:
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Sum-rate (Mnats/s)', fontsize=15)
                if len(model)==1:
                    plt.plot(result[0]['epoch'], np.array(result[0]['sum_rate']) / pow(10, 6), self.display[0],
                             MarkerSize=MarkerSize,
                             label=model[0])
                else:
                    for i in range(len(model)):
                        plt.plot(result[i]['epoch'], np.array(result[i]['sum_rate']) / pow(10, 6), self.display[i], MarkerSize=MarkerSize,
                                 label=model[i])
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' + 'Fig1 model-sum_rate' + save_inform + '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/ ' + 'Fig1 model-sum_rate ' + save_inform + '.pdf',
                            dpi=600, bbox_inches='tight')


            '''Reward'''
            plt.figure(figsize=(8, 5))
            plt.xlabel(r'Epoches', fontsize=15)
            plt.ylabel(r'Reward', fontsize=15)
            if len(model) == 1:
                plt.plot(result[0]['epoch'], np.array(result[0]['reward']), self.display[0],
                         MarkerSize=MarkerSize,
                         label=model[0])
            else:
                for i in range(len(model)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['reward']), self.display[i], MarkerSize=MarkerSize,
                             label=model[i])
            plt.grid()  # 生成网格
            plt.legend(fontsize=12)
            plt.savefig(self.simulation_result_path + 'png/' + 'Fig1 model-reward ' + save_inform + '.png',
                        dpi=600, bbox_inches='tight')
            plt.savefig(self.simulation_result_path + 'pdf/ ' + 'Fig1 model-reward ' + save_inform + '.pdf',
                        dpi=600, bbox_inches='tight')



            '''loss'''
            plt.figure(figsize=(8, 5))
            plt.xlabel(r'Epoches', fontsize=15)
            plt.ylabel(r'Loss', fontsize=15)
            if len(model) == 1:
                plt.plot(result[0]['epoch'], np.array(result[0]['total_loss']), self.display[0],
                         MarkerSize=MarkerSize,
                         label=model[0])
            else:
                for i in range(len(model)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['total_loss']), self.display[i], MarkerSize=MarkerSize,
                             label=model[i])
            plt.grid()  # 生成网格
            plt.legend(fontsize=12)
            plt.savefig(self.simulation_result_path + 'png/' + 'Fig1 model-total_loss ' + save_inform + '.png',
                        dpi=600, bbox_inches='tight')
            plt.savefig(self.simulation_result_path + 'pdf/ ' + 'Fig1 model-total_loss ' + save_inform + '.pdf',
                        dpi=600, bbox_inches='tight')



    def compare_crr_effect(self, pu_num=3, su_num=4, crr_num=10, EPOCH=2, ITERATION=80, learningrate=0.01, batchsize=8,
                           model='double_dueling',sum_rate_type='su',show=False):
        result = []
        crrnum_range=crr_num
        for i in range(len(crrnum_range)):
            # crrnum=crr_num[0]+i
            crr_num = crrnum_range[i]
            self.cchn.reset_network()
            metrics2, Sum_rate_PU1, Sum_rate_SU1, Fairness, reward = self.model_train.multi_agent_vary_model(
                cchn=self.cchn,
                pu_num=pu_num,
                su_num=su_num,
                crr_num=crr_num,
                EPOCH=EPOCH,
                ITERATION=ITERATION,
                learningrate=learningrate,
                batchsize=batchsize,
                inform='-1',
                model=model)
            result.append(metrics2)

        scio.savemat(self.simulation_result_path + 'data/Fig3_crr_effect' + '.mat',{'result':result})
        # for i in range(len(crrnum_range)):
        #     scio.savemat(self.simulation_result_path+'data/Fig3_'+str(crrnum_range[i])+'.mat',
        #                  {'epoch':result[i]['epoch'],'sum_rate_su':result[i]['sum_rate_su'],
        #                   'sum_rate_pu':result[i]['sum_rate_pu'],'sum_rate':result[i]['sum_rate'],
        #                   'total_loss':result[i]['total_loss'],
        #                   'fairness': result[i]['fairness'],'reward':result[i]['reward'],})

        if show==True:
            MarkerSize = 1
            if 'su' in sum_rate_type:
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Sum-rate of IoT links (Mnats/s)', fontsize=15)
                for i in range(len(crrnum_range)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['sum_rate_su']) / pow(10, 6), self.display[i],
                             MarkerSize=MarkerSize,
                             label=u'CR-routers=' + str(crrnum_range[i]))
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' + 'Fig3 CR-effect sum_rate_su ' + 'epoch-' + str(EPOCH) +'-' + model+ '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/ ' +'Fig3 CR-effect sum_rate_su ' +  'epoch-' + str(EPOCH) +'-' + model+ '.pdf',
                            dpi=600, bbox_inches='tight')

                '''Fairness'''
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Fairness', fontsize=15)
                for i in range(len(crrnum_range)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['fairness']), self.display[i],
                             MarkerSize=MarkerSize,
                             label=u'CR-routers=' + str(crrnum_range[i]))
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' + 'Fig3 CR-effect fairness ' + 'epoch-' + str(
                    EPOCH) +'-' + model+ '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/ ' + 'Fig3 CR-effect fairness ' + 'epoch-' + str(
                    EPOCH)+'-' + model + '.pdf',
                            dpi=600, bbox_inches='tight')

            if 'pu' in sum_rate_type:
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Sum-rate of PU links (Mnats/s)', fontsize=15)
                for i in range(len(crrnum_range)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['sum_rate_pu']) / pow(10, 6), self.display[i],
                             MarkerSize=MarkerSize,
                             label=u'CR-routers=' + str(crrnum_range[i]))
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' + 'Fig3 CR-effect sum_rate_pu ' + 'epoch-' + str(EPOCH)+'-' + model + '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/ ' +'Fig3 CR-effect sum_rate_pu ' +  'epoch-' + str(EPOCH)+'-' + model + '.pdf',
                            dpi=600, bbox_inches='tight')

            if 'sum' in sum_rate_type:
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Sum-rate of PU links (Mnats/s)', fontsize=15)
                for i in range(len(crrnum_range)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['sum_rate']) / pow(10, 6), self.display[i],
                             MarkerSize=MarkerSize,
                             label=u'CR-routers=' + str(crrnum_range[i]))
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(
                    self.simulation_result_path + 'png/' + 'Fig3 CR-effect sum_rate ' + 'epoch-' + str(EPOCH)+'-' + model + '.png',
                    dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/ ' + 'Fig3 CR-effect sum_rate ' + 'epoch-' + str(
                    EPOCH) +'-' + model+ '.pdf',
                            dpi=600, bbox_inches='tight')



            plt.figure(figsize=(8, 5))
            plt.xlabel(r'Epoches', fontsize=15)
            plt.ylabel(r'Reward', fontsize=15)
            for i in range(len(crrnum_range)):
                plt.plot(result[i]['epoch'], np.array(result[i]['reward']), self.display[i],
                         MarkerSize=MarkerSize,
                         label=u'CR-routers=' + str(crrnum_range[i]))
            plt.grid()  # 生成网格
            plt.legend(fontsize=12)
            plt.savefig(self.simulation_result_path + 'png/' + 'Fig3 CR-effect reward ' + 'epoch-' + str(EPOCH)+'-' + model + '.png',
                        dpi=600, bbox_inches='tight')
            plt.savefig(self.simulation_result_path + 'pdf/ ' +'Fig3 CR-effect reward ' +  'epoch-' + str(EPOCH) +'-' + model+ '.pdf',
                        dpi=600, bbox_inches='tight')



    def sum_rate_varying_batchsize(self, pu_num=3, su_num=6, crr_num=5, EPOCH=2, ITERATION=80, learningrate=0.01,
                                   batchsize=8, model='double_dueling',sum_rate_type='su',show=False):
        result = []
        self.cchn.reset_network()
        batchsize_mode=batchsize

        for i in range(len(batchsize_mode)):
            batchsize = batchsize_mode[i]
            # metrics1, Sum_rate_PU1, Sum_rate_SU1, Fairness = self.model_train.multi_agent(cchn=self.cchn, pu_num=pu_num,
            #                                                                               su_num=su_num,
            #                                                                               crr_num=crr_num,
            #                                                                               EPOCH=EPOCH,
            #                                                                               ITERATION=ITERATION,
            #                                                                               learningrate=learningrate,
            #                                                                               batchsize=batchsize,
            #                                                                               inform='-1',
            #                                                                               model=model)
            metrics1, Sum_rate_PU1, Sum_rate_SU1, Fairness, reward = self.model_train.multi_agent_vary_model(
                                                                                            cchn=self.cchn,
                                                                                            pu_num=pu_num,
                                                                                            su_num=su_num,
                                                                                            crr_num=crr_num,
                                                                                            EPOCH=EPOCH,
                                                                                            ITERATION=ITERATION,
                                                                                            learningrate=learningrate,
                                                                                            batchsize=batchsize,
                                                                                            inform='-1',
                                                                                            model=model)
            result.append(metrics1)

        scio.savemat(self.simulation_result_path + 'data/Fig2_batchsize'+ '.mat',{'result':result})
        # for i in range(len(batchsize_mode)):
        #     scio.savemat(self.simulation_result_path+'data/Fig2_'+str(batchsize_mode[i])+'.mat',
        #                  {'epoch':result[i]['epoch'],'sum_rate_su':result[i]['sum_rate_su'],
        #                   'sum_rate_pu':result[i]['sum_rate_pu'],'sum_rate':result[i]['sum_rate'],
        #                   'total_loss':result[i]['total_loss'],
        #                   'fairness': result[i]['fairness'],'reward':result[i]['reward'],})
        if show==True:
            maker = 1

            if 'su' in sum_rate_type:
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Sum-rate of IoT links (Mnats/s)', fontsize=15)
                for i in range(len(batchsize_mode)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['sum_rate_su']) / pow(10, 6), self.display[i],
                             MarkerSize=maker, label='Batch size=' + str(batchsize_mode[i]))
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' +'Fig2 batchsize sum_rate_su bs='+ str(batchsize_mode[0]) + '-' + str(batchsize_mode[-1]) +'-' + model + '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/' + 'Fig2 batchsize sum_rate_su bs='+str(batchsize_mode[0]) + '-' + str(batchsize_mode[-1]) +'-' + model + '.pdf',
                            dpi=600, bbox_inches='tight')

                '''Fairness'''
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Fairness', fontsize=15)
                for i in range(len(batchsize_mode)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['fairness']), self.display[i],
                             MarkerSize=maker, label='Batch size=' + str(batchsize_mode[i]))
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(
                    self.simulation_result_path + 'png/' + 'Fig2 batchsize fairness bs=' + str(batchsize_mode[0]) + '-' + str(
                        batchsize_mode[-1]) + '-' + model + '.png',
                    dpi=600, bbox_inches='tight')
                plt.savefig(
                    self.simulation_result_path + 'pdf/' + 'Fig2 batchsize fairness bs=' + str(batchsize_mode[0]) + '-' + str(
                        batchsize_mode[-1]) + '-' + model + '.pdf',
                    dpi=600, bbox_inches='tight')

            if 'pu' in sum_rate_type:
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Sum-rate of PU links (Mnats/s)', fontsize=15)
                for i in range(len(batchsize_mode)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['sum_rate_pu']) / pow(10, 6), self.display[i],
                             MarkerSize=maker, label='Batch size=' + str(batchsize_mode[i]))
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' +'Fig2 batchsize sum_rate_pu bs='+ str(batchsize_mode[0]) + '-' + str(batchsize_mode[-1]) +'-' + model + '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/' +'Fig2 batchsize sum_rate_pu bs='+str(batchsize_mode[0]) + '-' + str(batchsize_mode[-1]) +'-' + model + '.pdf',
                            dpi=600, bbox_inches='tight')

            if 'sum' in sum_rate_type:
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Sum-rate (Mnats/s)', fontsize=15)
                for i in range(len(batchsize_mode)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['sum_rate']) / pow(10, 6), self.display[i],
                             MarkerSize=maker, label='Batch size=' + str(batchsize_mode[i]))
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' +'Fig2 batchsize sum_rate bs='+ str(batchsize_mode[0]) + '-' + str(batchsize_mode[-1]) +'-' + model + '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/' + 'Fig2 batchsize sum_rate bs='+str(batchsize_mode[0]) + '-' + str(batchsize_mode[-1]) +'-' + model + '.pdf',
                            dpi=600, bbox_inches='tight')


            plt.figure(figsize=(8, 5))
            plt.xlabel(r'Epoches', fontsize=15)
            plt.ylabel(r'Reward', fontsize=15)
            for i in range(len(batchsize_mode)):
                plt.plot(result[i]['epoch'], np.array(result[i]['reward']) , self.display[i],
                         MarkerSize=maker, label='Batch size=' + str(batchsize_mode[i]))
            plt.grid()  # 生成网格
            plt.legend(fontsize=12)
            plt.savefig(self.simulation_result_path + 'png/' +'Fig2 batchsize reward bs='+ str(batchsize_mode[0]) + '-' + str(batchsize_mode[-1]) +'-' + model + '.png',
                        dpi=600, bbox_inches='tight')
            plt.savefig(self.simulation_result_path + 'pdf/' + 'Fig2 batchsize reward bs='+str(batchsize_mode[0]) + '-' + str(batchsize_mode[-1]) +'-' + model + '.pdf',
                        dpi=600, bbox_inches='tight')

            # plt.show()

    def compare_different_learning_rate(self, pu_num=3, su_num=4, crr_num=5,
                                        EPOCH=2, ITERATION=80, learningrate=0.01, batchsize=8, model='double_dueling',sum_rate_type='su',show=False):

        self.cchn.reset_network()
        Learningrate = learningrate
        result = []
        for i in range(len(Learningrate)):
            learningrate = Learningrate[i]
            metrics2, Sum_rate_PU1, Sum_rate_SU1, Fairness, reward = self.model_train.multi_agent_vary_model(
                cchn=self.cchn,
                pu_num=pu_num,
                su_num=su_num,
                crr_num=crr_num,
                EPOCH=EPOCH,
                ITERATION=ITERATION,
                learningrate=learningrate,
                batchsize=batchsize,
                inform='-1',
                model=model, stational_lr=True)
            result.append(metrics2)

        scio.savemat(self.simulation_result_path + 'data/Fig4_learning_rate'  + '.mat',{'result':result})
        # for i in range(len(Learningrate)):
        #     scio.savemat(self.simulation_result_path+'data/Fig4_'+str(Learningrate[i])+'.mat',
        #                  {'epoch':result[i]['epoch'],'sum_rate_su':result[i]['sum_rate_su'],
        #                   'sum_rate_pu':result[i]['sum_rate_pu'],'sum_rate':result[i]['sum_rate'],
        #                   'total_loss':result[i]['total_loss'],
        #                   'fairness': result[i]['fairness'],'reward':result[i]['reward'],})
        if show==True:
            save_inform = ' ep-' + str(EPOCH)
            MarkerSize = 1

            if 'su' in sum_rate_type:
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Sum-rate of IoT links (Mnats/s)', fontsize=15)
                for i in range(len(Learningrate)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['sum_rate_su']) / pow(10, 6), self.display[i],
                             MarkerSize=MarkerSize, label=r'$\alpha $=' + str(Learningrate[i]))
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' +'Fig4 learning_rate sum_rate_su'+ save_inform +'-' + model + '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/' + 'Fig4 learning_rate sum_rate_su'+save_inform +'-' + model + '.pdf',
                            dpi=600, bbox_inches='tight')


                '''Fairness'''
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Fairness', fontsize=15)
                for i in range(len(Learningrate)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['fairness']), self.display[i],
                             MarkerSize=MarkerSize, label=r'$\alpha $=' + str(Learningrate[i]))
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)

                plt.savefig(
                    self.simulation_result_path + 'png/' + 'Fig4 learning_rate fairness' + save_inform + '-' + model + '.png',
                    dpi=600, bbox_inches='tight')
                plt.savefig(
                    self.simulation_result_path + 'pdf/' + 'Fig4 learning_rate fairness' + save_inform + '-' + model + '.pdf',
                    dpi=600, bbox_inches='tight')

            if 'pu' in sum_rate_type:
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Sum-rate of PU links (Mnats/s)', fontsize=15)
                for i in range(len(Learningrate)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['sum_rate_pu']) / pow(10, 6), self.display[i],
                             MarkerSize=MarkerSize, label=r'$\alpha $=' + str(Learningrate[i]))
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)

                plt.savefig(self.simulation_result_path + 'png/' +'Fig4 learning_rate sum_rate_pu'+ save_inform +'-' + model + '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/' + 'Fig4 learning_rate sum_rate_pu'+save_inform +'-' + model + '.pdf',
                            dpi=600, bbox_inches='tight')

            if 'sum' in sum_rate_type:
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Sum-rate (Mnats/s)', fontsize=15)
                for i in range(len(Learningrate)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['sum_rate']) / pow(10, 6), self.display[i],
                             MarkerSize=MarkerSize, label=r'$\alpha $=' + str(Learningrate[i]))
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)

                plt.savefig(self.simulation_result_path + 'png/' +'Fig4 learning_rate sum_rate'+ save_inform +'-' + model + '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/' + 'Fig4 learning_rate sum_rate'+save_inform +'-' + model + '.pdf',
                            dpi=600, bbox_inches='tight')




            plt.figure(figsize=(8, 5))
            plt.xlabel(r'Epoches', fontsize=15)
            plt.ylabel(r'Reward', fontsize=15)
            for i in range(len(Learningrate)):
                plt.plot(result[i]['epoch'], np.array(result[i]['reward']), self.display[i],
                         MarkerSize=MarkerSize, label=r'$\alpha $=' + str(Learningrate[i]))
            plt.grid()  # 生成网格
            plt.legend(fontsize=12)

            plt.savefig(self.simulation_result_path + 'png/' + 'Fig4 learning_rate reward' + save_inform + '-' + model + '.png',
                        dpi=600, bbox_inches='tight')
            plt.savefig(self.simulation_result_path + 'pdf/' + 'Fig4 learning_rate reward' + save_inform + '-' + model + '.pdf',
                        dpi=600, bbox_inches='tight')


    def compare_different_e_greedy_epsion(self, pu_num=3, su_num=4, crr_num=5,
                                          EPOCH=2, ITERATION=80, learningrate=0.01, batchsize=8,
                                          model='double_dueling',epsion=0.8,sum_rate_type='su',show=False):

        self.cchn.reset_network()
        e_greedy_epsion_mode = epsion
        result = []
        for i in range(len(e_greedy_epsion_mode)):
            e_greedy_epsion = e_greedy_epsion_mode[i]
            metrics2, Sum_rate_PU1, Sum_rate_SU1, Fairness, reward = self.model_train.multi_agent_vary_model(
                cchn=self.cchn,
                pu_num=pu_num,
                su_num=su_num,
                crr_num=crr_num,
                EPOCH=EPOCH,
                ITERATION=ITERATION,
                learningrate=learningrate,
                batchsize=batchsize,
                inform='-1',
                model=model, epsion=e_greedy_epsion)
            result.append(metrics2)
        scio.savemat(self.simulation_result_path + 'data/Fig5_epsion'  + '.mat',{'result':result})
        # for i in range(len(e_greedy_epsion_mode)):
        #     scio.savemat(self.simulation_result_path+'data/Fig5_'+str(e_greedy_epsion_mode[i])+'.mat',
        #                  {'epoch':result[i]['epoch'],'sum_rate_su':result[i]['sum_rate_su'],
        #                   'sum_rate_pu':result[i]['sum_rate_pu'],'sum_rate':result[i]['sum_rate'],
        #                   'total_loss':result[i]['total_loss'],
        #                   'fairness': result[i]['fairness'],'reward':result[i]['reward'],})

        if show==True:
            save_inform = ' ep-' + str(EPOCH)
            MarkerSize = 1

            if 'su' in sum_rate_type:
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Sum-rate of IoT links (Mnats/s)', fontsize=15)
                for i in range(len(e_greedy_epsion_mode)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['sum_rate_su']) / pow(10, 6), self.display[i],
                             MarkerSize=MarkerSize, label=r'$\varepsilon $=' + str(e_greedy_epsion_mode[i]))
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' +'Fig5 epsion sum_rate_su'+ save_inform +'-' + model + '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/' + 'Fig5 epsion sum_rate_su'+save_inform +'-' + model + '.pdf',
                            dpi=600, bbox_inches='tight')

                '''Fairness'''
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Fairness', fontsize=15)
                for i in range(len(e_greedy_epsion_mode)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['fairness']), self.display[i],
                             MarkerSize=MarkerSize, label=r'$\varepsilon $=' + str(e_greedy_epsion_mode[i]))
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(
                    self.simulation_result_path + 'png/' + 'Fig5 epsion fairness' + save_inform + '-' + model + '.png',
                    dpi=600, bbox_inches='tight')
                plt.savefig(
                    self.simulation_result_path + 'pdf/' + 'Fig5 epsion fairness' + save_inform + '-' + model + '.pdf',
                    dpi=600, bbox_inches='tight')

            if 'pu' in sum_rate_type:
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Sum-rate of PU links (Mnats/s)', fontsize=15)
                for i in range(len(e_greedy_epsion_mode)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['sum_rate_pu']) / pow(10, 6), self.display[i],
                             MarkerSize=MarkerSize, label=r'$\varepsilon $=' + str(e_greedy_epsion_mode[i]))
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' +'Fig5 epsion sum_rate_pu'+ save_inform +'-' + model + '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/' + 'Fig5 epsion sum_rate_pu'+save_inform +'-' + model + '.pdf',
                            dpi=600, bbox_inches='tight')

            if 'sum' in sum_rate_type:
                plt.figure(figsize=(8, 5))
                plt.xlabel(r'Epoches', fontsize=15)
                plt.ylabel(r'Sum-rate (Mnats/s)', fontsize=15)
                for i in range(len(e_greedy_epsion_mode)):
                    plt.plot(result[i]['epoch'], np.array(result[i]['sum_rate']) / pow(10, 6), self.display[i],
                             MarkerSize=MarkerSize, label=r'$\varepsilon $=' + str(e_greedy_epsion_mode[i]))
                plt.grid()  # 生成网格
                plt.legend(fontsize=12)
                plt.savefig(self.simulation_result_path + 'png/' +'Fig5 epsion sum_rate'+ save_inform +'-' + model + '.png',
                            dpi=600, bbox_inches='tight')
                plt.savefig(self.simulation_result_path + 'pdf/' + 'Fig5 epsion sum_rate'+save_inform +'-' + model + '.pdf',
                            dpi=600, bbox_inches='tight')




            plt.figure(figsize=(8, 5))
            plt.xlabel(r'Epoches', fontsize=15)
            plt.ylabel(r'Reward', fontsize=15)
            for i in range(len(e_greedy_epsion_mode)):
                plt.plot(result[i]['epoch'], np.array(result[i]['reward']) , self.display[i],
                         MarkerSize=MarkerSize, label=r'$\varepsilon $=' + str(e_greedy_epsion_mode[i]))
            plt.grid()  # 生成网格
            plt.legend(fontsize=12)
            plt.savefig(self.simulation_result_path + 'png/' +'Fig5 epsion reward'+ save_inform +'-' + model + '.png',
                        dpi=600, bbox_inches='tight')
            plt.savefig(self.simulation_result_path + 'pdf/' + 'Fig5 epsion reward'+save_inform +'-' + model + '.pdf',
                        dpi=600, bbox_inches='tight')


    def optimal_allocation(self, pu_num=3, su_num=6, crr_num=5, pu_min_rate=20000):
        self.cchn.reset_network()
        self.cchn.abstract_network(primary_number=pu_num, secondary_number=su_num, CR_router_number=crr_num)
        self.cchn.reset_min_pu_rate(pu_min_rate)
        each_user_action = self.cchn.get_action()

        total_action = pow(each_user_action, su_num)

        Count = [0 for _ in range(su_num)]
        decision_temp = [0 for _ in range(su_num)]

        print('total_action:', total_action)

        temp_count = []

        for i in range(total_action):
            Count[-1] = i
            temp_count.append(Count)
            print('Count:', Count)

            decision_temp[0] = Count[0] % each_user_action
            # print('decision_temp:',decision_temp)

            for j in range(su_num):
                if Count[su_num - j - 1] % (each_user_action - 1) == 0 and su_num - j - 2 >= 0 and Count[
                    su_num - j - 1] != 0:
                    Count[su_num - j - 2] = Count[su_num - j - 2] + 1

        print('temp_count:', temp_count)

    def memery(self):
        user = 3

        states = 3
        action = 3
        reward = 1

        memery = np.random.randn(user * states * 2 + 1 + user)
        print('memery:', memery)

        i = 2

        b_s = memery[i * states:(i + 1) * states]
        b_a = memery[user * states + i:user * states + i + 1]
        b_r = memery[user * states + user:user * states + user + 1]
        b_s_ = memery[user * states + user + 1 + i * states:user * states + user + 1 + (i + 1) * states]

        print('b_s:', b_s, 'b_a:', b_a, 'b_r:', b_r, 'b_s_:', b_s_)

    def random_allocation(self, pu_num=3, su_num=6, crr_num=5, pu_min_rate=20000):
        self.cchn.reset_network()
        self.cchn.abstract_network(primary_number=pu_num, secondary_number=su_num, CR_router_number=crr_num)
        self.cchn.reset_min_pu_rate(pu_min_rate)

        random_decision = self.cchn.init_alloation()

        Sum_rate_PU, Sum_rate_SU, SINR_PU, Interference_PU = self.cchn.get_sum_rate_test(random_decision)
        Rate_PU, Rate_SU = self.cchn.get_sum_rate_game(random_decision)

        sum_rate = (Sum_rate_SU + Sum_rate_PU) ** 2
        user_num = len(Rate_PU) + len(Rate_SU)
        user_rate = sum(np.array(Rate_SU) ** 2) + sum(np.array(Rate_PU) ** 2)
        # Fairness = sum_rate/(user_num*user_rate)
        # print('Fairness:',Fairness)
        Fairness = Sum_rate_SU ** 2 / (len(Rate_SU) * sum(np.array(Rate_SU) ** 2))
        # print(Sum_rate_SU,Fairness)
        # print('random_decision:',random_decision)
        reward_old = self.cchn.calculate_reward(random_decision)
        # print('reward_old:',reward_old)
        return Sum_rate_PU, Sum_rate_SU, Fairness, random_decision, reward_old

    def game_allocation(self, initdecision, pu_num=3, su_num=6, crr_num=5, max_failnum=50, iteration=10,
                        pu_min_rate=20000):
        # self.cchn.reset_network()
        # self.cchn.abstract_network(primary_number=pu_num, secondary_number=su_num, CR_router_number=crr_num)
        # print('initdecision:',initdecision)
        init_decision = initdecision
        rate_PU, Rate_SU = self.cchn.get_sum_rate_game(init_decision)
        Sum_rate_SU = sum(Rate_SU)
        Sum_rate_PU=sum(rate_PU)
        reward_old = self.cchn.calculate_reward(init_decision)
        # print('reward_old',reward_old)
        failnumber = 0

        while failnumber < max_failnum:
            su_index1 = np.random.choice(su_num, 1)[0]

            actions = self.cchn.get_action()
            choose_action = np.random.choice(actions, 1)[0]
            # print(choose_action)
            vec = self.cchn.action_mapping(choose_action)
            # print(vec)

            temp_decision = init_decision
            temp_decision[0][su_index1] = vec[0]
            temp_decision[1][su_index1] = vec[1]
            rate_PU1, rate_SU2 = self.cchn.get_sum_rate_game(init_decision)
            sum_rate_su_new=sum(rate_SU2)
            reward_new = self.cchn.calculate_reward(temp_decision)

            # print('SU_',str(su_index1).rjust(1), ' | choose_action:',str(choose_action).rjust(1),
            #       ' | failnumber:', str(failnumber).rjust(1),' | reward_old:%.2f'%(reward_old),
            #       ' | reward_new:%.2f'%reward_new)or sum_rate_su_new>Sum_rate_SU

            if reward_new > reward_old :
                reward_old = reward_new
                Rate_SU=rate_SU2
                Sum_rate_PU=sum(rate_PU1)
                Sum_rate_SU=sum_rate_su_new
                init_decision = temp_decision
                failnumber = 0
            else:
                failnumber += 1

        Sum_rate_PU1, Sum_rate_SU1, SINR_PU1, Interference_PU = self.cchn.get_sum_rate_test(init_decision)
        # Rate_PU, Rate_SU = self.cchn.get_sum_rate_game(init_decision)
        # print('reward_new',reward_old)
        # sum_rate = (Sum_rate_SU + Sum_rate_PU) ** 2
        # user_num = len(Rate_PU) + len(Rate_SU)
        # user_rate = sum(np.array(Rate_SU) ** 2) + sum(np.array(Rate_PU) ** 2)
        # Fairness = sum_rate / (user_num * user_rate)
        Fairness = Sum_rate_SU ** 2 / (len(Rate_SU) * sum(np.array(Rate_SU) ** 2))
        # print('Fairness:',Fairness)
        return Sum_rate_PU, Sum_rate_SU, Fairness, reward_old





    # def calculate_reward(self,Sum_rate_PU, Sum_rate_SU,pu_num):
    #     reward=self.weight_reward[0]*Sum_rate_PU+\
    #            self.weight_reward[1]*Sum_rate_SU+\
    #            self.weight_reward[2]*(Sum_rate_PU-pu_num*self.primary_rate_min)
    #     return  reward

    # def game_allocation1(self, pu_num=3, su_num=6, crr_num=5, max_failnum=50, iteration=10):
    #     self.cchn.reset_network()
    #     self.cchn.abstract_network(primary_number=pu_num, secondary_number=su_num, CR_router_number=crr_num)
    #     failnumber = 0
    #     count = 0
    #     init_decision = self.cchn.init_alloation()
    #     rate_PU1, rate_SU1 = self.cchn.get_sum_rate_game(init_decision)
    #     max_sum_rate = sum(rate_SU1)
    #     reward_old = self.cchn.calculate_reward(init_decision)
    #
    #     while failnumber < max_failnum:
    #
    #         su_index1 = np.random.choice(su_num, 1)
    #         spectrum = init_decision[0][su_index1]
    #         power = init_decision[1][su_index1]
    #
    #         rate_PU1, rate_SU1 = self.cchn.get_sum_rate_game(init_decision)
    #         punish1 = (np.array(rate_PU1)[spectrum] - self.cchn.primary_rate_min)
    #
    #         fail_each_su_num = 0
    #         init_decision1 = init_decision
    #         # reward_old1=reward_old
    #         # max_sum_rate1=
    #         reward_old1 = reward_old
    #         max_sum_rate1 = max_sum_rate
    #
    #         for i in range(iteration):
    #
    #             temp_decision = init_decision1
    #             random_spectrum = np.random.choice(pu_num, 1)
    #             random_power = np.random.choice(self.cchn.power_set_number, 1)
    #             temp_decision[0][su_index1] = random_spectrum
    #             temp_decision[1][su_index1] = random_power
    #
    #             reward_new = self.cchn.calculate_reward(temp_decision)
    #
    #             rate_PU2, rate_SU2 = self.cchn.get_sum_rate_game(temp_decision)
    #
    #             rate_SU_old = max_sum_rate
    #             re = reward_old
    #             ne = reward_new
    #
    #             if reward_new > reward_old:
    #                 reward_old1 = reward_new
    #                 max_sum_rate1 = sum(rate_SU2)
    #                 max_pu = sum(rate_PU2)
    #                 init_decision1 = temp_decision
    #                 fail_each_su_num = 0
    #             else:
    #                 fail_each_su_num += 1
    #             # print('init_decision:', init_decision,'init_decision1:', init_decision1)
    #             # print('fail_each_su_num:', fail_each_su_num, 'reward_old:%.2f'%re,'reward_new:%.2f'%reward_new,)
    #         if fail_each_su_num < iteration - 1:
    #             failnumber = 0
    #
    #         else:
    #             failnumber += 1
    #         reward_old = reward_old1
    #         init_decision = init_decision1
    #         max_sum_rate = max_sum_rate1
    #         # count+=1
    #         # if count%1==0:
    #         #     print('random_spectrum:',random_spectrum,'random_power:',random_power,
    #         #            'rate_SU_old:%.2f'%max_sum_rate,'rate_SU_new:%.2f'%sum(rate_SU2),
    #         #           'reward:%.2f'%reward_old,'reward_new:%.2f'%ne,'failnumber:',failnumber)#
    #
    #     # rate_PU1, rate_SU1 = self.cchn.get_sum_rate_game(init_decision)
    #     # print(sum(rate_SU1))
    #     Sum_rate_PU, Sum_rate_SU, SINR_PU, Interference_PU = self.cchn.get_sum_rate_test(init_decision)
    #     Rate_PU, Rate_SU = self.cchn.get_sum_rate_game(init_decision)
    #
    #     sum_rate = (Sum_rate_SU + Sum_rate_PU) ** 2
    #     user_num = len(Rate_PU) + len(Rate_SU)
    #     user_rate = sum(np.array(Rate_SU) ** 2) + sum(np.array(Rate_PU) ** 2)
    #     Fairness = sum_rate / (user_num * user_rate)
    #     # print(max_sum_rate,Fairness)
    #
    #     return Sum_rate_PU, Sum_rate_SU, Fairness  # ,sum(rate_PU1),sum(rate_SU1)
    #
    # def nearest_first_allocation(self, pu_num=3, su_num=6, crr_num=5):
    #     self.cchn.reset_network()
    #     self.cchn.abstract_network(primary_number=pu_num, secondary_number=su_num, CR_router_number=crr_num)
    #
    # def farthest_first_allocation(self, pu_num=3, su_num=6, crr_num=5):
    #     self.cchn.reset_network()
    #     self.cchn.abstract_network(primary_number=pu_num, secondary_number=su_num, CR_router_number=crr_num)



        ''''
        B：蓝色    
        G：绿色
        r：红色
        C：青色
        M：洋红      
        Y：黄色       
        K：黑色
        W：白色
        线型
        '-'       solid line style
        '--'      dashed line style
        '-.'      dash-dot line style
        ':'       dotted line style

        …点标记
        '.'       point marker 
        ','       pixel marker 像素标记
        'o'       circle marker 圆形标记
        'v'       triangle_down marker 三角形下标记
        '^'       triangle_up marker “三角形”上标记
        '<'       triangle_left marker
        '>'       triangle_right marker
        '1'       tri_down marker 三下标记
        '2'       tri_up marker 三向标记
        '3'       tri_left marker 三个左标记
        '4'       tri_right marker 三个右标记
        's'       square marker 方形标记
        'p'       pentagon marker 五角大楼标记
        '*'       star marker 星域
        'h'       hexagon1 marker 六边形标记
        'H'       hexagon2 marker
        '+'       plus marker 加号
        'x'       x marker x标记
        'D'       diamond marker 菱形标记
        'd'       thin_diamond marker 细菱形标记
        '|'       vline marker 蝴蝶标记
        '_'       hline marker uhline标记
        '''
