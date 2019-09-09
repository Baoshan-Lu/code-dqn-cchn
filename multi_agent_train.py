import os
import torch
import numpy as np
from  model  import DQN,JointDQN
import argparse
import time
from datetime import datetime
import matplotlib.pyplot as plt
import json
from json import encoder
import sys

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf8')
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

from  cchn_model  import  CCHN

class Model_train(object):
    def __init__(self, parameters):
       # self.states=self.cchn.get_state()
       # self.actions=self.cchn.get_action()
       self.parameters=parameters
       self.device = torch.device(
           "cuda" if torch.cuda.is_available() else "cpu"
       )
       self.simulation_result_path = parameters.simulation_results_path
       try:
           os.mkdir(self.simulation_result_path)
       except Exception:
           pass

       self.parameters=parameters
       self.pretrain=parameters.pretrain
       self.batchsize = parameters.batchsize
       self.sigma_factor = parameters.sigma_factor
       self.test_number = parameters.test_number
       self.start_train=parameters.start_train
       self.transition_steps=parameters.transition_steps
       self.iteration=parameters.iteration

       self.primary_rate_min = parameters.primary_rate_min
       self.secodary_rate_min = parameters.secodary_rate_min

       # self.primary_number = parameters.primary_number
       # self.secondary_number = parameters.secondary_number
       # self.CR_router_number = parameters.CR_router_number
       #
       self.save_path=parameters.save_path
       self.data_path=parameters.data_path
       self.picture_path=parameters.picture_path
       self.learning_rate = parameters.learning_rate
       self.gpu_type = parameters.gpu_type
       self.memory_capacity = parameters.memory_capacity
       self.double_dqn = parameters.double_dqn
       self.dueling_dqn = parameters.dueling_dqn
       self.noisy_dqn = parameters.noisy_dqn

       self.epoch=parameters.epoch
       self.learning_rate = parameters.learning_rate
       self.CR_router_number = parameters.CR_router_number
       self.power_set_number=parameters.power_set_number

       try:
           os.mkdir(self.save_path)
       except Exception:
           pass

    def multi_agent(self, cchn, pu_num=6, su_num=8, crr_num=10,
                    EPOCH=10, ITERATION=80, learningrate=0.01, batchsize=32,
                    inform='-1', model='double_dueling', runtype='running', modelsaving='nosave'):

        cchn.abstract_network(primary_number=pu_num, secondary_number=su_num, CR_router_number=crr_num)

        actions = cchn.get_action()
        states = cchn.get_state()

        print('==========System parameters==========')
        print('Primary_number=', pu_num, '|  Secondary_number=', su_num, '|  CR-routers=', crr_num, )
        print('Power_number=', self.power_set_number, '|  Spectrum_number=', pu_num, '|  primary_rate_min=',self.primary_rate_min)
        print('States=', states, '| Actions =', actions, )
        print('Memory_capacity=', self.memory_capacity, '|  Batch size=', batchsize)
        print('Epoches=', EPOCH, '| Iterations=', ITERATION, '|  Learning_rate=', learningrate)
        print('Device =', self.device, '| pretrained model=', self.pretrain, )


        '''1:生成各个模型'''
        agent = []
        for su_i in range(su_num):
            name = DQN(self.parameters, actions,states,learningrate,batchsize)
            agent.append(name)

        print('Model=',model,'  | Start training...')
        metrics = {'epoch': [], 'sum_rate_su': [], 'sum_rate_pu': [], 'sum_rate': [],'fairness':[],'reward':[]}
        for epoch in range(EPOCH):
            for su_i in range(su_num):
                dqn_agent=agent[su_i]
                '''2:生成初始分配'''
                old_decision = cchn.init_alloation()
                old_decision_for_each_su=old_decision
                score=0
                for iteration in range(ITERATION):
                    state=cchn.obtain_state(old_decision_for_each_su,su_i)
                    action=dqn_agent.choose_action(state,epoch/self.epoch)
                    reward,next_state,new_decision=cchn.evaluate_environmet1(action,su_i)
                    score += reward

                    dqn_agent.store_transition(state, action, reward, next_state)

                    old_decision_for_each_su=new_decision

                    if dqn_agent.memory_counter > self.start_train:
                        loss =dqn_agent.learn()

                        if iteration % 100 == 0:
                            if  runtype=='deburg':
                                print(
                                    # datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'Epoch:', epoch+1,
                                    '| SU_i:',su_i+1,
                                    '| iteration:', iteration+1,
                                    '| Loss: %.5f' % loss,
                                    '| Reward: %.5f'%reward,
                                    '| Score: %.5f'%score,
                                )
            if epoch%1==0:
                self.learning_rate=self.learning_rate*0.97

                # Sum_rate_PU, Sum_rate_SU = self.calculte_user_sum_rate(cchn,agent,su_num)
                Sum_rate_PU, Sum_rate_SU, Fairness, reward = self.calculte_user_fairness(cchn, agent, su_num)
                sumrate = Sum_rate_PU + Sum_rate_SU

                metrics['epoch'].append(epoch+1)
                metrics['sum_rate_pu'].append(Sum_rate_PU)
                metrics['sum_rate_su'].append(Sum_rate_SU)
                metrics['sum_rate'].append(sumrate)
                metrics['fairness'].append(Fairness)
                metrics['reward'].append(reward)

                # print('\n')
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    '| Epoch:', epoch+1,
                    '| Sum-rate of PUs: %.2f' % Sum_rate_PU,
                    '| Sum-rate of SUs: %.2f' % Sum_rate_SU,
                    '| Sum-rate: %.2f' % sumrate,
                    '| Fairness: %.2f' % Fairness,
                    '| Reward: %.2f' % reward,
                      # '\n'
                )

        '''保存模型,保存数据'''
        if modelsaving=='save':
            torch.save(agent,self.save_path+'agent'+inform)
            json.dump({'metrics': metrics}, fp=open(self.data_path+'traing_data'+inform +'.rs', 'w'),indent=4)

        if runtype=='deburg':
            '''性能画图1: sum-rate'''
            plt.figure(figsize=(8, 5))
            plt.xlabel(r'The number of epoches', fontsize=15)
            plt.ylabel(r'Sum-rate (Mnats/s)', fontsize=15)
            plt.plot(metrics['epoch'], np.array(metrics['sum_rate_pu'])/pow(10,6), '->b', MarkerSize=1, label=u'PU')
            plt.plot(metrics['epoch'], np.array(metrics['sum_rate_su'])/pow(10,6), '-or', MarkerSize=1, label=u'SU')
            plt.plot(metrics['epoch'], np.array(metrics['sum_rate_su'])/pow(10,6)+np.array(metrics['sum_rate_pu'])/pow(10,6),
                     '-sg', MarkerSize=1, label=u'PU+SU')
            plt.legend(fontsize=12)
            plt.grid()
            plt.savefig(self.simulation_result_path+ 'Fig1 sum_rate '+model+' PU-' + str(pu_num) +
                        ' SU-' + str(su_num) +' CRR-' + str(crr_num) +' EP-'+str(EPOCH)+
                        ' iter-' + str(ITERATION) +' batsz-' + str(batchsize) +
                        ' lr-' + str(learningrate) + '.pdf', dpi=600, bbox_inches='tight')

            '''性能画图2: reward'''
            plt.figure(figsize=(8, 5))
            plt.xlabel(r'The number of epoches', fontsize=15)
            plt.ylabel(r'Reward', fontsize=15)
            plt.plot(metrics['epoch'], np.array(metrics['reward']), '-or', MarkerSize=1)
            # plt.legend(fontsize=12)
            plt.grid()

            plt.savefig(self.simulation_result_path+ 'Fig1 reward '+model+' PU-' + str(pu_num) +
                        ' SU-' + str(su_num) +' CRR-' + str(crr_num) +' EP-'+str(EPOCH)+
                        ' iter-' + str(ITERATION) +' batsz-' + str(batchsize) +
                        ' lr-' + str(learningrate) + '.pdf', dpi=600, bbox_inches='tight')

            # plt.show()

        Sum_rate_PU, Sum_rate_SU = self.calculte_user_sum_rate(cchn, agent, su_num)
        Sum_rate_PU, Sum_rate_SU, Fairness,reward=self.calculte_user_fairness(cchn, agent, su_num)

        return  metrics,Sum_rate_PU,Sum_rate_SU,Fairness

    def multi_agent_vary_model(self,cchn, pu_num=6, su_num=8, crr_num=10,
                               EPOCH=10, ITERATION=80, learningrate=0.01, batchsize=32,
                    inform='-1', model='double_dueling',epsion=0.8,pu_min_rate=20000,stational_lr=False):

        cchn.abstract_network(primary_number=pu_num,secondary_number=su_num,CR_router_number=crr_num)
        cchn.reset_min_pu_rate(pu_min_rate)

        actions=cchn.get_action()
        states=cchn.get_state()

        print('==========System parameters==========')
        print('Primary_number=', pu_num, '|  Secondary_number=', su_num, '|  CR-routers=', crr_num,)
        print('Power_number=', self.power_set_number, '|  primary_rate_min=', pu_min_rate)
        print('States=',states, '| Actions =', actions, '|  epsion=',epsion,)
        print('Memory_capacity=', self.memory_capacity, '|  Batch size=', batchsize)
        print('Epoches=', EPOCH, '| Iterations=',ITERATION,'|  Learning_rate=', learningrate,'|  Stational lr=',stational_lr)
        print('Device =', self.device,'| pretrained model=', self.pretrain, )


        '''1:生成各个模型'''
        agent = []
        for su_i in range(su_num):
            name = DQN(self.parameters, actions,states,learningrate,batchsize)
            name.reset_model(model=model)
            name.reset_e_greedy_epsion(epsion)
            agent.append(name)

        print('Model= ',model,'  | Start training...')

        metrics = {'epoch': [], 'sum_rate_su': [], 'sum_rate_pu': [], 'sum_rate': [],'fairness':[],'reward':[],'total_loss':[]}
        for epoch in range(EPOCH):
            loss_sum=0

            for su_i in range(su_num):

                dqn_agent=agent[su_i]
                '''2:生成初始分配'''
                old_decision = cchn.init_alloation()
                old_decision_for_each_su=old_decision
                score=0
                total_loss = 0

                for iteration in range(ITERATION):
                    state=cchn.obtain_state(old_decision_for_each_su,su_i)
                    action=dqn_agent.choose_action(state,epoch/self.epoch)
                    reward,next_state,new_decision=cchn.evaluate_environmet1(action,su_i)
                    score += reward

                    dqn_agent.store_transition(state, action, reward, next_state)

                    old_decision_for_each_su=new_decision

                    if dqn_agent.memory_counter > self.start_train:
                        loss =dqn_agent.learn()
                        total_loss=total_loss+loss

                dqn_agent.reset_learning_rate(learningrate)

            loss_sum=loss_sum+total_loss/ITERATION

            '''学习率调节'''
            if epoch % 20 == 0:
                if stational_lr == False:
                    learningrate = pow(0.99, epoch) * learningrate


            if epoch%1==0:

                Sum_rate_PU, Sum_rate_SU, Fairness, reward = self.calculte_user_fairness(cchn, agent, su_num)
                sumrate = Sum_rate_PU + Sum_rate_SU

                metrics['epoch'].append(epoch+1)
                metrics['sum_rate_pu'].append(Sum_rate_PU)
                metrics['sum_rate_su'].append(Sum_rate_SU)
                metrics['sum_rate'].append(sumrate)
                metrics['fairness'].append(Fairness)
                metrics['reward'].append(reward)
                metrics['total_loss'].append(loss_sum/su_num)
                # print('\n')
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    '| Epoch:', epoch+1,
                    '| total_loss: %.3f' % (loss_sum/su_num),
                    '| Sum-rate of PUs: %.2f' % Sum_rate_PU,
                    '| Sum-rate of SUs: %.2f' % Sum_rate_SU,
                    # '| Sum-rate: %.2f' % sumrate,
                    '| Fairness: %.2f' % Fairness,
                    '| Reward: %.2f' % reward,
                    '| Learing_rate: %.10f' %learningrate
                      # '\n'
                )
        Sum_rate_PU, Sum_rate_SU = self.calculte_user_sum_rate(cchn, agent, su_num)
        Sum_rate_PU, Sum_rate_SU, Fairness,reward=self.calculte_user_fairness(cchn, agent, su_num)


        return  metrics,Sum_rate_PU,Sum_rate_SU,Fairness,reward


    def joint_model_trainning(self,cchn, pu_num=6, su_num=8, crr_num=10,
                    epoch=10,  learningrate=0.01, batchsize=32,
                              epsion=0.8,gamma=0.9,targetnet_update_rate=100, memory_capacity = 2000,pu_min_rate=20000,
                     model='double_dueling', runtype='running', modelsaving='nosave',stational_lr=False,inform='-1'):


        cchn.abstract_network(primary_number=pu_num,secondary_number=su_num,CR_router_number=crr_num)
        cchn.reset_min_pu_rate(pu_min_rate)

        actions=cchn.get_action()
        states=cchn.get_state_for_joint_model()

        print('==========System parameters==========')
        print('Primary_number=', pu_num, '|  Secondary_number=', su_num, '|  CR-routers=', crr_num,)
        print('Power_number=', self.power_set_number, '|  primary_rate_min=', pu_min_rate)
        print('States=',states, '| Actions =', actions, '|  epsion=',epsion,)
        print('Memory_capacity=', self.memory_capacity, '|  Batch size=', batchsize)
        print('Epoches=', epoch, '|  Learning_rate=', learningrate,'|  Stational lr=',stational_lr)
        print('Device =', self.device,'| Model=', model, )

        metrics = {'epoch': [],'loss': [], 'sum_rate_su': [], 'sum_rate_pu': [], 'sum_rate': [],'fairness':[],'reward':[]}

        agent=JointDQN(self.parameters,states = states,actions=actions,user=su_num,learningrate = learningrate,
                       batchsize = batchsize,epsion = epsion,gamma = gamma,targetnet_update_rate = targetnet_update_rate,
                       memory_capacity = memory_capacity,model =model)
        count=0
        while count<epoch:
            score = 0

            state=[]
            action=[]

            '''开始动作'''
            old_decision = cchn.init_alloation_for_joint_model()

            for su_i in range(su_num):
                state_su_i = cchn.obtain_state_for_joint_model(old_decision, su_i)
                # print('state_su_i:',state_su_i)
                action_su_i = agent.choose_action(state_su_i, epoch / self.epoch)
                new_decision=cchn.environmet_for_joint_model(action_su_i,su_i)
                for i in range(states):
                    state.append(state_su_i[i]/10)

                action.append(action_su_i)

            # print('\nold_decision:',old_decision,'\naction:',action,'\nnew_decision:', new_decision)

            # print('state:',len(state))

            '''计算回报'''
            reward=cchn.calculate_reward(new_decision)

            '''下一个状态'''
            next_state = []
            for su_i in range(su_num):
                state_su_i = cchn.obtain_state_for_joint_model(new_decision, su_i)
                for i in range(states):
                    next_state.append(state_su_i[i]/10)
                # next_state.append(state_su_i)

            '''存储'''
            agent.store_transition(state, action, reward, next_state)

            # print('state:',state,'next_state:',next_state)

            # '''切换到下一个状态'''
            # old_decision = new_decision

            score += reward

            if agent.memory_counter > self.start_train:
                count+=1
                loss = agent.learn()
                metrics['epoch'].append(count)
                metrics['loss'].append(loss)
                # metrics['sum_rate_pu'].append(Sum_rate_PU)
                # metrics['sum_rate_su'].append(Sum_rate_SU)
                # metrics['sum_rate'].append(sumrate)
                # metrics['fairness'].append(Fairness)
                metrics['reward'].append(reward)

                # print('\n')
                if count%50==0:
                    if stational_lr == False:
                        self.learning_rate = self.learning_rate * 0.99

                    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        '| Epoch:', count,
                        '| Loss: %.8f' % loss,
                        # '| Sum-rate of PUs: %.2f' % Sum_rate_PU,
                        # '| Sum-rate of SUs: %.2f' % Sum_rate_SU,
                        # '| Sum-rate: %.2f' % sumrate,
                        # '| Fairness: %.2f' % Fairness,
                        '| Reward: %.2f' % reward,
                          # '\n'
                    )

        '''性能画图1: sum-rate'''
        plt.figure(figsize=(8, 5))
        plt.xlabel(r'The number of epoches', fontsize=15)
        plt.ylabel(r'Loss', fontsize=15)
        plt.plot(metrics['epoch'], np.array(metrics['loss']), '-or', MarkerSize=1, label=u'loss')
        plt.legend(fontsize=12)
        plt.grid()


        plt.figure(figsize=(8, 5))
        plt.xlabel(r'The number of epoches', fontsize=15)
        plt.ylabel(r'Reward', fontsize=15)
        plt.plot(metrics['epoch'], np.array(metrics['reward']), '->b', MarkerSize=1, label=u'reward')
        plt.legend(fontsize=12)
        plt.grid()
        plt.show()








    def calculte_user_sum_rate(self,cchn,agent_name,su_num):

        decision=cchn.init_alloation_test()
        # print('\nInit_decision:',decision)

        for su_i in range(su_num):
            state = cchn.obtain_state(decision, su_i)
            '''根据state，确定su_i的动作'''
            action =agent_name[su_i].choose_max_Q_action(state)
            '''执行动作'''
            cchn.evaluate_environmet_test(action, su_i)

        '''测试，执行所有动作，系统进入下一个状态'''
        decision=cchn.test_environmet()

        # print('Final_decision:',decision)

        Sum_rate_PU, Sum_rate_SU, SINR_PU, Interference_PU = cchn.get_sum_rate_test(decision)


        return  Sum_rate_PU, Sum_rate_SU

    def calculte_user_fairness(self,cchn,agent_name,su_num):

        sum_rate_su=[]
        sum_rate_pu=[]
        fairness=[]
        reward_=[]

        for i in range(10):
            decision=cchn.init_alloation_test()
            # print('\nInit_decision:',decision)

            for su_i in range(su_num):
                state = cchn.obtain_state(decision, su_i)
                '''根据state，确定su_i的动作'''
                action =agent_name[su_i].choose_max_Q_action(state)
                '''执行动作'''
                cchn.evaluate_environmet_test(action, su_i)

            '''测试，执行所有动作，系统进入下一个状态'''
            decision=cchn.test_environmet()

            # print('Final_decision:',decision)

            Sum_rate_PU, Sum_rate_SU, SINR_PU, Interference_PU = cchn.get_sum_rate_test(decision)
            Rate_PU, Rate_SU = cchn.get_sum_rate_game(decision)

            sum_rate=(Sum_rate_SU+Sum_rate_PU) ** 2
            user_num=len(Rate_PU)+len(Rate_SU)
            user_rate=sum(np.array(Rate_SU) ** 2)+sum(np.array(Rate_PU) ** 2)
            # Fairness = sum_rate/(user_num*user_rate)
            Fairness = Sum_rate_SU**2/(len(Rate_SU)*sum(np.array(Rate_SU) ** 2))
            reward= cchn.calculate_reward(decision)

            sum_rate_su.append(Sum_rate_SU)

            sum_rate_pu.append(Sum_rate_PU)
            fairness.append(Fairness)
            reward_.append(reward)

        Sum_rate_PU=np.mean(sum_rate_pu)
        Sum_rate_SU=np.mean(sum_rate_su)
        Fairness=np.mean(fairness)
        reward=np.mean(reward_)

        return  Sum_rate_PU, Sum_rate_SU,Fairness,reward