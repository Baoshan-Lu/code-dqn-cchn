import numpy  as np
import matplotlib.pyplot as plt
from itertools import combinations, permutations

class CCHN(object):
    def __init__(self,parameters,show_model=False):

        self.radius= parameters.radius
        self.primary_number = parameters.primary_number
        self.secondary_number = parameters.secondary_number
        self.CR_router_number = parameters.CR_router_number
        self.distance=parameters.distance

        self.save_path = parameters.save_path
        self.picture_path = parameters.picture_path

        self.reward = parameters.reward
        self.sigma_factor = parameters.sigma_factor
        self.pu_power_mode = parameters.pu_power_mode
        self.bandwidth = parameters.bandwidth
        self.weight_reward = parameters.weight_reward

        self.user_power_max = parameters.user_power_max
        self.user_power_min = parameters.user_power_min
        self.power_set_number = parameters.power_set_number
        self.noise_power = parameters.noise_power
        self.mu = parameters.mu
        self.sigma = parameters.sigma
        self.channel_gain = parameters.channel_gain
        self.primary_rate_min = parameters.primary_rate_min
        self.secodary_rate_min = parameters.secodary_rate_min
        self.primary_init_power = parameters.primary_init_power

        self.sense_coverage=parameters.sense_coverage

        '''功率可选项'''
        self.power_set = np.round(np.linspace(self.user_power_min, self.user_power_max, self.power_set_number), 2)


        ''''建立基准网络'''

        self.network = self.create_network()

        self.network_origin=self.network

        if show_model==True:
            self.plot_network(primary_coord=self.network[0],
                              secondary_coord=self.network[1],
                              CR_router_coord=self.network[2])


        '''初始化资源分配'''
        self.decision = self.init_alloation()

        self.decision_temp = self.decision
        self.decision_for_joint_model=self.decision

        self.decision_test = self.decision
        #
        # self.plot_network()

    # def abstract_network(self,primary_number, secondary_number, CR_router_number):
    #     self.primary_number = primary_number
    #     self.secondary_number = secondary_number
    #     self.CR_router_number = CR_router_number

    def reset_min_pu_rate(self,mrq_pu):
        self.primary_rate_min=mrq_pu

    def reset_network_powerset(self,power_set_number):
        self.power_set_number=power_set_number
        self.power_set = np.round(np.linspace(self.user_power_min, self.user_power_max, self.power_set_number), 2)
        # self.network=self.network_origin

    def reset_network(self):
        self.network=self.network_origin

    def abstract_network(self,primary_number, secondary_number, CR_router_number):

        user=[primary_number, secondary_number, CR_router_number]

        cood=[]

        # print(self.network[0][0][1],self.network[0][0][1])
        for i in range(3):
            x1 = self.network[i][0][:user[i]]
            y1 = self.network[i][1][:user[i]]
            x2 = self.network[i][2][:user[i]]
            y2 = self.network[i][3][:user[i]]
            cood1=[x1,y1,x2,y2]
            cood.append(cood1)

        primary_coord=cood[0]
        secondary_coord=cood[1]
        CR_router_coord=cood[2]


        self.primary_number = primary_number
        self.secondary_number = secondary_number
        self.CR_router_number = CR_router_number

        '''初始化资源分配'''
        self.decision = self.init_alloation()

        # print('init:',self.decision)

        self.decision_temp = self.decision

        self.decision_test = self.decision

        self.network=[primary_coord,secondary_coord,CR_router_coord]
        # print(self.network[0][0][1],self.network[0][0][1])
        # self.plot_network(primary_coord=primary_coord,
        #                   secondary_coord=secondary_coord,
        #                   CR_router_coord=CR_router_coord)
        # # print('dis_2:',self.channelgain(self.network[0],self.network[1],0,0))
        # print('Coord SU')
        # for i in range(self.secondary_number):
        #     # dis=np.sqrt((secondary_coord[0][i]-secondary_coord[2][i])**2+(secondary_coord[1][i]-secondary_coord[3][i])**2)
        #     # print(dis)
        #     print('SU'+str(i)+': ',self.channelgain2(secondary_coord,secondary_coord,i,i))
        # print('Coord PU')
        # for i in range(self.primary_number):
        #     # dis=np.sqrt((primary_coord[0][i]-primary_coord[2][i])**2+(primary_coord[1][i]-primary_coord[3][i])**2)
        #     # print(dis)
        #     print('PU'+str(i)+': ',self.channelgain2(primary_coord,primary_coord,i,i))

        return self.network

    def get_state(self):
        state = self.obtain_state(self.decision, 0)
        state_num = len(state)
        '''state: 
             1-cr_router_energy_state1  
             2-reward                   1
             3-interference of SU       1
             4-power decision of SU     1
             5-spectrum decision of SU  1
             6-interference of PU       1'''
        return state_num

    def get_state_for_distributed_model(self):
        state = self.obtain_state_for_distributed_model(self.decision, 0)
        state_num = len(state)
        '''state: 
             1-cr_router_energy_state1  
             2-reward                   1
             3-interference of SU       1
             4-power decision of SU     1
             5-spectrum decision of SU  1
             6-interference of PU       1'''
        return state_num


    def get_state_for_joint_model(self):
        state = self.obtain_state_for_joint_model(self.decision, 0)
        state_num = len(state)
        '''state: 
             1-cr_router_energy_state1  
             2-reward                   1
             3-interference of SU       1
             4-power decision of SU     1
             5-spectrum decision of SU  1
             6-interference of PU       1'''
        return state_num

    def get_action(self):
        action = self.primary_number * self.power_set_number
        '''action: 
             1-spectrum
             2-power '''

        return action

    def create_network(self):
        '''产生模型'''
        '''Primary'''
        length = np.random.uniform(0, self.radius, self.primary_number)
        angel = np.random.uniform(0, 2 * np.pi, self.primary_number)
        x1 = length * np.cos(angel)
        y1 = length * np.sin(angel)

        length = np.random.uniform(self.distance[0], self.distance[1], self.primary_number)
        angel = np.random.uniform(0, 2 * np.pi, self.primary_number)

        x2 =x1+ length * np.cos(angel)
        y2 =y1+ length * np.sin(angel)

        primary_coord = [x1, y1, x2, y2]

        '''Secodary'''
        length = np.random.uniform(0, self.radius,self.secondary_number)
        angel = np.random.uniform(0, 2 * np.pi, self.secondary_number)
        x1 = length * np.cos(angel)
        y1 = length * np.sin(angel)

        length =  np.random.uniform(self.distance[2], self.distance[3], self.secondary_number)
        angel = np.random.uniform(0, 2 * np.pi, self.secondary_number)
        x2 =x1+length * np.cos(angel)
        y2 =y1+length * np.sin(angel)

        secondary_coord = [x1, y1, x2, y2]

        '''CR-router'''
        length = np.random.uniform(0, self.radius, self.CR_router_number)
        angel = np.random.uniform(0, 2 * np.pi, self.CR_router_number)
        x1 = length * np.cos(angel)
        y1 = length * np.sin(angel)
        length = np.random.uniform(0, self.radius, self.CR_router_number)
        angel = np.random.uniform(0, 2 * np.pi, self.CR_router_number)
        x2 = length * np.cos(angel)
        y2 = length * np.sin(angel)
        CR_router_coord = [x1, y1, x2, y2]

        '''保存数据'''
        network_model = [primary_coord, secondary_coord, CR_router_coord]

        # np.save(self.save_path + 'network_model', network_model)

        return network_model  # primary_coord,secondary_coord,CR_router_coord

    def plot_network(self,primary_coord,secondary_coord,CR_router_coord):
        ''''绘图'''
        marksize = 6
        # primary_coord = self.network[0]
        # secondary_coord = self.network[1]
        # CR_router_coord = self.network[2]

        plt.figure()
        plt.plot(primary_coord[0], primary_coord[1],
                 'r<', MarkerSize=marksize, label=u'PU_Tx')
        plt.plot(primary_coord[2], primary_coord[3],
                 'rs', MarkerSize=marksize, label=u'PU_Rx')

        plt.plot(secondary_coord[0], secondary_coord[1],
                 'g>', MarkerSize=marksize, label=u'SU_Tx')
        plt.plot(secondary_coord[2], secondary_coord[3],
                 'gs', MarkerSize=marksize, label=u'SU_Rx')

        plt.plot(CR_router_coord[0], CR_router_coord[1],
                 'bo', MarkerSize=marksize, label=u'CR-router')

        plt.legend(fontsize=12)

        plt.xlabel(u'x', fontsize=15)
        plt.ylabel(u'y', fontsize=15)
        plt.show()

    def user_distance(self):
        primary_coord = self.network[0]
        secondary_coord = self.network[1]



    def channelgain(self, coord1, coord2, pu_i, su_j):

        '''取出坐标'''
        Tx1 = coord1[0][pu_i]
        Ty1 = coord1[1][pu_i]

        Rx1 = coord1[2][pu_i]
        Ry1 = coord1[3][pu_i]

        '''传输链路距离'''
        dis_Tx1_Rx1 = np.sqrt((Tx1 - Rx1) ** 2 + (Ty1 - Ry1) ** 2)
        '''信道增益计算'''
        Gain_Tx1_Rx1 = np.power(dis_Tx1_Rx1, self.channel_gain)

        try:
            Tx2 = coord2[0][su_j]
            Ty2 = coord2[1][su_j]

            '''干扰链路距离'''
            dis_Tx2_Rx1 = np.sqrt((Tx2 - Rx1) ** 2 + (Ty2 - Ry1) ** 2)
            Gain_Tx2_Rx1 = np.power(dis_Tx2_Rx1, self.channel_gain)
        except Exception:
            '''干扰链路距离'''
            dis_Tx2_Rx1 = 0
            Gain_Tx2_Rx1 = 0
            pass

        return [Gain_Tx1_Rx1, Gain_Tx2_Rx1]

    def channelgain2(self, coord1, coord2, pu_i, su_j):

        '''取出坐标'''
        Tx1 = coord1[0][pu_i]
        Ty1 = coord1[1][pu_i]

        Rx1 = coord1[2][pu_i]
        Ry1 = coord1[3][pu_i]

        '''传输链路距离'''
        dis_Tx1_Rx1 = np.sqrt((Tx1 - Rx1) ** 2 + (Ty1 - Ry1) ** 2)
        '''信道增益计算'''
        Gain_Tx1_Rx1 = np.power(dis_Tx1_Rx1, self.channel_gain)

        try:
            Tx2 = coord2[0][su_j]
            Ty2 = coord2[1][su_j]

            '''干扰链路距离'''
            dis_Tx2_Rx1 = np.sqrt((Tx2 - Rx1) ** 2 + (Ty2 - Ry1) ** 2)
            Gain_Tx2_Rx1 = np.power(dis_Tx2_Rx1, self.channel_gain)
        except Exception:
            '''干扰链路距离'''
            dis_Tx2_Rx1 = 0
            Gain_Tx2_Rx1 = 0
            pass

        return dis_Tx1_Rx1
    ''''su的速率和'''

    def calculate_su_sinr_rate(self, vec):

        primary_coord = self.network[0]
        secondary_coord = self.network[1]

        # vec=self.decision

        ''' 
        # SU_spectrum_decision = vec[0] ==>频谱接入情况：vec[0][0]=1，则第一个SU接入第1号频谱
        # SU_power_decision = vec[1]  ==> 功率选择：vec[1][0]=1，则第一个SU接入第1号功率
        '''

        '''Secondary user's SINR and transmission rate'''
        SINR_SU = []
        Rate_SU = []
        Interference_SU = []

        for su_i in range(self.secondary_number):
            # print('\nSU_i:',i)
            PU_j = vec[0][su_i]  # 接入的频谱: 与PU_j共享的频谱号
            # su_power_selection=vec[1][i]
            # print('su_power_selection',vec[1][i])
            SU_power = self.power_set[vec[1][su_i]]  # 功率选择

            ''''Calculate the  interference from PU  that shares the same RB '''
            Gain_Tx1_Rx1_SU_i, Gain_PU_Tx2_SU_Rx1 = self.channelgain(secondary_coord, primary_coord, su_i, PU_j)
            interference_from_pu = self.primary_init_power * Gain_PU_Tx2_SU_Rx1

            ''''Calculate the total interference from othe SUs that share the same RB except for SU-i'''
            interference_from_su = []
            for su_k in range(self.secondary_number):
                if vec[0][su_k] == PU_j and su_k != su_i:  # 除了SU-i之外的SU对SU-i的干扰
                    # print('SU_k:',k)
                    Gain_Tx1_Rx1, Gain_SU_k_Tx2_SU_i_Rx1 = self.channelgain(secondary_coord, secondary_coord, su_i, su_k)
                    interference_from_su_k = self.power_set[vec[1][su_k]] * Gain_SU_k_Tx2_SU_i_Rx1
                    interference_from_su.append(interference_from_su_k)

            total_interference_from_su = sum(interference_from_su)

            '''SINR of SU-i'''
            SINR_SU_i = SU_power * Gain_Tx1_Rx1_SU_i /(total_interference_from_su + interference_from_pu + self.noise_power)

            SINR_SU.append(SINR_SU_i)

            '''Rate of SU-i'''
            rate_SU_i = self.bandwidth * np.log(1 + SINR_SU_i)
            Rate_SU.append(rate_SU_i)

            '''Interference of SU-i'''
            Interference_SU.append(total_interference_from_su + interference_from_pu+self.noise_power)

        return Rate_SU, SINR_SU, Interference_SU




    def calculate_pu_sinr_rate(self, vec):

        # vec=self.decision

        primary_coord = self.network[0]
        secondary_coord = self.network[1]
        CR_router_coord = self.network[2]

        '''Primary user's SINR and transmission rate'''
        SINR_PU = []
        Rate_PU = []
        Interference_PU = []

        for PU_j in range(self.primary_number):
            # print('\nPU_j:',PU_j)

            '''SINR of PU-j'''
            Gain_Tx1_Rx1_PU_j, Gain_Tx2_Rx1 = self.channelgain(primary_coord, primary_coord, PU_j, PU_j)

            ''''Calculate the total interference from  SUs that share the same RB'''
            interference_from_su = []
            for su_k in range(self.secondary_number):
                if vec[0][su_k] == PU_j:  # 占用同一频谱的全部SU的干扰功率
                    # print('SU_j:', k)
                    Gain_Tx1_Rx11, Gain_Tx2_Rx11 = self.channelgain(primary_coord, secondary_coord, PU_j, su_k)
                    interference_from_su_k = self.power_set[vec[1][su_k]] * Gain_Tx2_Rx11
                    interference_from_su.append(interference_from_su_k)
            total_interference_from_su = sum(interference_from_su)

            # print('total_interference_from_su:',total_interference_from_su)

            '''SINR of PU-j'''
            SINR_PU_j = self.primary_init_power * Gain_Tx1_Rx1_PU_j / (total_interference_from_su + self.noise_power)
            SINR_PU.append(SINR_PU_j)

            '''Rate of PU-j'''
            rate_PU_j = self.bandwidth * np.log(1 + SINR_PU_j)
            Rate_PU.append(rate_PU_j)

            '''Interference of PU-j'''
            Interference_PU.append((total_interference_from_su+self.noise_power))

        return Rate_PU, SINR_PU, Interference_PU

    def get_sum_rate(self):
        vec = self.decision
        Rate_SU, SINR_SU, Interference_SU = self.calculate_su_sinr_rate(vec)
        Rate_PU, SINR_PU, Interference_PU = self.calculate_pu_sinr_rate(vec)

        Sum_rate_PU, Sum_rate_SU = sum(Rate_PU), sum(Rate_SU)
        return Sum_rate_PU, Sum_rate_SU

    def get_sum_rate_test(self, vec):
        # vec=self.decision
        Rate_SU, SINR_SU, Interference_SU = self.calculate_su_sinr_rate(vec)
        Rate_PU, SINR_PU, Interference_PU = self.calculate_pu_sinr_rate(vec)

        Sum_rate_PU, Sum_rate_SU = sum(Rate_PU), sum(Rate_SU)

        # print('Rate_pu:',Rate_PU )
        # print('Interference_PU:',Interference_PU)
        #
        # print('Rate_su:',Rate_SU )

        return Sum_rate_PU, Sum_rate_SU, SINR_PU,Interference_PU

    def get_sum_rate_game(self, vec):
        # vec=self.decision
        Rate_SU, SINR_SU, Interference_SU = self.calculate_su_sinr_rate(vec)
        Rate_PU, SINR_PU, Interference_PU = self.calculate_pu_sinr_rate(vec)

        # Sum_rate_PU, Sum_rate_SU = sum(Rate_PU), sum(Rate_SU)

        # print('Rate_pu:',Rate_PU )
        # print('Interference_PU:',Interference_PU)
        #
        # print('Rate_su:',Rate_SU )

        return Rate_PU, Rate_SU#, SINR_PU, Interference_PU

    def init_alloation(self):
        vec = []
        spectrum_for_SU = np.random.randint(0, self.primary_number, self.secondary_number)
        power_for_SU = np.random.randint(0, self.power_set_number, self.secondary_number)
        vec.append(spectrum_for_SU)
        vec.append(power_for_SU)

        return vec

    def init_alloation_for_joint_model(self):
        vec = []
        spectrum_for_SU = np.random.randint(0, self.primary_number, self.secondary_number)
        power_for_SU = np.random.randint(0, self.power_set_number, self.secondary_number)
        vec.append(spectrum_for_SU)
        vec.append(power_for_SU)
        self.decision_for_joint_model=vec

        return vec


    def init_alloation_test(self):
        vec = []
        spectrum_for_SU = np.random.randint(0, self.primary_number, self.secondary_number)
        power_for_SU = np.random.randint(0, self.power_set_number, self.secondary_number)
        vec.append(spectrum_for_SU)
        vec.append(power_for_SU)

        return vec

    def action_mapping2(self):

        for su_i in range(self.secondary_number):
            pass

    def calculate_reward(self, vec):

        Rate_SU, SINR_SU, Interference_SU = self.calculate_su_sinr_rate(vec)
        Rate_PU, SINR_PU, Interference_PU = self.calculate_pu_sinr_rate(vec)
        # print('Rate_SU:',Rate_SU)
        # print('SINR_SU:',SINR_SU)
        # print('Interference_SU:',Interference_SU)
        # print('Rate_PU:',Rate_PU)
        # print('SINR_PU:',SINR_PU)
        # print('Interference_PU:',Interference_PU)
        # vec=self.decision
        # PU_j=vec[0][su_j]
        # print('SINR_PU[PU_j]: ', SINR_PU[PU_j])
        x1 = self.weight_reward[0] * sum(Rate_PU)
        x2 = self.weight_reward[1] * sum(Rate_SU)
        x3 = self.weight_reward[2] * (sum(Rate_PU) - self.primary_number * self.primary_rate_min)

        reward =10*(x1 + x2 + x3)/self.bandwidth
        # reward=1000*(x1+x2)/self.bandwidth

        # print('\nx1:',x1)
        # print('x2:',x2)
        # print('x3:',x3)
        # print('reward:',reward)

        return reward

    def calculate_reward_amend(self, vec, co_channel_pu):

        Rate_SU, SINR_SU, Interference_SU = self.calculate_su_sinr_rate(vec)
        Rate_PU, SINR_PU, Interference_PU = self.calculate_pu_sinr_rate(vec)
        # print('co_channel_pu:',co_channel_pu, '  Rate_PU[co_channel_pu]:',Rate_PU[co_channel_pu],' A:',Rate_PU[co_channel_pu] - self.primary_rate_min)
        # print('Rate_SU:',Rate_SU)
        # print('SINR_SU:',SINR_SU)
        # print('Interference_SU:',Interference_SU)
        # print('Rate_PU:',Rate_PU)
        # print('SINR_PU:',SINR_PU)
        # print('Interference_PU:',Interference_PU)
        # vec=self.decision
        # PU_j=vec[0][su_j]
        # print('SINR_PU[PU_j]: ', SINR_PU[PU_j])
        '''reward: sum-rate of IoT links + sum-rate of PU links + the MRQ of co-channel PU link'''
        x1 = self.weight_reward[0] * sum(Rate_PU)
        x2 = self.weight_reward[1] * sum(Rate_SU)
        x3 = self.weight_reward[2] * (Rate_PU[co_channel_pu] - self.primary_rate_min)

        x4 = self.weight_reward[3] * (sum(Rate_PU) - self.primary_number * self.primary_rate_min)

        reward = 10 * (x1 + x2 + x3+ x4) / self.bandwidth
        # reward=1000*(x1+x2)/self.bandwidth

        # print('\nx1:',x1)
        # print('x2:',x2)
        # print('x3:',x3)
        # print('reward:',reward)

        return reward

    def calculate_cr_router_state(self, vec):

        primary_coord = self.network[0]
        secondary_coord = self.network[1]
        CR_router_coord = self.network[2]

        # vec=self.decision

        cr_router_energy_state = []
        cr_router_energy_state1 = []
        for cr_router_i in range(self.CR_router_number):
            energy_detector_cr_router_i = []
            for PU_j in range(self.primary_number):
                Gain_Tx1_Rx11, Gain_Tx2_Rx11 = self.channelgain(CR_router_coord, primary_coord, cr_router_i, PU_j)
                '''PU'''
                energy_from_pu = self.primary_init_power * Gain_Tx2_Rx11

                energy_from_su = []
                for su_k in range(self.secondary_number):
                    if vec[0][su_k] == PU_j:  # 占用同一频谱的全部SU的干扰功率
                        # print('SU_j:', k)
                        Gain_Tx1_Rx11, Gain_Tx2_Rx11 = self.channelgain(CR_router_coord, secondary_coord, cr_router_i,
                                                                        su_k)
                        interference_from_su_k = self.power_set[vec[1][su_k]] * Gain_Tx2_Rx11
                        energy_from_su.append(interference_from_su_k)
                '''SU'''
                total_energy_from_su = sum(energy_from_su)

                energy_detector_cr_router_i.append((energy_from_pu + total_energy_from_su + self.noise_power))
                cr_router_energy_state1.append(-np.log(energy_from_pu + total_energy_from_su + self.noise_power) / 10)

            cr_router_energy_state.append(energy_detector_cr_router_i)

            # # cr_router_energy_state=list(np.concatenate(cr_router_energy_state))
            # cr_router_energy_state=list(np.array(cr_router_energy_state).flat)

        return cr_router_energy_state, cr_router_energy_state1

    def calculate_cr_router_state_for_CI_link(self, vec):
        primary_coord = self.network[0]
        secondary_coord = self.network[1]
        CR_router_coord = self.network[2]

        # vec=self.decision
        cr_router_energy_state_total = []
        cr_router_energy_state_individual = []
        for cr_router_i in range(self.secondary_number):
            energy_detector_cr_router_i = []
            for PU_j in range(self.primary_number):
                Gain_Tx1_Rx11, Gain_Tx2_Rx11 = self.channelgain(secondary_coord, primary_coord, cr_router_i, PU_j)
                '''PU'''
                energy_from_pu = self.primary_init_power * Gain_Tx2_Rx11

                energy_from_su = []
                for su_k in range(self.secondary_number):
                    if vec[0][su_k] == PU_j:  # 占用同一频谱的全部SU的干扰功率
                        # print('SU_j:', k)
                        Gain_Tx1_Rx11, Gain_Tx2_Rx11 = self.channelgain(secondary_coord, secondary_coord, cr_router_i,su_k)
                        interference_from_su_k = self.power_set[vec[1][su_k]] * Gain_Tx2_Rx11
                        energy_from_su.append(interference_from_su_k)
                '''SU'''
                total_energy_from_su = sum(energy_from_su)

                energy_detector_cr_router_i.append((total_energy_from_su) / (self.noise_power + energy_from_pu))

                '''所有CR-router的信息'''
                cr_router_energy_state_total.append((total_energy_from_su) / (self.noise_power + energy_from_pu))

            '''单独CR-router的信息'''
            cr_router_energy_state_individual.append(energy_detector_cr_router_i)

        return cr_router_energy_state_individual, cr_router_energy_state_total




    def calculate_cr_router_state_amend(self, vec):
        ''''''''
        ''''CR-router 只控制一定范围的频谱感应，不是全网式的搜索'''
        self.sense_coverage
        primary_coord = self.network[0]
        secondary_coord = self.network[1]
        CR_router_coord = self.network[2]

        # vec=self.decision

        cr_router_energy_state = []
        cr_router_energy_state1 = []
        for cr_router_i in range(self.CR_router_number):
            energy_detector_cr_router_i = []
            for PU_j in range(self.primary_number):
                Gain_Tx1_Rx11, Gain_Tx2_Rx11 = self.channelgain(CR_router_coord, primary_coord, cr_router_i, PU_j)
                '''PU'''
                '''只有在CR-router的覆盖面内才接收'''
                if Gain_Tx2_Rx11>=pow(self.sense_coverage,self.channel_gain):
                    energy_from_pu = self.primary_init_power * Gain_Tx2_Rx11
                else:
                    energy_from_pu=0

                energy_from_su = []
                for su_k in range(self.secondary_number):
                    if vec[0][su_k] == PU_j:  # 占用同一频谱的全部SU的干扰功率
                        # print('SU_j:', k)
                        Gain_Tx1_Rx11, Gain_Tx2_Rx11 = self.channelgain(CR_router_coord, secondary_coord, cr_router_i,su_k)

                        '''只有在CR-router的覆盖面内才接收'''
                        if Gain_Tx2_Rx11 >=pow(self.sense_coverage, self.channel_gain):
                            interference_from_su_k = self.power_set[vec[1][su_k]] * Gain_Tx2_Rx11
                        else:
                            interference_from_su_k=0

                        energy_from_su.append(interference_from_su_k)
                '''SU'''
                total_energy_from_su = sum(energy_from_su)

                energy_detector_cr_router_i.append((energy_from_pu + total_energy_from_su + self.noise_power))
                # cr_router_energy_state1.append(-np.log(energy_from_pu + total_energy_from_su + self.noise_power) / 10)
                # ''''CR-router上探测到的各个频谱的接收信噪比，作为状态信息'''
                # cr_router_energy_state1.append((energy_from_pu + total_energy_from_su ) / self.noise_power)

                ''''CR-router上探测到的各个频谱的PU/SU+N接收信噪比，作为状态信息'''
                cr_router_energy_state1.append((  total_energy_from_su) / (self.noise_power+ energy_from_pu))


            cr_router_energy_state.append(energy_detector_cr_router_i)

            # # cr_router_energy_state=list(np.concatenate(cr_router_energy_state))
            # cr_router_energy_state=list(np.array(cr_router_energy_state).flat)

        return cr_router_energy_state, cr_router_energy_state1

    def obtain_state_for_distributed_model(self, vec, su_j):
        # vec=self.decision
        # cr_router_energy_state, cr_router_energy_state1 = self.calculate_cr_router_state(vec)
        cr_router_energy_state_individual, cr_router_energy_state_total= self.calculate_cr_router_state_for_CI_link(vec)

        # if su_j+1 < self.secondary_number:
        #     su_j=su_j+1
        # else:
        #     su_j=0
        state=cr_router_energy_state_individual[su_j]

        # reward = self.calculate_reward(vec)

        return state


    def obtain_state(self, vec, su_j):
        # vec=self.decision
        # cr_router_energy_state, cr_router_energy_state1 = self.calculate_cr_router_state(vec)
        cr_router_energy_state, cr_router_energy_state1 = self.calculate_cr_router_state_amend(vec)

        # if su_j+1 < self.secondary_number:
        #     su_j=su_j+1
        # else:
        #     su_j=0

        reward = self.calculate_reward(vec)

        Rate_SU, SINR_SU, Interference_SU = self.calculate_su_sinr_rate(vec)
        Rate_PU, SINR_PU, Interference_PU = self.calculate_pu_sinr_rate(vec)

        '''state: 
             1-cr_router_energy_state1  
             2-reward                   1
             3-interference of SU       1
             4-power decision of SU     1
             5-spectrum decision of SU  1
             6-interference of PU       1'''
        # cr_router_energy_state1=np.log(cr_router_energy_state1)

        state = cr_router_energy_state1
        # state.append(reward)
        # state.append(-np.log(SINR_SU[su_j]))
        # state.append(-np.log(Interference_SU[su_j])/10)
        # state.append(self.power_set[vec[1][su_j]]*5)
        # state.append((vec[0][su_j]))
        # state.append(-np.log(Interference_PU[vec[0][su_j]])/10)

        # print('\n state:', state)

        return state


    def obtain_state_for_joint_model(self, vec, su_j):
        # vec=self.decision
        cr_router_energy_state, cr_router_energy_state1 = self.calculate_cr_router_state(vec)
        reward = self.calculate_reward(vec)

        Rate_SU, SINR_SU, Interference_SU = self.calculate_su_sinr_rate(vec)
        Rate_PU, SINR_PU, Interference_PU = self.calculate_pu_sinr_rate(vec)

        '''state: 
             1-cr_router_energy_state1  
             2-reward                   1
             3-interference of SU       1
             4-power decision of SU     1
             5-spectrum decision of SU  1
             6-interference of PU       1'''
        # cr_router_energy_state1=np.log(cr_router_energy_state1)

        state = cr_router_energy_state1
        state.append(su_j+1)

        # state.append(-np.log(SINR_SU[su_j]))
        # state.append(-np.log(Interference_SU[su_j])/10)
        # state.append(self.power_set[vec[1][su_j]]*5)
        # state.append((vec[0][su_j]))
        # state.append(-np.log(Interference_PU[vec[0][su_j]])/10)

        # print('\n state:', state)

        return state

    def action_mapping(self, act):

        act_vec = []
        for spe_i in range(self.primary_number):
            for pow_j in range(self.power_set_number):
                act_vec.append([spe_i, pow_j])

        return act_vec[act]

    def action_central_mode(self):
        action_number = self.primary_number * self.power_set_number
        L = list(range(0, action_number - 1))
        print('L:', L)
        L = list(permutations(L, action_number - 1))
        print('L:', L)

        # for su_i in range(self.secondary_number):
        #     for r in range():
        #         self.action_mapping(r)

    def evaluate_environmet_test(self, act, su_i):
        action = self.action_mapping(act)

        vec = self.decision_test

        '''new spectrum decision'''
        vec[0][su_i] = action[0]

        '''new power decision'''
        vec[1][su_i] = action[1]

        self.decision_test = vec

        return self.decision_test

    def evaluate_environmet(self, act, su_i):
        action = self.action_mapping(act)
        # print('action_map:',action)

        # vec=self.decision
        vec1 = self.decision_temp

        # print('Old Spect_SU:', vec[0])
        #         # print('Old Power_SU:', vec[1])

        '''new spectrum decision'''
        # vec[0][su_i]=action[0]
        vec1[0][su_i] = action[0]

        '''new power decision'''
        # vec[1][su_i]=action[1]
        vec1[1][su_i] = action[1]

        self.decision_temp = vec1
        # reward=self.calculate_reward(vec1)
        #
        # print('\nNew Spect_SU:', vec[0])
        # print('New Power_SU:', vec[1])

        return vec1  # , self.decision_temp#self.decision
    def calculate_fairness(self,init_decision):
        Sum_rate_PU, Sum_rate_SU, SINR_PU, Interference_PU = self.get_sum_rate_test(init_decision)
        Rate_PU, Rate_SU = self.get_sum_rate_game(init_decision)

        Fairness = sum(np.array(Rate_SU) ** 2) / (np.array(Sum_rate_SU) ** 2)

        return Fairness

    def environmet_for_joint_model(self,act,su_i):
        action = self.action_mapping(act)
        # # if su_i==0:
        # self.decision_for_joint_model=vec

        '''new spectrum decision'''
        # vec[0][su_i]=action[0]
        # vec[0][su_i] = action[0]
        self.decision_for_joint_model[0][su_i] = action[0]
        '''new power decision'''
        # vec[1][su_i]=action[1]
        # vec[1][su_i] = action[1]
        # print('\nspectrum:',action[0],'power:',action[1])
        self.decision_for_joint_model[1][su_i] = action[1]

        return   self.decision_for_joint_model
    def update_environment(self):
        return self.decision

    def evaluate_environmet_for_distributed_model(self, act, su_i):
        action = self.action_mapping(act)
        # print('action_map:',action)

        # vec=self.decision
        vec1 = self.decision_temp

        # print('Old Spect_SU:', vec[0])
        #         # print('Old Power_SU:', vec[1])

        '''new spectrum decision'''
        # vec[0][su_i]=action[0]
        vec1[0][su_i] = action[0]
        co_channel_pu=action[0]
        '''new power decision'''
        # vec[1][su_i]=action[1]
        vec1[1][su_i] = action[1]

        self.decision_temp = vec1
        reward = self.calculate_reward_amend(vec1,co_channel_pu)

        next_state = self.obtain_state_for_distributed_model(vec1, su_i)
        #
        # print('\nNew Spect_SU:', vec[0])
        # print('New Power_SU:', vec[1])

        return reward, next_state, vec1  # , self.decision_temp#self.decision


    def evaluate_environmet1(self, act, su_i):
        action = self.action_mapping(act)
        # print('action_map:',action)

        # vec=self.decision
        vec1 = self.decision_temp

        # print('Old Spect_SU:', vec[0])
        #         # print('Old Power_SU:', vec[1])

        '''new spectrum decision'''
        # vec[0][su_i]=action[0]
        vec1[0][su_i] = action[0]
        co_channel_pu=action[0]
        '''new power decision'''
        # vec[1][su_i]=action[1]
        vec1[1][su_i] = action[1]

        self.decision_temp = vec1
        reward = self.calculate_reward_amend(vec1,co_channel_pu)

        next_state = self.obtain_state(vec1, su_i)
        #
        # print('\nNew Spect_SU:', vec[0])
        # print('New Power_SU:', vec[1])

        return reward, next_state, vec1  # , self.decision_temp#self.decision

    def change_environmet_to_next_state(self):

        vec = self.decision_temp

        return vec

    def test_environmet(self):

        return self.decision_test