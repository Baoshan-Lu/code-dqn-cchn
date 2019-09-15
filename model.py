import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NatureNet(nn.Module):
    def __init__(self,states,actions,):
        super(NatureNet, self).__init__()

        # self.embeding_layer = nn.Embedding(states, 300)

        self.input_layer = nn.Sequential(
            nn.Linear(states, 256),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x=self.embeding_layer(x)

        feature = self.input_layer(x)
        actions_value = self.output_layer(feature)

        return actions_value

class DuelingNet(nn.Module):
    def __init__(self, states: int, actions: int):
        """Initialization."""
        super(DuelingNet, self).__init__()
        # set common feature layer
        # self.embeding_layer = nn.Embedding(states, 300)

        self.feature_layer = nn.Sequential(
            nn.Linear(states, 256),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, actions),
        )

        # set value layer
        self.value_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        # x=self.embeding_layer(x)

        feature = self.feature_layer(x)
        # feature = self.hidden_layer(feature)
        # print('--------------')
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))

        return x.sign().mul(x.abs().sqrt())


class NoisyNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(NoisyNet, self).__init__()

        self.feature = nn.Linear(in_dim, 256)
        self.noisy_layer1 = NoisyLinear(256, 128)
        self.noisy_layer2 = NoisyLinear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = F.relu(self.feature(x))
        hidden = F.relu(self.noisy_layer1(feature))
        out = self.noisy_layer2(hidden)

        return out

    def reset_noise(self):
        """Reset all noisy layers."""
        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()



class DQN(object):
    def __init__(self,parameters,actions,states,learningrate,batchsize):

        self.states = states
        self.action = actions
        self.loss_function=parameters.loss_function

        self.gpu_type = parameters.gpu_type
        self.save_path = parameters.save_path

        self.epsion = parameters.epsion
        # self.ENV_A_SHAPE = parameters.ENV_A_SHAPE
        self.targetnet_update_rate = parameters.targetnet_update_rate
        self.memory_capacity = parameters.memory_capacity
        # self.states = parameters.CR_router_number

        self.double_dqn = parameters.double_dqn
        self.dueling_dqn = parameters.dueling_dqn
        self.noisy_dqn=parameters.noisy_dqn


        self.pretrain = parameters.pretrain
        self.learning_rate = learningrate
        self.batchsize = batchsize

        # self.learning_rate = parameters.learning_rate
        # self.batchsize = parameters.batchsize

        self.gamma = parameters.gamma
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.learn_step_counter = 0  # 用于 target 更新计时
        self.memory_counter = 0  # 记忆库记数
        self.memory = np.zeros((self.memory_capacity, states * 2 + 2))  # 初始化记忆库

        self.count = 0

        ''''Nature DQN'''
        self.eval_net, self.target_net = NatureNet(states,actions).to(self.device), \
                                         NatureNet(states,actions).to(self.device)

        ''''Dueling DQN'''
        if self.dueling_dqn ==True:
            self.eval_net, self.target_net = DuelingNet(states, actions).to(self.device), \
                                             DuelingNet(states, actions).to(self.device),
        ''''Noisy DQN'''
        if self.noisy_dqn ==True:
            self.eval_net, self.target_net = NoisyNet(states, actions).to(self.device), \
                                             NoisyNet(states, actions).to(self.device),

        # if self.pretrain==True: #使用预训练模型
        #     self.eval_net, self.target_net = torch.load(self.save_path + 'eval_net'), \
        #                                      torch.load(self.save_path + 'eval_net')

        # ''''GPU训练'''
        # if self.gpu_type == True:
        #     self.eval_net, self.target_net=self.eval_net.cuda(), self.target_net.cuda()

        ''''优化器选择'''
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate) #优化器选择
        '''默认损失函数'''
        self.loss_func = nn.MSELoss()
        # Smooth L1
        # 相比于L1损失函数，可以收敛得更快。
        # 相比于L2损失函数，对离群点、异常值不敏感，梯度变化相对更小，训练时不容易跑飞。

        '''学习率调节'''
        self.scheduler_min = torch.optim.lr_scheduler.ReduceLROnPlateau\
            (self.optimizer, 'min', patience=parameters.lr_patience,
                                                         factor=parameters.lr_decay, threshold=1e-3)

        self.scheduler_max = torch.optim.lr_scheduler.ReduceLROnPlateau\
            (self.optimizer, 'max', patience=parameters.lr_patience,
                                                         factor=parameters.lr_decay, threshold=1e-3)
    def reset_learning_rate(self,learning_rate):
        self.learning_rate=learning_rate

    def reset_e_greedy_epsion(self, e_greedy_epsion=0.8):
        self.epsion=e_greedy_epsion


    def reset_model(self,model='double'):
        if model=='nature':
            self.double_dqn =False
            self.dueling_dqn =False
            self.noisy_dqn=False

        if model=='double':
            self.double_dqn =True
            self.dueling_dqn =False
            self.noisy_dqn=False

        if model=='dueling':
            self.double_dqn =False
            self.dueling_dqn =True
            self.noisy_dqn=False

        if model=='double_dueling':
            self.double_dqn =True
            self.dueling_dqn =True
            self.noisy_dqn=False

        if model=='noisy':
            self.double_dqn =False
            self.dueling_dqn =False
            self.noisy_dqn=True

        if model=='noisy_double':
            self.double_dqn =True
            self.dueling_dqn =False
            self.noisy_dqn=True
        # print('Double_dqn= ', self.double_dqn, '| dueling_dqn= ', self.dueling_dqn,'| noisy_dqn=',self.noisy_dqn)
    def choose_action(self, x,k):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(self.device)
        # if self.gpu_type == True:
        #     x=x.cuda()

        # 这里只输入一个 sample
        if np.random.uniform() < self.epsion*(1-k):   # 选最优动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].cpu().data.numpy()
            action = action[0]
        else:   # 选随机动作
            action = np.random.randint(0, self.action)
        return action

    def choose_max_Q_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(self.device)

        # if self.gpu_type == True:
        #     x=x.cuda()

            # 选最优动作
        actions_value1 = self.eval_net.forward(x)
        action1 = torch.max(actions_value1, 1)[1].cpu().data.numpy()
        action1 = action1[0]

        return action1

    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s, a, r, s_))

        # print(self.count, 's=',s,'a=',a, 'r=',r, 's_=',s_,'\ntransition=', transition)
        # print('transition: \n',transition)
        # r # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % self.memory_capacity

        # print('index=',index)
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新 Q现实
        # print('lr:',self.learning_rate)

        if self.learn_step_counter % self.targetnet_update_rate == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # print('self.learn_step_counter=',self.learn_step_counter)
        # print('self.memory_counter =',self.memory_counter )

        # 抽取记忆库中的批数据
        if self.memory_counter>self.memory_capacity:
            sample_index = np.random.choice(self.memory_capacity, self.batchsize)  #随机抽取记忆
        else:
            sample_index = np.random.choice(self.memory_counter, self.batchsize)  # 随机抽取记忆

        b_memory = self.memory[sample_index, :]

        # print(len(b_memory))
        b_s = torch.FloatTensor(b_memory[:, :self.states]).to(self.device) #当前状态,:N_STATES==取前 N_STATES 个状态的大小
        b_a = torch.LongTensor(b_memory[:, self.states:self.states+1].astype(int)).to(self.device) #拟采用的动作
        b_r = torch.FloatTensor(b_memory[:, self.states+1:self.states+2]).to(self.device) #获得的回报
        b_s_ = torch.FloatTensor(b_memory[:, -self.states:]).to(self.device) #采取此动作后进去的新的状态，-N_STATES: 倒取最后N_STATES

        # if self.gpu_type == True:
        #     b_s=b_s.cuda()
        #     b_a=b_a.cuda()
        #     b_r=b_r.cuda()
        #     b_s_=b_s_.cuda()
        # print(b_s.shape,b_a.shape,b_r.shape,b_s_.shape)

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)

        '''Q估计:过去的经验值'''
        q_eval = self.eval_net(b_s).gather(1, b_a)  # 输入当期状态，进行估计。shape (batch, 1)

        if self.double_dqn==True:
            # Double DQN
            q_next = self.target_net(b_s).gather(1, self.eval_net(b_s_).argmax(dim=1, keepdim=True)).detach()

        else:
            '''Q现实:真实回报+现实与估计的差距'''
            q_next = self.target_net(b_s_).detach()  # 输入对应新的状态获得真实回报。detach from graph, don't backpropagate 不进行反向传播
            self.count = self.count + 1

        '''就是预期获得的回报'''
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batchsize, 1)   # shape (batch, 1)'\nq_next=',q_next,

        # print(self.count,'\nq_eval=', q_eval.view(1, self.batchsize),'\nq_target=',q_target.view(1, self.batchsize))

        # print(self.count,'\nq_next.max(1)=', q_next,q_next.max(1),'\nq_next.max(1)[0]=',q_next.max(1)[0])
        if self.loss_function == 'MSE':
            loss = self.loss_func(q_eval, q_target)

        if self.loss_function == 'Smoothl1loss':
            loss = F.smooth_l1_loss(q_eval, q_target)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return  loss.cpu().detach().numpy()




class JointDQN(object):
    def __init__(self,parameters,
                 states=10,
                 actions=3,
                 user=3,
                 learningrate=0.01,
                 batchsize=32,
                 epsion=0.9,
                 gamma=0.9,
                 targetnet_update_rate=100,
                 memory_capacity=2000,
                 model='double_dueling'
                 ):
        # model = ['nature', 'double', 'dueling', 'double_dueling', 'noisy', 'noisy_double']

        self.states = states
        self.action = actions
        self.epsion = epsion
        self.learning_rate = learningrate
        self.batchsize = batchsize
        self.targetnet_update_rate = targetnet_update_rate
        self.memory_capacity = memory_capacity
        self.gamma = gamma
        self.model=model
        self.user=user

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.double_dqn = parameters.double_dqn
        self.dueling_dqn = parameters.dueling_dqn
        self.noisy_dqn=parameters.noisy_dqn

        self.learn_step_counter = 0  # 用于 target 更新计时
        self.memory_counter = 0  # 记忆库记数

        '''记忆库定义：
        n个用户
        state: 用户数*状态数=n*state
        1：CR-router的状态
        2：用户ID信息
        action=n*action
        reward=1
        next_state: 用户数*状态数=n*state
        '''
        # print('mem:',states *self.user+self.user+ 1+states *self.user)
        self.memory = np.zeros((self.memory_capacity, states *self.user+self.user+ 1+states *self.user ))  # 初始化记忆库


        ''''Nature DQN'''
        self.eval_net, self.target_net = NatureNet(states,actions).to(self.device), \
                                         NatureNet(states,actions).to(self.device)

        ''''Dueling DQN'''
        if self.dueling_dqn ==True:
            self.eval_net, self.target_net = DuelingNet(states, actions).to(self.device), \
                                             DuelingNet(states, actions).to(self.device),
        ''''Noisy DQN'''
        if self.noisy_dqn ==True:
            self.eval_net, self.target_net = NoisyNet(states, actions).to(self.device), \
                                             NoisyNet(states, actions).to(self.device),


        ''''优化器选择'''
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate) #优化器选择
        self.loss_func = nn.MSELoss()


    def choose_action(self, x,k):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(self.device)
        if np.random.uniform() < self.epsion*(1-k):   # 选最优动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].cpu().data.numpy()
            action = action[0]
        else:   # 选随机动作
            action = np.random.randint(0, self.action)
        return action

    def choose_max_Q_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(self.device)
        actions_value1 = self.eval_net.forward(x)
        action1 = torch.max(actions_value1, 1)[1].cpu().data.numpy()
        action1 = action1[0]

        return action1

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        # r # 如果记忆库满了, 就覆盖老数据
        # print('transition',len(transition))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):

        # target net 参数更新 Q现实
        if self.learn_step_counter % self.targetnet_update_rate == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # 抽取记忆库中的批数据
        if self.memory_counter>self.memory_capacity:
            sample_index = np.random.choice(self.memory_capacity, self.batchsize)  #随机抽取记忆
        else:
            sample_index = np.random.choice(self.memory_counter, self.batchsize)  # 随机抽取记忆

        b_memory = self.memory[sample_index, :]

        # print(len(b_memory))
        # b_s = torch.FloatTensor(b_memory[:, :self.states]).to(self.device) #当前状态,:N_STATES==取前 N_STATES 个状态的大小
        # b_a = torch.LongTensor(b_memory[:, self.states:self.states+1].astype(int)).to(self.device) #拟采用的动作
        # b_r = torch.FloatTensor(b_memory[:, self.states+1:self.states+2]).to(self.device) #获得的回报
        # b_s_ = torch.FloatTensor(b_memory[:, -self.states:]).to(self.device) #采取此动作后进去的新的状态，-N_STATES: 倒取最后N_STATES
        loss_=0
        for i in range(self.user):
            # b_s = memery[i * states:(i + 1) * states]
            # b_a = memery[user * states + i:user * states + i + 1]
            # b_r = memery[user * states + user:user * states + user + 1]
            # b_s_ = memery[user * states + user + 1 + i * states:user * states + user + 1 + (i + 1) * states]

            b_s = torch.FloatTensor(b_memory[:, i * self.states:    (i + 1) * self.states]).to(self.device) #当前状态,:N_STATES==取前 N_STATES 个状态的大小
            b_a = torch.LongTensor(b_memory[:, self.user * self.states + i:     self.user * self.states + i + 1].astype(int)).to(self.device) #拟采用的动作
            b_r = torch.FloatTensor(b_memory[:, self.user * self.states + self.user:    self.user * self.states + self.user + 1]).to(self.device) #获得的回报
            b_s_ = torch.FloatTensor(b_memory[:, self.user * self.states + self.user + 1 + i * self.states:
                                                 self.user * self.states + self.user + 1 + (i + 1) * self.states]).to(self.device) #采取此动作后进去的新的状态，-N_STATES: 倒取最后N_STATES


            # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)

            '''Q估计:过去的经验值'''
            q_eval = self.eval_net(b_s).gather(1, b_a)  # 输入当期状态，进行估计。shape (batch, 1)

            if self.double_dqn==True:
                # Double DQN
                q_next = self.target_net(b_s).gather(1, self.eval_net(b_s_).argmax(dim=1, keepdim=True)).detach()

            else:
                '''Q现实:真实回报+现实与估计的差距'''
                q_next = self.target_net(b_s_).detach()  # 输入对应新的状态获得真实回报。detach from graph, don't backpropagate 不进行反向传播
                # self.count = self.count + 1

            '''就是预期获得的回报'''
            q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batchsize, 1)   # shape (batch, 1)'\nq_next=',q_next,

            loss= self.loss_func(q_eval, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_=loss+loss_


        return  loss_.cpu().detach().numpy()



    def reset_e_greedy_epsion(self, e_greedy_epsion=0.8):
        self.epsion=e_greedy_epsion


    def reset_model(self,model='double'):
        if model=='nature':
            self.double_dqn =False
            self.dueling_dqn =False
            self.noisy_dqn=False

        if model=='double':
            self.double_dqn =True
            self.dueling_dqn =False
            self.noisy_dqn=False

        if model=='dueling':
            self.double_dqn =False
            self.dueling_dqn =True
            self.noisy_dqn=False

        if model=='double_dueling':
            self.double_dqn =True
            self.dueling_dqn =True
            self.noisy_dqn=False

        if model=='noisy':
            self.double_dqn =False
            self.dueling_dqn =False
            self.noisy_dqn=True

        if model=='noisy_double':
            self.double_dqn =True
            self.dueling_dqn =False
            self.noisy_dqn=True
        # print('Double_dqn= ', self.double_dqn, '| dueling_dqn= ', self.dueling_dqn,'| noisy_dqn=',self.noisy_dqn)
