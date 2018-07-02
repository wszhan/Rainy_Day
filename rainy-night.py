import numpy as np
from collections import defaultdict
from itertools import permutations
import random

class RainyNight:
    """
    在一个雷暴雨的夜晚，你一个人开车到站台，看到了你的梦中情人、救命恩人、
    一个危重的病人，（他们都会开车）在这站台没有其他人，也没有其他的交通工具，
    如果从最优策略出发里应该如何处理，尝试编程实现这一过程。

    尝试从强化学习的角度出发，把每个人的状态定为（离开，留下）两种状态。
    为每一种状态考虑一个价值函数，于是该问题可以转化为找到价值最大的状态。

    另外，为每一个人的去、留定义奖励函数，此处主要考虑留下的不满程度（负数），
    离开的满意程度（正数），以及意愿的强烈程度（正数）。
    第一个值的范围是[-10, 0], 后两个值的范围都是[0, 10]。

    此外，在问题中没有提到，但是要考虑的是车的容量。车的容量的范围必须要是[1, 4]
    才有意义：0属于不可能，大于4的值在此没有意义。
    """
    def __init__(self, parties=4, max_capacity=2,
        random_reward=True, 
        happiness_range=[0, 10],
        unhappiness_range=[-10, 0],
        willingness_range=[0, 10],
        happiness=None,
        unhappiness=None,
        willingness=None):
        """
        参数：
        parties - int. 相关人数。
        max_capacity - int. 车的最大容量
        random_reward - boolean. 所有人的满意、不满意、意愿程度是否
            随机生成。默认为随机。如果不为随机的话，happiness, unhappiness, 
            willingness必须要是shape=(1,4)的NumPy array.

        Note:
        所有奖励函数list的顺序为[你，梦中情人，恩人，病危的人]。
        """
        self.max_capacity = max_capacity
        self.parties = parties # 生成states的参数

        # 是否随机生成满意程度、不满程度及意愿强度。
        if random_reward:
            self.happiness_range = happiness_range
            self.unhappiness_range = unhappiness_range
            self.willingness_range = willingness_range

            # 三个状态值
            self.leave_happy = self.happiness()
            self.stay_unhappy = self.unhappiness()
            self.decision_will = self.willingness()
        else:
            self.leave_happy = happiness
            self.stay_unhappy = unhappiness
            self.decision_will = willingness

        # 状态的数量
        factorial = lambda x : x * factorial(x - 1) if x != 0 else 1
        comb = lambda x, y : factorial(x) / (factorial(x - y) * factorial(y))
        self.combinations = comb(self.parties, self.max_capacity)

        # 生成状态
        self.stay_states, self.leave_states = self.generate_states()

        # 状态的价值 state values
        self.state_values = self.state_value_function()
        _, self.optimal_state = self.explore()


    # 通过价值函数和状态相乘，找到最大价值对应的状态
    def explore(self, decision_type='general'):
        """
        在这里决定选择离开的人。人数由车的载客量决定，属于客观因素。

        首先选出价值最高的状态。如果有价值相同的不同状态，都作为最后的参考。

        选出价值最高的状态后，一般来说有三种策略：
        - positive: 选择离开意愿最高，即状态价值最高、离开的话最为满意的人。
        - negative: 选择留下意愿最低，即状态价值最低、留下的话最为不满的人。
        - general: 直接根据状态确定谁去谁留。
        绝大多数情况下，三种状态的结果是一样的。这里默认为第三种策略。
        """
        state_values = np.sum(self.state_values, axis=1)

        # 找到最大价值的状态
        optimal_state_index = np.argmax(state_values)

        # 输出最大价值的状态
        return self.leave_states[optimal_state_index], self.state_values[optimal_state_index]

    # 价值函数
    def state_value_function(self):
        return self.decision_will * (self.leave_states * self.leave_happy + 
            self.stay_states * self.stay_unhappy)

    # 用字符串描述最佳策略
    def to_string(self):
        decision, _ = self.explore()
        persons = ["我", "梦中情人", "恩人", "病危的人"]

        leave_persons = list()

        for i in range(len(decision)):
            if decision[i] == 1:
                leave_persons.append(persons[i])

        print (u"最终结果:\n上车离开的人是", *leave_persons)
        print (u"考虑因素:")
        print (u"加上\"我\"在内，一共有", self.parties, "人")
        print (u"车的载客量是", self.max_capacity, "人")
        print (u"最希望能够离开的人是", persons[self.optimal_state.argmax()])
        print (u"最不希望留下的人是", persons[self.optimal_state.argmin()])
        print (u"最无所谓是否能够离开的人是", persons[np.abs(self.optimal_state).argmin()])
        
    ## Helper Functions
    def generate_states(self):
        """
        生成4个人分别的离开或留下的两种状态。用one-hot-encoding表示。

        [0, 1, 1, 0] 表示第二位（梦中情人）和第三位（恩人）留下。
        """
        parties = self.parties
        base_state = np.zeros(parties)
        i = 0
        while i < self.max_capacity:
            base_state[i] = 1
            i += 1;

        #print (base_state)

        # list of all states
        perm = permutations(base_state, parties)

        # convert to list
        leave_states = list()
        for p in perm:
            #print (p)
            if list(p) not in leave_states:
                leave_states.append(list(p))

        leave_states = np.array(leave_states)
        stay_states = (leave_states - 2) // (-2)

        # 输出所有状态    
        return stay_states, leave_states

    def happiness(self, pos_reward=None):
        """
        每一个人的离开的满意程度。

        参数：
        pos_reward - list/tuple. 代表每个人离开的满意程度，为正数，绝对值
            越大表示越为满意。
        """
        mini, maxi = self.happiness_range[0], self.happiness_range[1]
        happy = np.random.randint(mini, maxi+1, 4)
        return happy

    def unhappiness(self, neg_reward=None):
        """
        每一个人的留下的不满程度。

        参数：
        neg_reward - list/tuple. 代表每个人留下的不满程度，为负数，绝对值
            越大表示越为不满。
        """
        mini, maxi = self.unhappiness_range[0], self.unhappiness_range[1]
        unhappy = np.random.randint(mini, maxi+1, 4)
        return unhappy

    def willingness(self, willingness=None):
        """
        每一个人的作出选择的意愿强度。

        参数：
        willingness - list/tuple. 代表每个人留下的不满程度，为负数，绝对值
            越大表示越为不满。
        """
        mini, maxi = self.willingness_range[0], self.willingness_range[1]
        will = np.random.randint(mini, maxi+1, 4)
        return will

    def duplicate_state(self, state, states):
        for s in states:
            if (s == state).all():
                return True
        return False


#### Unit Test ####
HAPPINESS = [6, 6, 7, 10]
UNHAPPINESS = [-6, -6, 5, 10]
WILLINGNESS = [5, 5, 5, 9]

rn = RainyNight(max_capacity=2)
#rn = RainyNight(max_capacity=2, random_reward=False,
#    happiness=HAPPINESS, unhappiness=UNHAPPINESS,
#    willingness=WILLINGNESS)
print (rn.optimal_state)
rn.to_string()