from pypower.api import case118, ppoption, runpf, printpf
# import pypower.idx_bus as idx_bus
from pypower.idx_bus import PD, QD, VM, VA, BUS_AREA
from pypower.idx_gen import PG, QG, VG
# import pypower.idx_gen as idx_gen
import pypower.api as pp
# from case118_my import case118_my
import numpy as np
import scipy.io as scio
import random
import os
import copy


# dir = r'E:\Users\hz\DQN_stastic'
# fail_dir = r'E:\Users\hz\DQN_stastic\sim_fail'
# abnormal_dir = r'E:\Users\hz\DQN_stastic\sim_abnormal'
dir = r'D:\DQN_stastic'
fail_dir = r'D:\DQN_stastic\sim_fail'
abnormal_dir = r'D:\DQN_stastic\sim_abnormal'
ppc = case118()
# ppc['gen'][:,PG]=ppc['gen'][:,PG]*0.9
# ppc['gen'][:,QG]=ppc['gen'][:,QG]*0.5
# #改变母线，必须改变相应的端电压设定值
# ppc['bus'][35,VM]=1.045
# ppc['bus'][[24,25,26,27,28],VM]=1

ppc['gen'][0:27, VG] = 1.025   # 1.025
ppc['gen'][27:54, VG] = 0.975   # 0.975
ppc['gen'][4:16, VG] = 1.05
ppc['gen'][21:25, VG] = 1.05
# ppc['gen'][0, VG] = 0.985

propt = pp.ppoption(VERBOSE=0, OUT_ALL=0)
r = pp.runpf(ppc)
sim_fail = []
sim_abnormal_high = []
sim_abnormal_low = []
sim_abnormal = []
# ppopt = ppoption(PF_ALG=2)
# 负荷从80%增加到120%

PD_original = copy.deepcopy(ppc['bus'])
QD_original = copy.deepcopy(ppc['bus'])
# PD_original = ppc['bus'][:, PD].copy()
# QD_original = ppc['bus'][:, QD].copy()
record_arr = np.zeros([10000, 4])
bus_area_1 = []
bus_area_2 = []
bus_area_3 = []
high = 0
low = 0
fail = 0
for item0 in range(10000):
    # ppc['gen'][4:16, VG] = 1.05
    # ppc['gen'][21:25, VG] = 1.05
    print(item0)

    # 分片区调整负荷，增加样本丰富度
    for item in range(len(ppc['bus'][:, BUS_AREA])):
        if ppc['bus'][item, BUS_AREA] == 1:
            bus_area_1.append(item)
        elif ppc['bus'][item, BUS_AREA] == 2:
            bus_area_2.append(item)
        elif ppc['bus'][item, BUS_AREA] == 3:
            bus_area_3.append(item)

    a = random.randint(0, 1000)
    record_arr[item0, 0] = a
    ppc['bus'][bus_area_1, PD] = PD_original[bus_area_1, PD] * ((1.4 - 0.6) / 1000 * a + 0.6)
    ppc['bus'][bus_area_1, QD] = QD_original[bus_area_1, QD] * ((1.4 - 0.6) / 1000 * a + 0.6)

    b = random.randint(0, 1000)
    record_arr[item0, 1] = b
    ppc['bus'][bus_area_2, PD] = PD_original[bus_area_2, PD] * ((1.4 - 0.6) / 1000 * b + 0.6)
    ppc['bus'][bus_area_2, QD] = QD_original[bus_area_2, QD] * ((1.4 - 0.6) / 1000 * b + 0.6)

    c = random.randint(0, 1000)
    record_arr[item0, 2] = c
    ppc['bus'][bus_area_3, PD] = PD_original[bus_area_3, PD] * ((1.4 - 0.6) / 1000 * c + 0.6)
    ppc['bus'][bus_area_3, QD] = QD_original[bus_area_3, QD] * ((1.4 - 0.6) / 1000 * c + 0.6)

    r = pp.runpf(ppc, pp.ppoption(VERBOSE=0, OUT_ALL=0))
    print(max(r[0]['bus'][:, VM]))
    print(min(r[0]['bus'][:, VM]))

    # if r[0]['success'] == 0:   # 潮流不收敛(1)
    #     sim_fail.append(item0)
    #     print(1)
    #     pass
    # elif any(r[0]['bus'][:, VM] > 1.05):  # 电压不正常
    #     sim_abnormal.append(item0)
    #     sim_abnormal_high.append(item0)
    #     record_arr[item0, 3] = 2     # 偏高
    #     high = high + 1
    #     print(2)
    # elif any(r[0]['bus'][:, VM] < 0.95):  # 电压不正常
    #     sim_abnormal.append(item0)
    #     sim_abnormal_low.append(item0)
    #     record_arr[item0, 3] = 3  # 偏低
    #     low = low + 1
    #     print(3)
    #     pass
    # print('high=', high, 'low=', low)

    if r[0]['success'] == 0:   # 潮流不收敛(1)
        sim_abnormal.append(item0)
        sim_fail.append(item0)
        record_arr[item0, 3] = 1     # 不收敛
        fail = fail + 1
        print(1)
        pass
    elif any(r[0]['bus'][:, VM] > 1.05):  # 电压不正常
        sim_abnormal.append(item0)
        sim_abnormal_high.append(item0)
        record_arr[item0, 3] = 2     # 偏高
        high = high + 1
        print(2)
    elif any(r[0]['bus'][:, VM] < 0.95):  # 电压不正常
        sim_abnormal.append(item0)
        sim_abnormal_low.append(item0)
        record_arr[item0, 3] = 3  # 偏低
        low = low + 1
        print(3)
        pass
    print('fail=', fail, 'high=', high, 'low=', low)
    pass

# scio.savemat(r'D:\DQN_stastic\sim_abnormal_all.mat', {'sim_abnormal_all': record_arr,
#                                                        'sim_abnormal_high': sim_abnormal_high,
#                                                        'sim_abnormal_low': sim_abnormal_low})
scio.savemat(r'D:\DQN_stastic\sim_abnormal_all.mat', {'sim_abnormal_all': record_arr,
                                                       'sim_abnormal_high': sim_abnormal_high,
                                                       'sim_abnormal_low': sim_abnormal_low,
                                                       'sim_fail': sim_fail})
oc_str = r"D:\DQN_stastic"
opercond = scio.loadmat(oc_str + "\sim_abnormal_all.mat")
num = opercond["sim_abnormal_all"]
num = num[num[:, 3] != 0]
# num=sim_abnormal[702:]
# num=np.array(num)
random.seed(1)
random_num_train = random.sample(range(0, 7000), 6000)
train = num[random_num_train]
random_num_test = []
for item in range(7000):
    if item not in random_num_train:
        random_num_test.append(item)
test = num[random_num_test]
# scio.savemat(os.path.join(r'D:\DQN_stastic\sim_abnormal20220622v2', 'abnormal_num.mat'), {'sim_abnormal_train': train,
#                          'sim_abnormal_test': test})
scio.savemat(os.path.join(r'D:\DQN_stastic', 'abnormal_num.mat'), {'sim_abnormal_train': train,
                         'sim_abnormal_test': test})

# scio.savemat(fail_dir,{'sim_fail':sim_fail})

# for item in test:
#     if item in train:
#         print('Tr')
# else:print('----')

