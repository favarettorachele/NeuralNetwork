#%% ##################### Select type of data #################################
# 'all' = all input data are used as input (data are all of same lenght)
# 'max' = zeros are added to shorter data, all data are as long as the maximum one
# 'min' = data are cutted to mach the shorter data length
data_length = 'all'

mainfolder = 'G:\Shared drives\MST Supremo\FTH\Setup FBK-UNITN\\'
subfolder = 'neural network\\Noise 0.1 30 min 0.1-1 nM'
# subfolder = 'data\signal-reference\\2022-06-16-24, set of 14 rings\\same L same number files'

#%%######################## import libraries ##################################
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import analysis.loading as load
import analysis.extract as extract
import analysis.fit as fit
import torch
import torch.nn as nn
from skorch import NeuralNet

fit_fcn = {
        'linear': (fit.linear_fit, fit.linear_fit_guess),
        'binding-unbinding': (fit.baseline_binding_unbinding_fit, 
                              fit.baseline_binding_unbinding_fit_guess)}
colors = fit.colors


#%%########################### import filenames ###############################
folder = mainfolder + subfolder
all_file_names = os.listdir(folder)

data_list = []
for file in all_file_names:
    if 'binding' in file:
        data_list.append(file)
Nfile = len(data_list)

random.Random(1348).shuffle(data_list)

#%% #################### create variables to store data ######################
# concentartions arrays
c = np.zeros(Nfile)     #all
c_unique = []           #uniques

# shifts matrix
S = []

# train and test list filenames
train_list = []
test_list = []

# one test curve for each concentration
for test in data_list:
    c_unique.append(float(test.split(sep='_')[1].split(sep='nM')[0]))
c_unique = list(set(c_unique))

for conc in c_unique:
    conc = str(conc) + 'nM'
    endif = 0
    for file in data_list:
        if (conc in file) and endif == 0:
            test_list.append(file)
            endif = 1

train_list = [x for x in data_list if x not in test_list]
data_list = np.concatenate((train_list, test_list))
Ntrain = len(train_list)
Ntest = len(test_list)

print(f'\n N all files = {Nfile}, N train files = {Ntrain}, N test files = {Ntest} \n')


# %%########################### load data #####################################

Ndataall, Ndatamax, Ndatamin = 0, 0, 10000

for ntest, test in enumerate(data_list):

    # print(ntest, test, Nfile)
    c[ntest] = float(test.split(sep='_')[1].split(sep='nM')[0])
    filepath = os.path.join(folder,test)
    
    data = np.genfromtxt(filepath, delimiter=',' )
    t_minutes = data[:,0]
    s = data[:,1]

    if data_length == 'all':
        m_reducing_factor = 1
        shift = s.reshape(-1, m_reducing_factor).mean(axis=1)

    if data_length == 'max':
        start_bin, *_ = extract.binding_boundaries( t_minutes, s, 20, 2, 0, 0)
        shift = s[start_bin:]

    if data_length == 'min':
        start_bin, stop_bin, *_ = extract.binding_boundaries( t_minutes, s, 20, 2, 0, 0)
        shift = s[start_bin:stop_bin]

    # start from shift = 0
    # shift = shift - shift[0]
    # # normalization
    # argmax_shift = np.argmax(shift)
    # shift = shift/np.mean(shift[argmax_shift-2:argmax_shift])
    #comvolution
    # shift = np.convolve(shift, np.ones(10)/10, mode='same')
    # shift = shift[5:-6]

    S.append(shift)
        
    if Ndatamax < len(S[-1]):
        Ndatamax = len(S[-1])
    if Ndatamin > len(S[-1]):
        Ndatamin = len(S[-1])
Ndataall = len(shift)


if data_length == 'all':
    Npoints_curves_input = Ndataall
if data_length == 'max':
    Npoints_curves_input = Ndatamax + 10
if data_length == 'min':
    Npoints_curves_input = Ndatamin

print(f'N data input (points of each curve)= {Npoints_curves_input}')

#%% ################## create X and c train and test arrays ###################
# train and test concentration arrays
c_train = c[0:Ntrain] # shape (Nfile, 1)
c_test = c[Ntrain:]
shift_train = np.zeros((Ntrain, Npoints_curves_input))
shift_test = np.zeros((Ntest, Npoints_curves_input))

if data_length == 'all':
    for i in range(0, Ntrain):
        shift_train[i]= S[i]
    for i in range(0, Ntest):
        shift_test[i] = S[Ntrain+i] 

if data_length == 'max':
    for i in range(0, Ntrain):
        shift_train[i][0:len(S[i])] = S[i]
    for i in range(0, Ntest):
        shift_test[i][0:len(S[Ntrain+i])] = S[Ntrain+i]  

if data_length == 'min':
    for i in range(0, Ntrain):
        shift_train[i] = S[i][0:Npoints_curves_input]
    for i in range(0, Ntest):
        shift_test[i] = S[Ntrain+i][0:Npoints_curves_input]

# train and test arrays plot
fig_ex, ax_ex = plt.subplots()
xplot = np.arange(0, Npoints_curves_input)*(t_minutes[2]-t_minutes[1])*m_reducing_factor
for datanum in np.arange(0, Ntrain):
    ax_ex.plot(xplot, shift_train[datanum], color='tab:gray', alpha=0.2)
for datanum in np.arange(0, Ntest):
    clabel = np.round(c_test[datanum], 1)
    ax_ex.plot(xplot, shift_test[datanum], label = f'{clabel} nM', color=colors[datanum+1])
ax_ex.set_xlabel('Time [min]', fontsize=12)
ax_ex.set_ylabel('Shift [pm]', fontsize=12)
ax_ex.legend(fontsize=12)
ax_ex.grid()
plt.pause(0.1)


# %% ###################### initialize the neural network ############################
class NN_Class(nn.Module):
    
    def __init__(self, Ni=Npoints_curves_input, Nh1=9, Nh2=3, Nh3=3, No=1):

        super().__init__()
        
        self.fc1 = nn.Linear(in_features=Ni, out_features=Nh1, bias=True) #`y = xA^T + b
        # self.fc2 = nn.Linear(in_features=Nh1, out_features=Nh2, bias=True)
        # self.fc3 = nn.Linear(in_features=Nh2, out_features=Nh3, bias=True)
        self.out = nn.Linear(in_features=Nh1, out_features=No)
        self.act = nn.ReLU() #rectified linear unit function
    
        # self.values = [Ni, 120, 27, 9, 4]
        # self.out = nn.Linear(in_features=self.values[-1], out_features=1)
        # self.act = nn.ReLU() #rectified linear unit function
        # self.functions = []
        # self.last_value = self.values[0]
        # for node in self.values[1:]:
        #     print(f"self.act(nn.Linear(in_features={self.last_value}, out_features={node}, bias=True)(activated_data))")
        #     self.functions.append(nn.Linear(in_features=self.last_value, out_features=node, bias=True))
        #     self.last_value = node

    
    print('Network initialized')


    def forward(self, shifts_data):
        activated_data = self.act(shifts_data)
        for function in self.functions:
            activated_data = self.act(function(activated_data))
        x = self.out(activated_data)
        return x



    def forward(self, x):
        x = x #self.flatten(x)
        x = self.act(x)
        x = self.act(self.fc1(x))
        # x = self.act(self.fc2(x))
        # x = self.act(self.fc3(x))
        x = self.out(x)
        return x

    # def forward(self, shifts_data):
    #     activated_data = self.act(shifts_data)
    #     for function in self.functions:
    #         activated_data = self.act(function(activated_data))
    #     x = self.out(activated_data)
    #     return x


# %%
torch.manual_seed(46)

net = NeuralNet(
    module = NN_Class,
    batch_size = Ntrain,
    max_epochs=1500,
    lr=0.01,
    optimizer=torch.optim.SGD,
    criterion=torch.nn.MSELoss, #default one
    verbose = 0
)

# %%
c_train.reshape(Ntrain,1)
c_test.reshape(Ntest,1)
Xtrain_net = torch.from_numpy(shift_train.astype(np.float32)) #.astype(np.float32)
ytrain_net = torch.from_numpy(c_train.astype(np.float32)).unsqueeze(1) # c_train.astype(np.float32).unsqueeze(1)
net.fit(Xtrain_net, ytrain_net)

A= list(net.module_.parameters())
weights= A[0][0]

fig_w, ax_w = plt.subplots()
ax_w.plot(np.abs(weights.detach().numpy()), 'o', linestyle='dotted')
ax_w.set_xlabel('Data', fontsize=12)
ax_w.set_ylabel('Weights', fontsize=12)
ax_w.grid()
plt.pause(0.1)
# %%

y_pred_train = net.predict(Xtrain_net)
Xtest_net = torch.from_numpy(shift_test.astype(np.float32))
y_pred_test = net.predict(Xtest_net)

# print(f'Train concentrations \n Real:     Predicted:')
# for j in range(0, Ntrain):
#     pred_train = np.concatenate(y_pred_train)[j]
#     print(f'{ctrain[j]}     {"%.2f" %  pred_train}')

# print(f'Test concentrations \n Real:     Predicted:')
# for j in range(0, Ntest):
#     pred_test = np.concatenate(y_pred_test)[j]
#     print(f'{ctest[j]}     {"%.2f" %  pred_test}')


# %%

realarray = np.linspace(start = np.min(c_train), stop = np.max(c_train))

fig_train, ax_train = plt.subplots()
ax_train.plot(c_train, np.concatenate(y_pred_train), 'o', label='Predicted')
ax_train.plot(realarray, realarray, linestyle='dashed', label='Real')
ax_train.set_xlabel('Real concentration [nM]', fontsize=12)
ax_train.set_ylabel('Predicted concentration [nM]', fontsize=12)
ax_train.set_title('Predicted concentrations for train curves', fontsize=12)
ax_train.grid()
ax_train.legend(fontsize=12)
plt.pause(0.1)


fig_test, ax_test = plt.subplots()
ax_test.plot(c_test, np.concatenate(y_pred_test), 'o', label='Predicted')
p = np.polyfit(c_test, np.concatenate(y_pred_test), 1)
ax_test.plot(realarray, realarray, linestyle='dashed', label='Real')
ax_test.plot(c_test, p[0]*c_test + p[1], '-', label=f'Fit')
ax_test.set_xlabel('Real concentration [nM]', fontsize=12)
ax_test.set_ylabel('Predicted concentration [nM]', fontsize=12)
ax_test.set_title('Predicted concentrations for test curves', fontsize=12)
ax_test.grid()
ax_test.legend(fontsize=12)
plt.show()


# %%
