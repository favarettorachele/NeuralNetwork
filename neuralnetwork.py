#%%
max_or_min = 'all'

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
# import pandas
import random
import analysis.loading as load
import analysis.extract as extract
import scipy.optimize as sopt
import analysis.fit as fit
fit_fcn = {
        'linear': (fit.linear_fit, fit.linear_fit_guess),
        'binding-unbinding': (fit.baseline_binding_unbinding_fit, 
                              fit.baseline_binding_unbinding_fit_guess)}
import torch
import torch.nn as nn
from skorch import NeuralNet
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import confusion_matrix, accuracy_score
import hiddenlayer as hl
import seaborn as sns
cubehelix = sns.color_palette("hls", n_colors=20)
colors = fit.colors


#%%
folder = 'G:\Shared drives\MST Supremo\FTH\Setup FBK-UNITN\\neural network\\Noise 0.1 30 min 0.1-1 nM'
# folder = 'G:\Shared drives\MST Supremo\FTH\Setup FBK-UNITN\data\signal-reference\\2022-06-16-24, set of 14 rings\\same L same number files'
all_file_names = os.listdir(folder)

data_list = []
for file in all_file_names:
    if 'binding' in file:
        # if '10nM' not in file:
            # if '7nM' not in file:
    # if 'binding1_gly_x0' in file:
        data_list.append(file)
Nfile = len(data_list)
# print(Nfile)

c = np.zeros(Nfile)
S = []
Ndatamax = 0
Ndatamin = 10000

#%%
random.Random(1348).shuffle(data_list) #143 #542

train_list = []
test_list = []

# for conc in [1, 3, 5, 7, 10]:
for conc in np.arange(0.1, 1.1, 0.1):
    conc = str(conc) + 'nM'
    endif = 0
    for file in data_list:
        if (conc in file) and endif == 0:
            test_list.append(file)
            endif = 1
            # print(file)

train_list = [x for x in data_list if x not in test_list]

Ntrain = len(train_list)
Ntest = len(test_list)

print(f'\n Nfile = {Nfile}, Ntrain = {Ntrain}, Ntest = {Ntest} \n')

data_list = np.concatenate((train_list, test_list))
#%%
for ntest, test in enumerate(data_list):

    # print(ntest, test, Nfile)
    c[ntest] = float(test.split(sep='_')[1].split(sep='nM')[0])
    x = load.load_R_diff( filepath=(folder,test) )[0]
    y = load.load_R_diff( filepath=(folder,test) )[1]

    if max_or_min == 'all':
        m_reducing_factor = 1
        shift = y.reshape(-1, m_reducing_factor).mean(axis=1)
        # argmax_shift = np.argmax(shift)
        # shift = shift/np.mean(shift[argmax_shift-2:argmax_shift])
        ka = 0.035
        kd = 0.070
        Rmax = 1
        # shift = shift*(ka*c[ntest] * Rmax / (ka*c[ntest] + kd))
        Ndataall = len(shift)
        S.append(shift)
    
    if max_or_min == 'max':
        start_bin, *_ = extract.binding_boundaries( x, y, 20, 2, 0, 0)
        shift = y[start_bin:]
        shift = shift - shift[0]
        argmax_shift = np.argmax(shift)
        shift = shift/np.mean(shift[argmax_shift-2:argmax_shift])
        S.append(shift)
        if Ndatamax < len(S[-1]):
            Ndatamax = len(S[-1])

    if max_or_min == 'min':
        # pguess = fit_fcn['binding-unbinding'][1](x, y)
        # popt, pcov = sopt.curve_fit(fit_fcn['binding-unbinding'][0], x, y, p0=pguess)
        # start_bin = np.searchsorted(x, popt[0])
        # stop_bin = np.searchsorted(x, popt[2])
        start_bin, stop_bin, *_ = extract.binding_boundaries( x, y, 20, 2, 0, 0)
        shift = y[start_bin:stop_bin]
        shift = shift - shift[0]
        argmax_shift = np.argmax(shift)
        shift = shift/np.mean(shift[argmax_shift-2:argmax_shift])
        # shift = np.convolve(shift, np.ones(10)/10, mode='same')
        # shift = shift[5:-6]
        S.append(shift) 
        if Ndatamin > len(S[-1]):
            Ndatamin = len(S[-1])

if max_or_min == 'all':
    Npoints_curves_input = Ndataall
if max_or_min == 'max':
    Npoints_curves_input = Ndatamax
if max_or_min == 'min':
    Npoints_curves_input = Ndatamin


#%%
print(f'N data input (points of each curve)= {Npoints_curves_input}')

c_train = c[0:Ntrain] # shape (Nfile, 1)
c_test = c[Ntrain:]

if max_or_min == 'all':
    shift_train = np.zeros((Ntrain, Npoints_curves_input))
    for i in range(0, Ntrain):
        # print(len(traindata[i]), len(S[i]))
        shift_train[i]= S[i]

    shift_test = np.zeros((Ntest, Npoints_curves_input))
    for i in range(0, Ntest):
        shift_test[i] = S[Ntrain+i] 

if max_or_min == 'max':
    Npoints_curves_input = Ndatamax + 10
    shift_train = np.zeros((Ntrain, Npoints_curves_input))
    for i in range(0, Ntrain):
        shift_train[i][0:len(S[i])] = S[i]

    shift_test = np.zeros((Ntest, Npoints_curves_input))
    for i in range(0, Ntest):
        shift_test[i][0:len(S[Ntrain+i])] = S[Ntrain+i]  

if max_or_min == 'min':
    shift_train = np.zeros((Ntrain, Npoints_curves_input))
    for i in range(0, Ntrain):
        shift_train[i] = S[i][0:Npoints_curves_input]

    shift_test = np.zeros((Ntest, Npoints_curves_input))
    for i in range(0, Ntest):
        shift_test[i] = S[Ntrain+i][0:Npoints_curves_input]


fig_ex, ax_ex = plt.subplots()
xplot = np.arange(0, Npoints_curves_input)*(x[2]-x[1])*m_reducing_factor
for datanum in np.arange(0, Ntrain):
    ax_ex.plot(xplot, shift_train[datanum], color='tab:gray', alpha=0.2)# label = f'{outputdata[datanum]} nM')
for datanum in np.arange(0, Ntest):
    clabel = np.round(c_test[datanum], 1)
    ax_ex.plot(xplot, shift_test[datanum], label = f'{clabel} nM', color=colors[datanum+1])
ax_ex.set_xlabel('Time [min]', fontsize=12)
ax_ex.set_ylabel('Shift [pm]', fontsize=12)
ax_ex.legend(fontsize=12)
ax_ex.grid()
plt.pause(0.1)

# data_x = torch.tensor(inputdata) 
# data_y = torch.tensor(outputdata) 
# %%
class NN_Class(nn.Module):
    
    def __init__(self, Ni=Npoints_curves_input):

        super().__init__()
        
        print('Network initialized')
        self.values = [Ni, 120, 27, 9, 4]
        self.out = nn.Linear(in_features=self.values[-1], out_features=1)
        self.act = nn.ReLU() #rectified linear unit function
        self.functions = []
        self.last_value = self.values[0]
        for node in self.values[1:]:
            print(f"self.act(nn.Linear(in_features={self.last_value}, out_features={node}, bias=True)(activated_data))")
            self.functions.append(nn.Linear(in_features=self.last_value, out_features=node, bias=True))
            self.last_value = node


    def forward(self, shifts_data):
        activated_data = self.act(shifts_data)
        for function in self.functions:
            activated_data = self.act(function(activated_data))
        x = self.out(activated_data)
        return x

# ### Define train dataloader
# train_dataloader = DataLoader(traindata, batch_size=10, shuffle=True)
# ### Define test dataloader
# test_dataloader = DataLoader(testdata, batch_size=10, shuffle=False)

# batch_data, batch_labels = next(iter(train_dataloader))
# print(f"TRAIN BATCH SHAPE")
# print(f"\t Data: {batch_data.shape}")
# print(f"\t Labels: {batch_labels.shape}")

# batch_data, batch_labels = next(iter(test_dataloader))
# print(f"TEST BATCH SHAPE")
# print(f"\t Data: {batch_data.shape}")
# print(f"\t Labels: {batch_labels.shape}")

# model = NN_Class()

# transforms = [ hl.transforms.Prune('Constant') ] # Removes Constant nodes from graph.
# graph = hl.build_graph(model, args = batch_data.text, transforms=transforms)
# graph.theme = hl.graph.THEMES['blue'].copy()

# %%
torch.manual_seed(46)

net = NeuralNet(
    module = NN_Class,
    batch_size = Ntrain,
    max_epochs=200,
    lr=0.01,
    optimizer=torch.optim.SGD,
    criterion=torch.nn.MSELoss, #default one
    verbose = 1
)

# %%
net.fit(shift_train.astype(np.float32), c_train.astype(np.float32))

A= list(net.module_.parameters())
weights= A[0][0]

fig_w, ax_w = plt.subplots()
ax_w.plot(np.abs(weights.detach().numpy()), 'o', linestyle='dotted')
ax_w.set_xlabel('Data', fontsize=12)
ax_w.set_ylabel('Weights', fontsize=12)
ax_w.grid()
plt.pause(0.1)
# %%

y_pred_train = net.predict(shift_train.astype(np.float32))
y_pred_test = net.predict(shift_test.astype(np.float32))

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
