import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import torch
####CASE1#######CCS
def cal_relative_error(f1,truth1):
    relative_error = []
    for i in range(40):
        error1 = np.abs(f1[:,:,i] - truth1[:,:,i]).flatten()
        error1_predict = np.sum(error1)/(128*128)
        truth1_tru = np.abs(truth1[:,:,i]).flatten()
        truth1_tru_s = np.sum(truth1_tru)/(128*128)
        relative_error.append(error1_predict/truth1_tru_s)
    relative_error_layer1 = np.array(relative_error)
    return relative_error_layer1

f1_layer1_50time_C1 = torch.load( "D:\pycharm\pytorch\FNO-net\CCS Predicted saturation for layer1 at T=50_250_450\Predicted saturation field at T=50timestep layer1.pth")
f1_layer1_50time_C1= np.array(f1_layer1_50time_C1.detach().cpu())
truth1_layer1_50time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Ground truth of saturaion field  for layer1 at T=50_250_450\Ground truth of saturation field at T=50timestep layer1.pth")
truth1_layer1_50time_C1 = np.array(truth1_layer1_50time_C1.detach().cpu())

f1_layer2_50time_C1 = torch.load( "D:\pycharm\pytorch\FNO-net\CCS Predicted saturation for layer2 at T=50_250_450\Predicted saturation field at T=50timestep layer2.pth")
f1_layer2_50time_C1 = np.array(f1_layer2_50time_C1.detach().cpu())
truth1_layer2_50time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Ground truth of saturaion field  for layer2 at T=50_250_450\Ground truth of saturation field at T=50timestep layer2.pth")
truth1_layer2_50time_C1 = np.array(truth1_layer2_50time_C1.detach().cpu())

f1_layer3_50time_C1 = torch.load( "D:\pycharm\pytorch\FNO-net\CCS Predicted saturation for layer3 at T=50_250_450\Predicted saturation field at T=50timestep layer3.pth")
f1_layer3_50time_C1 = np.array(f1_layer3_50time_C1.detach().cpu())
truth1_layer3_50time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Ground truth of saturaion field  for layer3 at T=50_250_450\Ground truth of saturation field at T=50timestep layer3.pth")
truth1_layer3_50time_C1 = np.array(truth1_layer3_50time_C1.detach().cpu())

f1_layer4_50time_C1 = torch.load( "D:\pycharm\pytorch\FNO-net\CCS Predicted saturation for layer4 at T=50_250_450\Predicted saturation field at T=50timestep layer4.pth")
f1_layer4_50time_C1 = np.array(f1_layer4_50time_C1.detach().cpu())
truth1_layer4_50time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Ground truth of saturaion field  for layer4 at T=50_250_450\Ground truth of saturation field at T=50timestep layer4.pth")
truth1_layer4_50time_C1 = np.array(truth1_layer4_50time_C1.detach().cpu())

f1_layer5_50time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Predicted saturation for layer5 at T=50_250_450\Predicted saturation field at T=50timestep layer5.pth")
f1_layer5_50time_C1 = np.array(f1_layer5_50time_C1.detach().cpu())
truth1_layer5_50time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Ground truth of saturaion field  for layer5 at T=50_250_450\Ground truth of saturation field at T=50timestep layer5.pth")
truth1_layer5_50time_C1 = np.array(truth1_layer5_50time_C1.detach().cpu())

#####
f1_layer1_250time_C1 = torch.load( "D:\pycharm\pytorch\FNO-net\CCS Predicted saturation for layer1 at T=50_250_450\Predicted saturation field at T=250timestep layer1.pth")
f1_layer1_250time_C1= np.array(f1_layer1_250time_C1.detach().cpu())
truth1_layer1_250time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Ground truth of saturaion field  for layer1 at T=50_250_450\Ground truth of saturation field at T=250timestep layer1.pth")
truth1_layer1_250time_C1 = np.array(truth1_layer1_250time_C1.detach().cpu())

f1_layer2_250time_C1 = torch.load( "D:\pycharm\pytorch\FNO-net\CCS Predicted saturation for layer2 at T=50_250_450\Predicted saturation field at T=250timestep layer2.pth")
f1_layer2_250time_C1 = np.array(f1_layer2_250time_C1.detach().cpu())
truth1_layer2_250time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Ground truth of saturaion field  for layer2 at T=50_250_450\Ground truth of saturation field at T=250timestep layer2.pth")
truth1_layer2_250time_C1 = np.array(truth1_layer2_250time_C1.detach().cpu())

f1_layer3_250time_C1 = torch.load( "D:\pycharm\pytorch\FNO-net\CCS Predicted saturation for layer3 at T=50_250_450\Predicted saturation field at T=250timestep layer3.pth")
f1_layer3_250time_C1 = np.array(f1_layer3_250time_C1.detach().cpu())
truth1_layer3_250time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Ground truth of saturaion field  for layer3 at T=50_250_450\Ground truth of saturation field at T=250timestep layer3.pth")
truth1_layer3_250time_C1 = np.array(truth1_layer3_250time_C1.detach().cpu())

f1_layer4_250time_C1 = torch.load( "D:\pycharm\pytorch\FNO-net\CCS Predicted saturation for layer4 at T=50_250_450\Predicted saturation field at T=250timestep layer4.pth")
f1_layer4_250time_C1 = np.array(f1_layer4_250time_C1.detach().cpu())
truth1_layer4_250time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Ground truth of saturaion field  for layer4 at T=50_250_450\Ground truth of saturation field at T=250timestep layer4.pth")
truth1_layer4_250time_C1 = np.array(truth1_layer4_250time_C1.detach().cpu())

f1_layer5_250time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Predicted saturation for layer5 at T=50_250_450\Predicted saturation field at T=250timestep layer5.pth")
f1_layer5_250time_C1 = np.array(f1_layer5_250time_C1.detach().cpu())
truth1_layer5_250time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Ground truth of saturaion field  for layer5 at T=50_250_450\Ground truth of saturation field at T=250timestep layer5.pth")
truth1_layer5_250time_C1 = np.array(truth1_layer5_250time_C1.detach().cpu())

#####
f1_layer1_450time_C1 = torch.load( "D:\pycharm\pytorch\FNO-net\CCS Predicted saturation for layer1 at T=50_250_450\Predicted saturation field at T=450timestep layer1.pth")
f1_layer1_450time_C1= np.array(f1_layer1_450time_C1.detach().cpu())
truth1_layer1_450time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Ground truth of saturaion field  for layer1 at T=50_250_450\Ground truth of saturation field at T=450timestep layer1.pth")
truth1_layer1_450time_C1 = np.array(truth1_layer1_450time_C1.detach().cpu())

f1_layer2_450time_C1 = torch.load( "D:\pycharm\pytorch\FNO-net\CCS Predicted saturation for layer2 at T=50_250_450\Predicted saturation field at T=450timestep layer2.pth")
f1_layer2_450time_C1 = np.array(f1_layer2_450time_C1.detach().cpu())
truth1_layer2_450time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Ground truth of saturaion field  for layer2 at T=50_250_450\Ground truth of saturation field at T=450timestep layer2.pth")
truth1_layer2_450time_C1 = np.array(truth1_layer2_450time_C1.detach().cpu())

f1_layer3_450time_C1 = torch.load( "D:\pycharm\pytorch\FNO-net\CCS Predicted saturation for layer3 at T=50_250_450\Predicted saturation field at T=450timestep layer3.pth")
f1_layer3_450time_C1 = np.array(f1_layer3_450time_C1.detach().cpu())
truth1_layer3_450time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Ground truth of saturaion field  for layer3 at T=50_250_450\Ground truth of saturation field at T=450timestep layer3.pth")
truth1_layer3_450time_C1 = np.array(truth1_layer3_450time_C1.detach().cpu())

f1_layer4_450time_C1 = torch.load( "D:\pycharm\pytorch\FNO-net\CCS Predicted saturation for layer4 at T=50_250_450\Predicted saturation field at T=450timestep layer4.pth")
f1_layer4_450time_C1 = np.array(f1_layer4_450time_C1.detach().cpu())
truth1_layer4_450time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Ground truth of saturaion field  for layer4 at T=50_250_450\Ground truth of saturation field at T=450timestep layer4.pth")
truth1_layer4_450time_C1 = np.array(truth1_layer4_450time_C1.detach().cpu())

f1_layer5_450time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Predicted saturation for layer5 at T=50_250_450\Predicted saturation field at T=450timestep layer5.pth")
f1_layer5_450time_C1 = np.array(f1_layer5_450time_C1.detach().cpu())
truth1_layer5_450time_C1 = torch.load("D:\pycharm\pytorch\FNO-net\CCS Ground truth of saturaion field  for layer5 at T=50_250_450\Ground truth of saturation field at T=450timestep layer5.pth")
truth1_layer5_450time_C1 = np.array(truth1_layer5_450time_C1.detach().cpu())




####CASE2############FRAC
f1_layer1_50time = torch.load( "D:\pycharm\pytorch\FNO-net\Frac CCS Predicted saturation for layer1 at T=50_250_450\Predicted saturation field at T=50timestep layer1.pth")
f1_layer1_50time= np.array(f1_layer1_50time.detach().cpu())
truth1_layer1_50time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Ground truth of saturaion field  for layer1 at T=50_250_450\Ground truth of saturation field at T=50timestep layer1.pth")
truth1_layer1_50time = np.array(truth1_layer1_50time.detach().cpu())

f1_layer2_50time = torch.load( "D:\pycharm\pytorch\FNO-net\Frac CCS Predicted saturation for layer2 at T=50_250_450\Predicted saturation field at T=50timestep layer2.pth")
f1_layer2_50time = np.array(f1_layer2_50time.detach().cpu())
truth1_layer2_50time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Ground truth of saturaion field  for layer2 at T=50_250_450\Ground truth of saturation field at T=50timestep layer2.pth")
truth1_layer2_50time = np.array(truth1_layer2_50time.detach().cpu())

f1_layer3_50time = torch.load( "D:\pycharm\pytorch\FNO-net\Frac CCS Predicted saturation for layer3 at T=50_250_450\Predicted saturation field at T=50timestep layer3.pth")
f1_layer3_50time = np.array(f1_layer3_50time.detach().cpu())
truth1_layer3_50time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Ground truth of saturaion field  for layer3 at T=50_250_450\Ground truth of saturation field at T=50timestep layer3.pth")
truth1_layer3_50time = np.array(truth1_layer3_50time.detach().cpu())

f1_layer4_50time = torch.load( "D:\pycharm\pytorch\FNO-net\Frac CCS Predicted saturation for layer4 at T=50_250_450\Predicted saturation field at T=50timestep layer4.pth")
f1_layer4_50time = np.array(f1_layer4_50time.detach().cpu())
truth1_layer4_50time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Ground truth of saturaion field  for layer4 at T=50_250_450\Ground truth of saturation field at T=50timestep layer4.pth")
truth1_layer4_50time = np.array(truth1_layer4_50time.detach().cpu())

f1_layer5_50time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Predicted saturation for layer5 at T=50_250_450\Predicted saturation field at T=50timestep layer5.pth")
f1_layer5_50time = np.array(f1_layer5_50time.detach().cpu())
truth1_layer5_50time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Ground truth of saturaion field  for layer5 at T=50_250_450\Ground truth of saturation field at T=50timestep layer5.pth")
truth1_layer5_50time = np.array(truth1_layer5_50time.detach().cpu())

#####
f1_layer1_250time = torch.load( "D:\pycharm\pytorch\FNO-net\Frac CCS Predicted saturation for layer1 at T=50_250_450\Predicted saturation field at T=250timestep layer1.pth")
f1_layer1_250time= np.array(f1_layer1_250time.detach().cpu())
truth1_layer1_250time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Ground truth of saturaion field  for layer1 at T=50_250_450\Ground truth of saturation field at T=250timestep layer1.pth")
truth1_layer1_250time = np.array(truth1_layer1_250time.detach().cpu())

f1_layer2_250time = torch.load( "D:\pycharm\pytorch\FNO-net\Frac CCS Predicted saturation for layer2 at T=50_250_450\Predicted saturation field at T=250timestep layer2.pth")
f1_layer2_250time = np.array(f1_layer2_250time.detach().cpu())
truth1_layer2_250time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Ground truth of saturaion field  for layer2 at T=50_250_450\Ground truth of saturation field at T=250timestep layer2.pth")
truth1_layer2_250time = np.array(truth1_layer2_250time.detach().cpu())

f1_layer3_250time = torch.load( "D:\pycharm\pytorch\FNO-net\Frac CCS Predicted saturation for layer3 at T=50_250_450\Predicted saturation field at T=250timestep layer3.pth")
f1_layer3_250time = np.array(f1_layer3_250time.detach().cpu())
truth1_layer3_250time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Ground truth of saturaion field  for layer3 at T=50_250_450\Ground truth of saturation field at T=250timestep layer3.pth")
truth1_layer3_250time = np.array(truth1_layer3_250time.detach().cpu())

f1_layer4_250time = torch.load( "D:\pycharm\pytorch\FNO-net\Frac CCS Predicted saturation for layer4 at T=50_250_450\Predicted saturation field at T=250timestep layer4.pth")
f1_layer4_250time = np.array(f1_layer4_250time.detach().cpu())
truth1_layer4_250time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Ground truth of saturaion field  for layer4 at T=50_250_450\Ground truth of saturation field at T=250timestep layer4.pth")
truth1_layer4_250time = np.array(truth1_layer4_250time.detach().cpu())

f1_layer5_250time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Predicted saturation for layer5 at T=50_250_450\Predicted saturation field at T=250timestep layer5.pth")
f1_layer5_250time = np.array(f1_layer5_250time.detach().cpu())
truth1_layer5_250time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Ground truth of saturaion field  for layer5 at T=50_250_450\Ground truth of saturation field at T=250timestep layer5.pth")
truth1_layer5_250time = np.array(truth1_layer5_250time.detach().cpu())

#####
f1_layer1_450time = torch.load( "D:\pycharm\pytorch\FNO-net\Frac CCS Predicted saturation for layer1 at T=50_250_450\Predicted saturation field at T=450timestep layer1.pth")
f1_layer1_450time= np.array(f1_layer1_450time.detach().cpu())
truth1_layer1_450time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Ground truth of saturaion field  for layer1 at T=50_250_450\Ground truth of saturation field at T=450timestep layer1.pth")
truth1_layer1_450time = np.array(truth1_layer1_450time.detach().cpu())

f1_layer2_450time = torch.load( "D:\pycharm\pytorch\FNO-net\Frac CCS Predicted saturation for layer2 at T=50_250_450\Predicted saturation field at T=450timestep layer2.pth")
f1_layer2_450time = np.array(f1_layer2_450time.detach().cpu())
truth1_layer2_450time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Ground truth of saturaion field  for layer2 at T=50_250_450\Ground truth of saturation field at T=450timestep layer2.pth")
truth1_layer2_450time = np.array(truth1_layer2_450time.detach().cpu())

f1_layer3_450time = torch.load( "D:\pycharm\pytorch\FNO-net\Frac CCS Predicted saturation for layer3 at T=50_250_450\Predicted saturation field at T=450timestep layer3.pth")
f1_layer3_450time = np.array(f1_layer3_450time.detach().cpu())
truth1_layer3_450time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Ground truth of saturaion field  for layer3 at T=50_250_450\Ground truth of saturation field at T=450timestep layer3.pth")
truth1_layer3_450time = np.array(truth1_layer3_450time.detach().cpu())

f1_layer4_450time = torch.load( "D:\pycharm\pytorch\FNO-net\Frac CCS Predicted saturation for layer4 at T=50_250_450\Predicted saturation field at T=450timestep layer4.pth")
f1_layer4_450time = np.array(f1_layer4_450time.detach().cpu())
truth1_layer4_450time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Ground truth of saturaion field  for layer4 at T=50_250_450\Ground truth of saturation field at T=450timestep layer4.pth")
truth1_layer4_450time = np.array(truth1_layer4_450time.detach().cpu())

f1_layer5_450time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Predicted saturation for layer5 at T=50_250_450\Predicted saturation field at T=450timestep layer5.pth")
f1_layer5_450time = np.array(f1_layer5_450time.detach().cpu())
truth1_layer5_450time = torch.load("D:\pycharm\pytorch\FNO-net\Frac CCS Ground truth of saturaion field  for layer5 at T=50_250_450\Ground truth of saturation field at T=450timestep layer5.pth")
truth1_layer5_450time = np.array(truth1_layer5_450time.detach().cpu())

####case1 relative_error#######
relative_error_layer1_C1 =cal_relative_error(f1_layer1_50time_C1,truth1_layer1_50time_C1)
relative_error_layer2_C1 =cal_relative_error(f1_layer2_50time_C1,truth1_layer2_50time_C1)
relative_error_layer3_C1 =cal_relative_error(f1_layer3_50time_C1,truth1_layer3_50time_C1)
relative_error_layer4_C1 =cal_relative_error(f1_layer4_50time_C1,truth1_layer4_50time_C1)
relative_error_layer5_C1 =cal_relative_error(f1_layer5_50time_C1,truth1_layer5_50time_C1)

relative_error_layer1_250time_C1 =cal_relative_error(f1_layer1_250time_C1,truth1_layer1_250time_C1)
relative_error_layer2_250time_C1 =cal_relative_error(f1_layer2_250time_C1,truth1_layer2_250time_C1)
relative_error_layer3_250time_C1 =cal_relative_error(f1_layer3_250time_C1,truth1_layer3_250time_C1)
relative_error_layer4_250time_C1 =cal_relative_error(f1_layer4_250time_C1,truth1_layer4_250time_C1)
relative_error_layer5_250time_C1 =cal_relative_error(f1_layer5_250time_C1,truth1_layer5_250time_C1)

relative_error_layer1_450time_C1 =cal_relative_error(f1_layer1_450time_C1,truth1_layer1_450time_C1)
relative_error_layer2_450time_C1 =cal_relative_error(f1_layer2_450time_C1,truth1_layer2_450time_C1)
relative_error_layer3_450time_C1 =cal_relative_error(f1_layer3_450time_C1,truth1_layer3_450time_C1)
relative_error_layer4_450time_C1 =cal_relative_error(f1_layer4_450time_C1,truth1_layer4_450time_C1)
relative_error_layer5_450time_C1 =cal_relative_error(f1_layer5_450time_C1,truth1_layer5_450time_C1)


####case2 relative_error#######
relative_error_layer1 =cal_relative_error(f1_layer1_50time,truth1_layer1_50time)
relative_error_layer2 =cal_relative_error(f1_layer2_50time,truth1_layer2_50time)
relative_error_layer3 =cal_relative_error(f1_layer3_50time,truth1_layer3_50time)
relative_error_layer4 =cal_relative_error(f1_layer4_50time,truth1_layer4_50time)
relative_error_layer5 =cal_relative_error(f1_layer5_50time,truth1_layer5_50time)

relative_error_layer1_250time =cal_relative_error(f1_layer1_250time,truth1_layer1_250time)
relative_error_layer2_250time =cal_relative_error(f1_layer2_250time,truth1_layer2_250time)
relative_error_layer3_250time =cal_relative_error(f1_layer3_250time,truth1_layer3_250time)
relative_error_layer4_250time =cal_relative_error(f1_layer4_250time,truth1_layer4_250time)
relative_error_layer5_250time =cal_relative_error(f1_layer5_250time,truth1_layer5_250time)

relative_error_layer1_450time =cal_relative_error(f1_layer1_450time,truth1_layer1_450time)
relative_error_layer2_450time =cal_relative_error(f1_layer2_450time,truth1_layer2_450time)
relative_error_layer3_450time =cal_relative_error(f1_layer3_450time,truth1_layer3_450time)
relative_error_layer4_450time =cal_relative_error(f1_layer4_450time,truth1_layer4_450time)
relative_error_layer5_450time =cal_relative_error(f1_layer5_450time,truth1_layer5_450time)
def to_percent(temp, position):
    return '%.1f' % (100 * temp) + '%'


plt.figure(figsize=(9, 3), dpi=300)
'''
data = pd.DataFrame({
    "oil_error ": oil_relative_error_case,
    "water_error ": water_relative_error_case,

})


data = pd.DataFrame({
    oil_relative_error_case,
    water_relative_error_case,


})
'''
plt.subplot(1,3,1)
plt.boxplot([relative_error_layer1_C1,relative_error_layer2_C1,relative_error_layer3_C1,relative_error_layer4_C1,relative_error_layer5_C1], 0, '',
                showmeans=True,
                vert=True,
                meanprops={'marker': 'o', 'markersize': 2, 'color': 'r'},
                labels=["Layer1","Layer2","Layer3","Layer4","Layer5"]
                )
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.yticks(fontproperties='Arial', size=8,fontweight='bold')
plt.xticks(fontproperties='Arial', size=8,fontweight='bold')
plt.ylabel("Relative error",fontdict={'family': 'Arial', 'size': 6, 'fontweight': 'bold'})
plt.legend(prop={'family' : 'Arial', 'size' : 7},loc="upper left",labels = ['Case1:T=50 time step'])
plt.subplot(1,3,2)
plt.boxplot([relative_error_layer1_250time_C1,relative_error_layer2_250time_C1,relative_error_layer3_250time_C1,relative_error_layer4_250time_C1,relative_error_layer5_250time_C1], 0, '',
                showmeans=True,
                vert=True,
                boxprops={'color':'red'},
                meanprops={'marker': 'o', 'markersize': 2, 'color': 'b'},
                labels=["Layer1","Layer2","Layer3","Layer4","Layer5"]
                )
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.yticks(fontproperties='Arial', size=8,fontweight='bold')
plt.xticks(fontproperties='Arial', size=8,fontweight='bold')
plt.legend(prop={'family' : 'Arial', 'size' : 7},loc="upper right",labels = ['Case1:T=250 time step'])
plt.subplot(1,3,3)
plt.boxplot([relative_error_layer1_450time_C1,relative_error_layer2_450time_C1,relative_error_layer3_450time_C1,relative_error_layer4_450time_C1,relative_error_layer5_450time_C1], 0, '',
                showmeans=True,
                vert=True,
                boxprops={'color':'b'},
                meanprops={'marker': 'o', 'markersize': 2, 'color': 'b'},
                labels=["Layer1","Layer2","Layer3","Layer4","Layer5"]
                )

# plt.ylim(0,2.0)
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.yticks(fontproperties='Arial', size=8,fontweight='bold')
plt.xticks(fontproperties='Arial', size=8,fontweight='bold')
plt.legend(prop={'family' : 'Arial', 'size' : 7},loc="upper left",labels = ['Case1:T=450 time step'])






plt.savefig("D:/Reaserch work/coarse grid paper/3d to 4d/error box_case1.png", dpi=300)
plt.show()