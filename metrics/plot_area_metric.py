import numpy as np
import matplotlib.pyplot as plt
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))
f_data = np.load(dir_path+'/frontier/08_02_2023_01_45_32.npy')
n_data = np.load(dir_path+'/naive/07_02_2023_23_37_02.npy')
h_data = np.load(dir_path+'/high/07_02_2023_23_19_58.npy')

plt.plot(f_data[:,0], f_data[:,1],'-g',label='frontier exploration')
plt.plot(n_data[:,0], n_data[:,1],'-b',label='naive with PGART')
plt.plot(h_data[:,0], h_data[:,1],'-r',label='HIGH with PGART')
plt.title('Explored Area')
plt.xlabel('Time [s]')
plt.ylabel('Explored Area [m^2]')
plt.legend()
plt.show()
