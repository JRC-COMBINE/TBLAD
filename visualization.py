#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
To get the dimension (# sample points) of each input file in a directory
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from params import *

'''Change ion name here'''
ion = 'Br' 


if ion == 'Br':
    path = path_raw_Br
elif ion == 'Cl':
    path = path_raw_Cl
elif ion == 'NO3':
    path = path_raw_NO3
elif ion == 'ClO4':
    path = path_raw_ClO4
elif ion == 'SCN':
    path = path_raw_SCN
else: 
    print("Invalid group name!")
    exit()
fig, ax = plt.subplots(linewidth=0.5)
for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        if name.endswith(".txt"):
            with open(os.path.join(root,name),'r') as f:
                print(name)
                # initialize empty array that saves the data in the current file "f"
                f_array = []; 
                # parse the ".txt" into python list
                for line in f: 
                    f_array.append(float(line.strip('\n')))
                # finished reading f, write the length of f_array back
                x = np.arange(0,len(f_array))
                x=x*1e-4
                plt.plot(x,f_array,label=name[:-4])
plt.xlabel('Time [s]',fontsize=10)
plt.ylabel('Current [pA]',fontsize=10)
ax.legend()
plt.show()             
fig.savefig(output_path + ion+".png",format="png")   
                
            
        

