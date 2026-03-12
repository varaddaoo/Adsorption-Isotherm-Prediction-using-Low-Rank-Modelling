import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams["figure.figsize"] = [11, 7]
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 14.0

fontdict_t = {'fontsize': 20, 'weight': 'bold', 'ha': 'center'}
fontdict_x = {'fontsize': 26, 'weight': 'bold', 'ha': 'center'}
fontdict_y = {'fontsize': 26, 'weight': 'bold', 'va': 'baseline', 'ha': 'center'}
fontdict_l = {'size': 80, 'weight': 'bold'}

def Isotherm(x, K):
    theta = (K * x) / (1 + K * x)
    return theta

# Relative pressure range
x = np.linspace(0, 1, 100)   # P/P0

# Langmuir constant
K = 5

# Fractional loading
theta = Isotherm(x, K)

#XPTS = np.array([0.1, 0.2, 0.4, 0.6, 0.8])
XPTS = np.array([0.1, 0.4, 0.7])
YPTS = Isotherm(XPTS, K)

for i in range(len(XPTS)):
    
    plt.figure()
    
    plt.plot(x, theta, linestyle='-', color='#4D4D4D', linewidth=24.0, alpha=0.6, label='Argon')
    
    xpts = XPTS[i]
    ypts = YPTS[i]
    
    plt.plot(xpts, ypts, marker='o', markersize=70, color='red')
    
    plt.legend(loc='best', prop=fontdict_l, frameon=False)
    plt.xticks([])
    plt.yticks([])
    
    #plt.grid(True)

    plt.tight_layout()

    plt.savefig(f"{i}_Upper.pdf", format='pdf')

icolor1=['#7030A0', '#C5E0B4', '#0000FF']
icolor2=['#F8CBAD', '#92D050', '#00B0F0']
icolor3=['#FFA500', '#934D75', '#D0CCE5']
icolor4=['#F48E92', '#FF0000', '#006400']

icolor = [icolor1, icolor2, icolor3, icolor4]

for i, iicolor in enumerate(icolor):
    plt.figure()
    
    plt.plot(x, theta, linestyle='-', color='#4D4D4D', linewidth=24.0, alpha=0.6, label='Argon')
    
    plt.plot(XPTS[0], YPTS[0], marker='o', markersize=70, color=iicolor[0])
    plt.plot(XPTS[1], YPTS[1], marker='o', markersize=70, color=iicolor[1])
    plt.plot(XPTS[2], YPTS[2], marker='o', markersize=70, color=iicolor[2])
    
    plt.legend(loc='best', prop=fontdict_l, frameon=False)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    plt.savefig(f"{i}_Lower.pdf", format='pdf')

plt.show()