import numpy as np
import networkx as nx
from numpy import linalg as LA
from scipy.linalg import expm
import matplotlib.pyplot as plt

# initialize all of the 4-vertex graphs using NetworkX & extract graph laplacians in np.array form 
g0 = nx.Graph()
g0.add_nodes_from([0,1,2,3])
g0L = nx.to_numpy_array(g0)

g1 = nx.Graph()
g1.add_nodes_from([0,1,2,3])
g1.add_edges_from([(0,1)])
g1L = nx.to_numpy_array(g1)

g2 = nx.Graph()
g2.add_nodes_from([0,1,2,3])
g2.add_edges_from([(0,1),(0,3)])
g2L = nx.to_numpy_array(g2)

g3 = nx.Graph()
g3.add_nodes_from([0,1,2,3])
g3.add_edges_from([(1,2),(0,3)])
g3L = nx.to_numpy_array(g3)

g4 = nx.Graph()
g4.add_nodes_from([0,1,2,3])
g4.add_edges_from([(1,2),(0,3),(0,2)])
g4L = nx.to_numpy_array(g4)

g5 = nx.Graph()
g5.add_nodes_from([0,1,2,3])
g5.add_edges_from([(0,1),(0,3),(0,2)])
g5L = nx.to_numpy_array(g5)

g6 = nx.Graph()
g6.add_nodes_from([0,1,2,3])
g6.add_edges_from([(0,1),(0,3),(2,3)])
g6L = nx.to_numpy_array(g6)

g7 = nx.Graph()
g7.add_nodes_from([0,1,2,3])
g7.add_edges_from([(0,1),(0,3),(2,3),(1,2)])
g7L = nx.to_numpy_array(g7)

g8 = nx.Graph()
g8.add_nodes_from([0,1,2,3])
g8.add_edges_from([(0,1),(0,2),(2,3),(1,2)])
g8L = nx.to_numpy_array(g8)

g9 = nx.Graph()
g9.add_nodes_from([0,1,2,3])
g9.add_edges_from([(0,1),(0,3),(2,3),(1,2),(1,3)])
g9L = nx.to_numpy_array(g9)

g10 = nx.Graph()
g10.add_nodes_from([0,1,2,3])
g10.add_edges_from([(0,1),(0,3),(2,3),(1,2),(1,3),(0,2)])
g10L = nx.to_numpy_array(g10)

# calculate density matrices for several beta values
beta = np.linspace(0,10,1000)
rho0 = []
for i in range(len(beta)):
    partition_func = np.sum(LA.eigvals(expm(-beta[i]*g0L)))
    rho0.append(expm(-beta[i]*g0L)/partition_func)
    
rho1 = []
for i in range(len(beta)):
    partition_func = np.sum(LA.eigvals(expm(-beta[i]*g1L)))
    rho1.append(expm(-beta[i]*g1L)/partition_func)
    
rho2 = []
for i in range(len(beta)):
    partition_func = np.sum(LA.eigvals(expm(-beta[i]*g2L)))
    rho2.append(expm(-beta[i]*g2L)/partition_func)
    
rho3 = []
for i in range(len(beta)):
    partition_func = np.sum(LA.eigvals(expm(-beta[i]*g3L)))
    rho3.append(expm(-beta[i]*g3L)/partition_func)
    
rho4 = []
for i in range(len(beta)):
    partition_func = np.sum(LA.eigvals(expm(-beta[i]*g4L)))
    rho4.append(expm(-beta[i]*g4L)/partition_func)
    
rho5 = []
for i in range(len(beta)):
    partition_func = np.sum(LA.eigvals(expm(-beta[i]*g5L)))
    rho5.append(expm(-beta[i]*g5L)/partition_func)
    
rho6 = []
for i in range(len(beta)):
    partition_func = np.sum(LA.eigvals(expm(-beta[i]*g6L)))
    rho6.append(expm(-beta[i]*g6L)/partition_func)
    
rho7 = []
for i in range(len(beta)):
    partition_func = np.sum(LA.eigvals(expm(-beta[i]*g7L)))
    rho7.append(expm(-beta[i]*g7L)/partition_func)
    
rho8 = []
for i in range(len(beta)):
    partition_func = np.sum(LA.eigvals(expm(-beta[i]*g8L)))
    rho8.append(expm(-beta[i]*g8L)/partition_func)
    
rho9 = []
for i in range(len(beta)):
    partition_func = np.sum(LA.eigvals(expm(-beta[i]*g9L)))
    rho9.append(expm(-beta[i]*g9L)/partition_func)
    
rho10 = []
for i in range(len(beta)):
    partition_func = np.sum(LA.eigvals(expm(-beta[i]*g10L)))
    rho10.append(expm(-beta[i]*g10L)/partition_func)

# calculate VNE & QJSD for all density matrices and all beta values 
vne0 = []
for i in range(len(beta)):
    vne = 0
    rho0[i] = LA.eigvals(rho0[i])
    for j in range(4):
        vne += -rho0[i][j]*np.log(rho0[i][j])
    vne0.append(vne)

vne1 = []
for i in range(len(beta)):
    vne = 0
    rho1[i] = LA.eigvals(rho1[i])
    for j in range(4):
        vne += -rho1[i][j]*np.log(rho1[i][j])
    vne1.append(vne)

vne2 = []
for i in range(len(beta)):
    vne = 0
    rho2[i] = LA.eigvals(rho2[i])
    for j in range(4):
        vne += -rho2[i][j]*np.log(rho2[i][j])
    vne2.append(vne)

vne3 = []
for i in range(len(beta)):
    vne = 0
    rho3[i] = LA.eigvals(rho3[i])
    for j in range(4):
        vne += -rho3[i][j]*np.log(rho3[i][j])
    vne3.append(vne)

vne4 = []
for i in range(len(beta)):
    vne = 0
    rho4[i] = LA.eigvals(rho4[i])
    for j in range(4):
        vne += -rho4[i][j]*np.log(rho4[i][j])
    vne4.append(vne)

vne5 = []
for i in range(len(beta)):
    vne = 0
    rho5[i] = LA.eigvals(rho5[i])
    for j in range(4):
        vne += -rho5[i][j]*np.log(rho5[i][j])
    vne5.append(vne)

vne6 = []
for i in range(len(beta)):
    vne = 0
    rho6[i] = LA.eigvals(rho6[i])
    for j in range(4):
        vne += -rho6[i][j]*np.log(rho6[i][j])
    vne6.append(vne)

vne7 = []
for i in range(len(beta)):
    vne = 0
    rho7[i] = LA.eigvals(rho7[i])
    for j in range(4):
        vne += -rho7[i][j]*np.log(rho7[i][j])
    vne7.append(vne)

vne8 = []
for i in range(len(beta)):
    vne = 0
    rho8[i] = LA.eigvals(rho8[i])
    for j in range(4):
        vne += -rho8[i][j]*np.log(rho8[i][j])
    vne8.append(vne)

vne9 = []
for i in range(len(beta)):
    vne = 0
    rho9[i] = LA.eigvals(rho9[i])
    for j in range(4):
        vne += -rho9[i][j]*np.log(rho9[i][j])
    vne9.append(vne)

vne10 = []
for i in range(len(beta)):
    vne = 0
    rho10[i] = LA.eigvals(rho10[i])
    for j in range(4):
        vne += -rho10[i][j]*np.log(rho10[i][j])
    vne10.append(vne)

# plot everything for VNE of all the density matrices 
xaxis = np.linspace(0,10,1000)
plt.plot(xaxis,vne0,label=r'$\mathcal{S}(\rho_{0})$')
plt.plot(xaxis,vne1,label=r'$\mathcal{S}(\rho_{1})$')
plt.plot(xaxis,vne2,label=r'$\mathcal{S}(\rho_{2})$')
plt.plot(xaxis,vne3,label=r'$\mathcal{S}(\rho_{3})$')
plt.plot(xaxis,vne4,label=r'$\mathcal{S}(\rho_{4})$')
plt.plot(xaxis,vne5,label=r'$\mathcal{S}(\rho_{5})$')
plt.plot(xaxis,vne6,label=r'$\mathcal{S}(\rho_{6})$')
plt.plot(xaxis,vne7,label=r'$\mathcal{S}(\rho_{7})$')
plt.plot(xaxis,vne8,label=r'$\mathcal{S}(\rho_{8})$')
plt.plot(xaxis,vne9,label=r'$\mathcal{S}(\rho_{9})$')
plt.plot(xaxis,vne10,label=r'$\mathcal{S}(\rho_{10})$')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
plt.xlabel(r'$\beta \in [0,10]$')
plt.ylabel(r'$\mathcal{S}(\rho_{i}), i \in [0,10]$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# plot everything for QJSD of all the density matrices 
# xaxis = np.linspace(0,10,1000)
# plt.plot(xaxis,qjsd0,label=r'$\lambda_{i}$')
# plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.show()