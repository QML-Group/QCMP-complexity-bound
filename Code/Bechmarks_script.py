import numpy as np
import networkx as nx
import scipy as sp
import time 
import pandas as pd
from numpy import linalg as LA
from scipy.linalg import expm
import matplotlib.pyplot as plt
import os

start_time=time.time()

#insert beta values
beta_test = [0.001,0.0001,0.00001] 

curdir = os.path.dirname(__file__)

#Import the information from the benchmarks 
file_path = os.path.join(curdir, "INPUT INITIAL PLACEMENT EXCEL")
benchmarkset = pd.read_excel(file_path).to_numpy()
#Functions for the preprocessing and importing of the graphs which we want to use.
#function which converts a Networkx graph to a Laplacian Matrix
def G_to_LM(x):
    LM = sp.sparse.csr_matrix.toarray(nx.laplacian_matrix(nx.from_numpy_array(x)))
    return LM
def G_to_LM_eig(x):
    LM = sp.sparse.csr_matrix.toarray(nx.laplacian_matrix(nx.from_numpy_array(x)))
    L = np.zeros((n,n))
    for i in range(n):
        L[i][i]=np.linalg.eig(LM)[0][i]
    return L
#function which converts a numpy array of a graph to a density matrix
def rho(x):
    rho = sp.linalg.expm(-beta*G_to_LM(x))/np.trace((sp.linalg.expm(-beta*G_to_LM_eig(x))))
    return rho
#function which determines the Von Neuman Entropy of a density matrix
def S(x):
    return -sum(np.linalg.eig(x)[0]*np.log2(np.linalg.eig(x)[0]))
#function which determines the Q_jsd for two given density matrices
def Q_jsd(x,y):
    sum_xy = (x+y)/2
    f = S(sum_xy)-(S(x)+S(y))/2
    return f

#function which determines the IG once the intial CG has been performed, in which y is the CG and x the IG 
def IG_iteration(x,y):
    IG_new = x-y
    IG_new[IG_new<0]=0
    return IG_new
#function which determines the new CG based on the SWAP we would like to perform. In which z is the CG and x&y are the node to be swapped. 
def CG_swap(x,y,z):
    CG_new = z.copy()
    for i in range(n):
        if z[x][i]==1 and i!=y:
            CG_new[y][i]=1
            CG_new[x][i]=0
            CG_new[i][y]=1
            CG_new[i][x]=0
    for i in range(n):
        if z[y][i]==1 and i!=x:
            CG_new[x][i]=1
            CG_new[y][i]=0
            CG_new[i][x]=1
            CG_new[i][y]=0
    return CG_new

#Optimisation function & constraints
#generate the permutation matrices (PM)
def permutation_matrix(CG):
    pms = []
    swap_index = []
    for j in range(n):
        for i in range(n-j):
            if CG[j][i+j]==1:
                swap_allowed=np.identity(n)
                swap_allowed[i+j][i+j]=0
                swap_allowed[j][j]=0
                swap_allowed[i+j][j]=1
                swap_allowed[j][i+j]=1
                index=[None]*2
                index[0]=i+j
                index[1]=j
                swap_index.append(index)
                pms.append(swap_allowed)
    return pms, swap_index

def optimize(CG,IG):
    pms = permutation_matrix(CG)[0]
    swap_index = permutation_matrix(CG)[1]
    #Define the function which we want to optimize 
    def f(x):
        f = 0
        #define the krauss operator in terms of the PM and x
        #Make a second loop which allows for a SUM of Kraus operators. 
        sigma = np.zeros((n,n))
        for i in range(len(pms)):
            sigma += ((np.dot(np.dot(np.sqrt(x[i])*pms[i],rho(CG)),np.sqrt(x[i])*pms[i])))
        f = Q_jsd(sigma,rho(IG))
        return f

    #Format the constraints for Optimisation
    def constraint1(x):
        return np.array([np.sum(x)-1])   
    con1 = {'type': 'eq', 'fun': constraint1}
    bounds = [(0,1)]
    cons=[con1]

    #Optimize for the function f 
    #Generate a random set of theta's with a value between 0-1
    x0 = np.random.uniform(0,1,len(pms))
    #Preform the optimizatoion
    sol = sp.optimize.minimize(f,x0,constraints=cons,bounds=bounds,method='SLSQP',tol=1e-18)

    #print the results in such a manner that they can be analysed
    index = []
    P = []
    res = []
    for i in range(len(pms)):
        res.append(sol.x[i])
        index.append(i)
        P.append(pms[i])
    print("--- %s seconds ---" % (time.time() - start_time))
    SWAPS.append(swap_index[np.argmax(res)])
    return SWAPS



benchmarks = 46

#interaction graphs (next to the edge description there are names of benchmarks that have this kind of IG)
G_I = [None]*benchmarks
for i in range(benchmarks):
    G_I[i] = nx.Graph()

#brute-force test
# G_I[0].add_edges_from([(0,1),(1,2),(2,0),(1,3)]) #4gt11_8 (5q bench)
# G_I[1].add_edges_from([(0,1),(0,2), (1,2),(1,3),(2,3),(3,0)]) # hwb4_52
# G_I[2].add_edges_from([(0,1),(1,2),(2,0)]) #3_17_13
# G_I[3].add_edges_from([(0,1),(1,2),(2,3),(3,0),(1,3)]) #rd32-v0_67,decod24-v3_46


#3Q
G_I[0].add_edges_from([(0,1),(1,2),(2,0)]) #fredkin_n3, grover_n3
G_I[1].add_edges_from([(0,1),(1,2)]) #basis_change_n3, teleportation_n3

#4Q
G_I[2].add_edges_from([(0,1),(1,2),(2,3),(3,0)]) #adder_n4
G_I[3].add_edges_from([(0,1),(1,2),(2,3)]) #variational_n4, bell_n4
G_I[4].add_edges_from([(0,1),(1,2),(2,0),(1,3)]) #cuccaro adder 1b
G_I[5].add_edges_from([(0,1),(0,2), (1,2),(1,3),(2,3),(3,0)]) #q=4_s=19996_2qbf=02_1, q=4_s=2996_2qbf=08_1
G_I[6].add_edges_from([(0,2),(0,3),(2,3),(1,2),(1,3)]) #vbe_adder_1b

#5Q
G_I[7].add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4), (3,4)]) #4gt10-v1_81, q=5_s=2995_2qbf=09_1, all to all
G_I[8].add_edges_from([(0,1),(0,4),(1,4),(3,4),(2,4),(2,3)]) #4gt13_92
G_I[9].add_edges_from([(0,1),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]) #4gt5_75
G_I[10].add_edges_from([(0,1),(0,3),(1,2),(1,3),(1,4),(2,4), (3,4)]) #alu-v1_28
G_I[11].add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4)]) #alu-v2_31
G_I[12].add_edges_from([(0,1),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4), (3,4)]) #decod24-v1_41
G_I[13].add_edges_from([(0,1),(0,2),(1,2),(2,3),(2,4)]) #error_correctiond3_n5
G_I[14].add_edges_from([(0,2),(1,2),(2,3),(2,4)]) #qec_en_n5, qec_sm_n5
G_I[15].add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,3),(1,4),(2,3),(3,4)]) #quantum_volume_n5
G_I[16].add_edges_from([(0,1),(0,3),(1,3),(2,3),(2,4)]) #simon_n6

# #6Q
G_I[17].add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5)]) #alu-v2_30,all-to-all,q=6_s=2994_2qbf=08_1
G_I[18].add_edges_from([(0,1),(0,2),(0,4),(0,5),(1,2),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5)]) #4gt12-v0_87
G_I[19].add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(1,3),(1,4),(2,3),(2,4),(2,5),(3,4),(3,5)]) #4gt4-v0_72
G_I[20].add_edges_from([(0,5),(5,3),(5,4),(0,1),(0,2),(4,3),(3,1),(2,1),(3,4),(4,2)]) #qaoa 6
G_I[21].add_edges_from([(0,1),(0,4),(0,5),(1,2),(1,4),(1,5),(2,3),(2,5),(3,4),(3,5)]) #ex3_229
G_I[22].add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5)]) #graycode6_47, line
G_I[23].add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(2,3),(2,4),(2,5),(3,4),(3,5)]) #mod5adder_127
G_I[24].add_edges_from([(0,1),(0,2),(0,4),(1,2),(1,3),(2,5),(3,4),(3,5)]) #q=6_s=54_2qbf=022_1
G_I[25].add_edges_from([(0,1),(0,5),(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5)]) #sf_274
G_I[26].add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5)]) #star,xor5_254 

#7Q
G_I[27].add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(1,2),(1,3),(1,4),(1,5),(1,6),(2,3),(2,4),(2,5),(2,6),(3,4),(3,5),(3,6), (4,5),(4,6),(5,6)]) #alltoall, q=7_s=29993_2qbf=08_1, q=7_s=2993_2qbf=08_1
G_I[28].add_edges_from([(0,4),(0,5),(0,6),(1,4),(2,4),(2,5),(2,6),(3,4),(4,5),(4,6),(5,6)]) #4mod5-bdd_287
G_I[29].add_edges_from([(0,1),(0,2),(0,5),(0,6),(1,4),(1,5),(1,6),(2,3),(2,4),(2,6),(3,4),(3,5),(3,6), (4,5),(4,6),(5,6)]) #majority_239
G_I[30].add_edges_from([(0,2),(0,3),(0,6),(1,3),(1,4),(1,6),(2,3),(2,4),(2,5),(2,6),(3,4),(3,5),(3,6), (4,5),(4,6),(5,6)]) #ham7_104
G_I[31].add_edges_from([(0,2),(0,3),(0,4),(0,5),(0,6),(1,2),(1,3),(1,4),(1,5),(1,6),(2,3),(2,4),(2,5),(2,6),(3,4),(3,5),(3,6), (4,5),(4,6),(5,6)]) #C17_204
G_I[32].add_edges_from([(0,3),(0,5),(0,6),(1,3),(1,5),(2,3),(2,6),(3,5),(3,6),(4,5),(4,6),(5,6)]) #alu-bdd_288


#8Q
G_I[33].add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(2,3),(2,4),(2,5),(2,6),(2,7),(3,4),(3,5),(3,6),(3,7), (4,5),(4,6),(4,7),(5,6),(5,7),(6,7)]) #alltoall, hwb7_59,q=8_s=2992_2qbf=01_1
G_I[34].add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(0,7),]) #ring, dnn_n8
G_I[35].add_edges_from([(0,3),(0,4),(0,7),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(2,3),(2,4),(2,5),(2,6),(2,7),(3,4),(3,5),(3,7), (4,5),(4,7),(5,6),(5,7),(6,7)]) #f2_232
G_I[36].add_edges_from([(0,1),(0,7),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(2,3),(2,4),(2,5),(2,6),(3,4),(3,5),(3,6), (4,5),(4,6),(5,6),(6,7)]) #vqe_uccsd_n8

#9Q
G_I[37].add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(3,4),(3,5),(3,6),(3,7),(3,8), (4,5),(4,6),(4,7),(4,8),(5,6),(5,8),(5,7),(6,7),(6,8),(7,8)]) #alltoall q=9_s=19991_2qbf=08_1,q=9_s=2991_2qbf=01_1
G_I[38].add_edges_from([(0,7),(0,8),(3,7),(1,3),(4,7),(4,5)]) #q=9_s=51_2qbf=012_1

# #10Q
G_I[39].add_edges_from([(0,1),(0,5),(1,5),(1,6),(1,2),(2,6),(2,3),(2,7),(3,7),(3,4),(3,8),(4,8),(4,9)]) #adder_n10
G_I[40].add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(4,5),(4,6),(4,7),(4,8),(4,9),(5,6),(5,8),(5,7),(5,9),(6,7),(6,8),(6,9),(7,8),(7,9),(8,9)]) #alltoall q=10_s=990_2qbf=091_1
G_I[41].add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),(3,5),(3,6),(3,7),(3,8),(3,9),(4,5),(4,6),(4,7),(4,8),(4,9),(5,6),(5,8),(5,7),(5,9),(6,7),(6,8),(6,9),(7,8),(7,9),(8,9)]) #sqn_258
G_I[42].add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),(3,5),(3,6),(3,7),(3,8),(3,9),(4,5),(4,6),(4,7),(4,8),(4,9),(5,6),(5,8),(5,7),(5,9),(6,7),(6,8),(6,9),(7,8),(7,9),(8,9)]) #sym9_148

# #11Q
G_I[43].add_edges_from([(0,2),(0,3),(0,4),(0,5),(0,6),(0,8),(0,9),(1,2),(1,3),(1,4),(1,5),(1,6),(1,8),(1,9),(2,4),(2,5),(2,8),(2,9),(3,5),(3,8),(3,9),(4,6),(5,6),(5,8),(5,7),(5,10),(6,7),(6,8),(6,10),(7,8),(7,9),(7,10),(8,9),(9,10)])  #shor_15


#16Q
G_I[44].add_edges_from([(7,13),(11,13),(9,11),(6,9),(0,6),(0,3),(3,15),(7,15),(0,12),(5,6),(5,12),(0,12),(4,12),(1,4),(1,14),(2,14),(2,8),(8,10),(5,10)]) #16QBT_100CYC_QSE_1


#20Q
G_I[45].add_edges_from([(0,3),(0,7),(0,14),(0,19),(1,14),(1,16),(2,12),(2,19),(3,7),(3,10),(3,13),(3,17),(4,9),(4,11),(4,13),(5,6),(5,15),(5,17),(6,8),(6,10),(6,19),(7,14),(8,10),(8,12),(8,19),(9,11),(9,13),(9,15),(10,13),(10,17),(10,19),(11,13),(15,17),(15,18),(16,19)]) #20QBT_45CYC_0D1_2D2_0

IG = [None]*benchmarks
for i in range(benchmarks):
    IG[i] = nx.to_numpy_array(G_I[i], nodelist= sorted(G_I[i].nodes))


results = [] 
benchmark_name = []
device_name = []
SWAPS_qubits = []
Num_SWAPS = []


for z in range(len(beta_test)):
    beta = beta_test[z]
    IG_bench = []
    CG_bench = []
    p=0

    for q in range(len(benchmarkset)):
        #-1
        i=0
        IG_bench = IG[p]
#     CG_bench_G = nx.Graph()
#        if benchmarkset[q][8]==0:
#            CG_bench = nx.to_numpy_array(G_I[p])
#        else:
        CG_bench_G = nx.Graph()
        ## !! need to convert the missing edges based on the intial placement 
        edges_to_add = list(eval(benchmarkset[q][5])) #data2
        intial_placement = list(eval(benchmarkset[q][4])) #data
        
        for V in range(len(edges_to_add[1])):
          edges_CG = []
          for j in range(len(intial_placement)):
            for k in range(2):
              if intial_placement[j][1] == edges_to_add[1][V][k]:
                edges_CG.append(intial_placement[j][0])
          CG_bench_G.add_edge(edges_CG[0],edges_CG[1])
        CG_bench = nx.to_numpy_array(CG_bench_G, nodelist= sorted(CG_bench_G.nodes))
        SWAPS = []
        n=len(CG_bench)
        while np.sum(IG_iteration(IG_bench,CG_bench))!=0: 
            s = optimize(CG_bench,IG_bench)
            IG_bench = IG_iteration(IG_bench,CG_bench)
            CG_bench = CG_swap(s[i][0],s[i][1],CG_bench)
            if i>7 and SWAPS[i]==SWAPS[i-1] and SWAPS[i]==SWAPS[i-2] and SWAPS[i]==SWAPS[i-3]:
                SWAPS = [0,0]

                break
            else: 
                i+=1
        if not SWAPS: 
            results.append([benchmarkset[q][0], benchmarkset[q][2], SWAPS,0]) 
        elif SWAPS==[0,0]:
            results.append([benchmarkset[q][0], benchmarkset[q][2], SWAPS,19970801])
        else: 
            results.append([benchmarkset[q][0], benchmarkset[q][2], SWAPS,len(SWAPS)])
        df = pd.DataFrame(results, columns=["Benchmark name","Device name","Qubit to SWAP", "Number of SWAPs"])
        writer = pd.ExcelWriter(curdir + "/output.xlsx", engine='openpyxl', mode='a', if_sheet_exists='overlay')
        file = pd.read_excel(curdir + "/output.xlsx")
        df.to_excel(writer,index=False)
        writer.close()
        if q < len(benchmarkset) - 1:
            if benchmarkset[q][0]!= benchmarkset[q+1][0]:
                p+=1

