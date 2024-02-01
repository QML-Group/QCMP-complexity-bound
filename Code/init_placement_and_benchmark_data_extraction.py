import itertools
import numpy as np
from graphviz import Graph
import networkx as nx
from networkx.algorithms import isomorphism
from networkx.algorithms import matching
from operator import itemgetter
from networkx.algorithms.isomorphism import GraphMatcher
import pandas as pd
import os

curdir = os.path.dirname(__file__)
data = []
data_qasm = pd.read_excel(r'BENCHMARKS DATA FILE')

#CHOOSE A CIRCUIT INTERACTION GRAPH FOR INTIAL PLACEMENT
bench_names = "rd32-v0_67,decod24-v3_46"
no_of_qubits = 4

#read 2q-gate count from the existing file:
twoqgates = 0
for i,b in data_qasm.iterrows():
    if bench_names.__contains__(b['benchmark']):
        twoqgates = round(b['gates before']*b['2-q gate perc'])

G1 = nx.Graph()
#brute-force example
G1.add_edges_from([(0,1),(1,2),(2,3),(3,0),(1,3)]) #rd32-v0_67,decod24-v3_46

#46 IGs, 56 circuits used in the paper
#3Q
#G1.add_edges_from([(0,1),(1,2),(2,0)]) #fredkin_n3, grover_n3,3_17_13
#G1.add_edges_from([(0,1),(1,2)]) #basis_change_n3, teleportation_n3

# #4Q
#G1.add_edges_from([(0,1),(1,2),(2,3),(3,0)]) #adder_n4
#G1.add_edges_from([(0,1),(1,2),(2,3)]) #variational_n4, bell_n4
#G1.add_edges_from([(0,1),(1,2),(2,0),(1,3)]) #cuccaro adder 1b, 4gt11_8 (5q bench)
#G1.add_edges_from([(0,1),(0,2), (1,2),(1,3),(2,3),(3,0)]) #all to all, q=4_s=19996_2qbf=02_1, q=4_s=2996_2qbf=08_1, hwb4_52
#G1.add_edges_from([(0,2),(0,3),(2,3),(1,2),(1,3)]) #vbe_adder_1b

# #5Q
#G1.add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4), (3,4)]) #4gt10-v1_81, q=5_s=2995_2qbf=09_1, all to all
#G1.add_edges_from([(0,1),(0,4),(1,4),(3,4),(2,4),(2,3)]) #4gt13_92
#G1.add_edges_from([(0,1),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]) #4gt5_75
#G1.add_edges_from([(0,1),(0,3),(1,2),(1,3),(1,4),(2,4), (3,4)]) #alu-v1_28
#G1.add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4)]) #alu-v2_31
#G1.add_edges_from([(0,1),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4), (3,4)]) #decod24-v1_41
#G1.add_edges_from([(0,1),(0,2),(1,2),(2,3),(2,4)]) #error_correctiond3_n5
#G1.add_edges_from([(0,2),(1,2),(2,3),(2,4)]) #qec_en_n5, qec_sm_n5
#G1.add_edges_from([(2,3),(0,2),(1,3),(4,0),(3,1)])
#G1.add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,3),(1,4),(2,3),(3,4)]) #quantum_volume_n5
#G1.add_edges_from([(0,1),(0,3),(1,3),(2,3),(2,4)]) #simon_n6

#6Q
#G1.add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5)]) #alu-v2_30,q=6_s=2994_2qbf=08_1,all-to-all
#G1.add_edges_from([(0,1),(0,2),(0,4),(0,5),(1,2),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5)]) #4gt12-v0_87
#G1.add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(1,3),(1,4),(2,3),(2,4),(2,5),(3,4),(3,5)]) #4gt4-v0_72
#G1.add_edges_from([(0,5),(5,3),(5,4),(0,1),(0,2),(4,3),(3,1),(2,1),(3,4),(4,2)]) #qaoa 6
#G1.add_edges_from([(0,1),(0,4),(0,5),(1,2),(1,4),(1,5),(2,3),(2,5),(3,4),(3,5)]) #ex3_229
#G1.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5)]) #graycode6_47, line
#G1.add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(2,3),(2,4),(2,5),(3,4),(3,5)]) #mod5adder_127
#G1.add_edges_from([(0,1),(0,2),(0,4),(1,2),(1,3),(2,5),(3,4),(3,5)]) #q=6_s=54_2qbf=022_1
#G1.add_edges_from([(0,1),(0,5),(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5)]) #sf_274
#G1.add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5)]) #star, xor5_254

#7Q
#G1.add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(1,2),(1,3),(1,4),(1,5),(1,6),(2,3),(2,4),(2,5),(2,6),(3,4),(3,5),(3,6), (4,5),(4,6),(5,6)]) #alltoall, q=7_s=29993_2qbf=08_1, q=7_s=2993_2qbf=08_1
#G1.add_edges_from([(0,4),(0,5),(0,6),(1,4),(2,4),(2,5),(2,6),(3,4),(4,5),(4,6),(5,6)]) #4mod5-bdd_287
#G1.add_edges_from([(0,1),(0,2),(0,5),(0,6),(1,4),(1,5),(1,6),(2,3),(2,4),(2,6),(3,4),(3,5),(3,6), (4,5),(4,6),(5,6)]) #majority_239
#G1.add_edges_from([(0,2),(0,3),(0,6),(1,3),(1,4),(1,6),(2,3),(2,4),(2,5),(2,6),(3,4),(3,5),(3,6), (4,5),(4,6),(5,6)]) #ham7_104
#G1.add_edges_from([(0,2),(0,3),(0,4),(0,5),(0,6),(1,2),(1,3),(1,4),(1,5),(1,6),(2,3),(2,4),(2,5),(2,6),(3,4),(3,5),(3,6),(4,5),(4,6),(5,6)]) #C17_204
#G1.add_edges_from([(0,3),(0,5),(0,6),(1,3),(1,5),(2,3),(2,6),(3,5),(3,6),(4,5),(4,6),(5,6)]) #alu-bdd_288


#8Q
#G1.add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(2,3),(2,4),(2,5),(2,6),(2,7),(3,4),(3,5),(3,6),(3,7), (4,5),(4,6),(4,7),(5,6),(5,7),(6,7)]) #alltoall, hwb7_59,q=8_s=2992_2qbf=01_1
#G1.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(0,7),]) #ring, dnn_n8
#G1.add_edges_from([(0,3),(0,4),(0,7),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(2,3),(2,4),(2,5),(2,6),(2,7),(3,4),(3,5),(3,7), (4,5),(4,7),(5,6),(5,7),(6,7)]) #f2_232
#G1.add_edges_from([(0,1),(0,7),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(2,3),(2,4),(2,5),(2,6),(3,4),(3,5),(3,6), (4,5),(4,6),(5,6),(6,7)]) #vqe_uccsd_n8

#9Q
#G1.add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(3,4),(3,5),(3,6),(3,7),(3,8), (4,5),(4,6),(4,7),(4,8),(5,6),(5,8),(5,7),(6,7),(6,8),(7,8)]) #alltoall q=9_s=19991_2qbf=08_1,q=9_s=2991_2qbf=01_1
#G1.add_edges_from([(0,7),(0,8),(3,7),(1,3),(4,7),(4,5)]) #q=9_s=51_2qbf=012_1

#10Q
#G1.add_edges_from([(0,1),(0,5),(1,5),(1,6),(1,2),(2,6),(2,3),(2,7),(3,7),(3,4),(3,8),(4,8),(4,9)]) #adder_n10
#G1.add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(4,5),(4,6),(4,7),(4,8),(4,9),(5,6),(5,8),(5,7),(5,9),(6,7),(6,8),(6,9),(7,8),(7,9),(8,9)]) #alltoall q=10_s=990_2qbf=091_1
#G1.add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),(3,5),(3,6),(3,7),(3,8),(3,9),(4,5),(4,6),(4,7),(4,8),(4,9),(5,6),(5,8),(5,7),(5,9),(6,7),(6,8),(6,9),(7,8),(7,9),(8,9)]) #sqn_258
#G1.add_edges_from([(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),(3,5),(3,6),(3,7),(3,8),(3,9),(4,5),(4,6),(4,7),(4,8),(4,9),(5,6),(5,8),(5,7),(5,9),(6,7),(6,8),(6,9),(7,8),(7,9),(8,9)]) #sym9_148

#11Q
#G1.add_edges_from([(0,2),(0,3),(0,4),(0,5),(0,6),(0,8),(0,9),(1,2),(1,3),(1,4),(1,5),(1,6),(1,8),(1,9),(2,4),(2,5),(2,8),(2,9),(3,5),(3,8),(3,9),(4,6),(5,6),(5,8),(5,7),(5,10),(6,7),(6,8),(6,10),(7,8),(7,9),(7,10),(8,9),(9,10)])  #shor_15

#16Q
#G1.add_edges_from([(7,13),(11,13),(9,11),(6,9),(0,6),(0,3),(3,15),(7,15),(0,12),(5,6),(5,12),(0,12),(4,12),(1,4),(1,14),(2,14),(2,8),(8,10),(5,10)]) #16QBT_100CYC_QSE_1

#20Q
#G1.add_edges_from([(0,3),(0,7),(0,14),(0,19),(1,14),(1,16),(2,12),(2,19),(3,7),(3,10),(3,13),(3,17),(4,9),(4,11),(4,13),(5,6),(5,15),(5,17),(6,8),(6,10),(6,19),(7,14),(8,10),(8,12),(8,19),(9,11),(9,13),(9,15),(10,13),(10,17),(10,19),(11,13),(15,17),(15,18),(16,19)]) #20QBT_45CYC_0D1_2D2_0

# #coupling graphs/quantum devices
#16 CGs
G2 = []
devices_qubits = [("2x2",4),("2x3",6),("IBM Athens" ,4),("Starmon-5",5),("IBM Yorktown",5),("IBM Ourense",5),("Surface-7",7),
    ("IBM Casablanca",7),("Rigetti Agave",8),("IBM Melbourne",15),("Rigetti Aspen-1",16),("Surface-17",17),("IBM Singapore",20),("IBM Johannesburg",20),("IBM Tokyo",20),("IBM Paris",27),('IBM Rochester',53),("Google Bristlecone",72)]


#IBM Athens - 5 qubits
G2_0 = nx.Graph()
G2_0.add_edges_from([(0,1),(1,2),(2,3),(3,4)])
G2.append(G2_0)

# #Starmon-5 - 5 qubits
G2_1 = nx.Graph()
G2_1.add_edges_from([(2,0),(1,2),(2,3),(4,2)])
G2.append(G2_1)

# #IBM Yorktown - 5 qubits
G2_2 = nx.Graph()
G2_2.add_edges_from([(1,0),(2,0),(1,2),(2,3),(4,2),(3,4)])
G2.append(G2_2)

# #IBM Ourense - 5 qubits
G2_3 = nx.Graph()
G2_3.add_edges_from([(0,1),(1,2),(1,3),(3,4)])
G2.append(G2_3)

# #Surface-7 - 7 qubits
G2_4 = nx.Graph()
G2_4.add_edges_from([(2,0),(2,5),(5,3), (0,3),(3,1),(3,6),(6,4),(4,1)])
G2.append(G2_4)

#IBM Casablanca - 7 qubits
G2_5 = nx.Graph()
G2_5.add_edges_from([(0,1),(1,2),(1,3),(3,5),(4,5),(5,6)])
G2.append(G2_5)

# #Rigetti Agave - 8 qubits
G2_6 = nx.Graph()
G2_6.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(0,7)])
G2.append(G2_6)

# #IBM Melbourne - 15 qubits
G2_7 = nx.Graph()
G2_7.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,8),(8,7),(8,9),(9,10),(10,11),(11,12),(12,13),(1,13),(2,12),(3,11),(4,10),(5,9),(6,8),(0,14),(13,14)])
G2.append(G2_7)

# #Rigetti Aspen-1 - 16 qubits
G2_8 = nx.Graph()
G2_8.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(0,7),(2,8),(3,9),(8,9),(8,10),(11,12),(12,13),(13,14),(14,15),(15,9)])
G2.append(G2_8)

# #Surface-17 - 17 qubits
G2_9 = nx.Graph()
G2_9.add_edges_from([(2,0),(4,1),(0,3),(1,5),(2,6),(6,3),(4,7),(7,5),(5,8),(8,6),(6,9),(7,10),(10,8),(8,11),(11,9),(9,12),(13,10),(10,14),(14,11),(11,15),(15,12),(13,16),(16,14)])
G2.append(G2_9)

# #IBM Singapore - 20 qubits
G2_10 = nx.Graph()
G2_10.add_edges_from([(0,1),(1,2),(2,3),(3,4),(1,6),(3,8),(5,6),(6,7),(8,7),(8,9),(5,10),(7,12),(9,14),(10,11),(11,12),(12,13),(13,14),(11,16),(13,18),(15,16),(16,17),(17,18),(18,19)])
G2.append(G2_10)

# #IBM Johannesburg - 20 qubits
G2_11 = nx.Graph()
G2_11.add_edges_from([(0,1),(1,2),(2,3),(3,4),(0,5),(4,9),(5,6),(6,7),(8,7),(8,9),(5,10),(7,12),(9,14),(10,11),(11,12),(12,13),(13,14),(10,15),(14,19),(15,16),(16,17),(17,18),(18,19)])
G2.append(G2_11)

# #IBM Tokyo - 20 qubits
G2_12 = nx.Graph()
G2_12.add_edges_from([(0,1),(1,2),(2,3),(3,4),(0,5),(1,6),(2,7),(3,8),(4,9),(1,7),(2,6),(3,9),(4,8),(5,6),(6,7),(8,7),(8,9),(5,10),(6,11),(7,12),(8,13),(9,14),(5,11),(6,10),(7,13),(8,12),(10,11),(11,12),(12,13),(13,14),(10,15),(11,16),(12,17),(13,18),(14,19),(11,17),(12,16),(13,19),(14,18),(15,16),(16,17),(17,18),(18,19)])
G2.append(G2_12)

# #IBM Paris - 27 qubits
G2_13 = nx.Graph()
G2_13.add_edges_from([(0,4),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(8,7),(8,9),(9,10),(8,11),(2,12),(6,13),(10,14),(15,12),(19,13),(23,14),(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(22,23),(23,24),(17,25),(21,26)])
G2.append(G2_13)


# #IBM Rochester - 53 qubits
G2_14 = nx.Graph()
G2_14.add_edges_from([(0,1),(1,2),(2,3),(3,4),(0,5),(4,6),(5,9),(6,13),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),
(7,16),(16,19),(15,18),(11,17),(17,23),(18,27),(19,20),(20,21),(21,22),(22,23),(23,24),(24,25),(25,26),(26,27),(21,28),(28,32),(25,29),
(29,36),(30,31),(31,32), (32,33),(33,34),(34,35),(35,36),(36,37),(37,38),(30,39),(39,42),(34,40),(40,46),(38,41),(41,50),(42,43),(43,44),(44,45),(45,46),(46,47),(47,48),(48,49),(48,49),(49,50),(44,51),(48,52)])
G2.append(G2_14)

#Google Bristlecone - 72 qubits
G2_15 = nx.Graph()
G2_15.add_edges_from([(0,1),  (1,2),  (2,3),  (3,4),  (4,5), (5,6),  (6,7), (7,8),  (8,9), (9,10),  (10,11), 
(12,13), (13,14),  (14,15),  (15,16), (16,17), (17,18), (18,19),  (19,20), (20,21),  (21,22),  (22,23), 
(24,25),  (25,26),  (26,27), (27,28),(28,29), (29,30), (31,32),  (32,33),  (33,34),  (34,35), 
(36,37), (37,38),  (38,39),  (39,40), (40,41), (42,43),  (43,44),  (44,45), (45,46),  (46,47), 
(48,49),(49,50),(50,51), (51,52),  (52,53), (53,54), (54,55),  (55,56),  (56,57), (57,58), (58,59),
(60,61),  (61,62), (62,63),  (63,64), (64,65),  (65,66),  (66,67), (67,68),  (68,69),  (69,70),  (70,71), 
(1,12),(1,14),(14,3),(3,16),(16,5),(5,18),(18,7),(7,20),(9,20),(9,22),(22,11),
(13,24),(13,26),(15,26),(15,28),(28,17),(30,17),(30,19),(19,32),(32,21),(21,34),(34,23),
(25,36),(25,38),(38,27),(40,27),(40,29),(42,29),(42,31),(31,44),(44,33),(33,46),(46,35),
(37,48),(37,50),(50,39),(39,52),(52,41),(41,54),(54,43),(43,56),(45,56),(45,58),(58,47),
(49,60),(49,62),(62,51),(51,64),(64,53),(53,66),(66,55),(55,68),(68,57),(57,70),(70,59)])
G2.append(G2_15)


#find all subgraphs of specific size: if some are isomorphic discard them
#alternative version for scalability: find the max connected subgraphs of specific size (not the focus here)
def getAllDistinctSubgraphs(num_of_nodes, c_graph):
    subgraph_list = []
    subgraphs = []

    #find all connected subgraphs:
    subgraphs += itertools.combinations(c_graph, num_of_nodes)
    for sg in subgraphs:
        if nx.is_connected(nx.induced_subgraph(c_graph,sg)):
           subgraph_list.append(nx.induced_subgraph(c_graph,sg)) 

    #sort the subgraphs based on number of edges, max degree and max degree in neighborhood (graph isomorphism process)
    subgraph_list.sort(key=lambda x: nx.number_of_edges(x))
    subgraph_list.sort(key=lambda x: max(dict(x.degree()).values()))
    subgraph_list.sort(key=lambda x: max(dict(nx.average_neighbor_degree(x)).values()))
    

    #discard subgraps that are isomorphic
    if len(subgraph_list) > 1:
        for sg, sg2 in itertools.combinations(subgraph_list,2):
            if GraphMatcher(sg,sg2).is_isomorphic() and sg2 in subgraph_list:
                subgraph_list.remove(sg2)
    

    return subgraph_list

iter=-1
for cg in G2: 
    iter+=1
    if G1.number_of_nodes() <= cg.number_of_nodes():
        GM = GraphMatcher(cg,G1)
        print(GM.subgraph_is_isomorphic())

        initial_placement = []
        distance = []
        missing_edges = []
        short_paths = []
        convert_edit = []
        max_swaps_bound = []


        #if isomorphism is possible:
        if GM.mapping:
            G2_trace = cg.subgraph(GM.mapping.keys())
            distance=nx.graph_edit_distance(G2_trace,G1)
            initial_placement = [nx.nodes(G2_trace), nx.edges(G2_trace)] 
            max_swaps_bound = 0
            OEP_nodes = nx.optimal_edit_paths(G1, G2_trace)[0][0][0]

        n=0
        #if isomorphism is not possible:
        if not GM.subgraph_is_isomorphic(): 
            subgraphs = getAllDistinctSubgraphs(G1.number_of_nodes(), cg)
            print (subgraphs)
            distance = 14*cg.number_of_edges()
            fin_subgraph = []
            for subgraph in subgraphs:
                new_distance = nx.graph_edit_distance(subgraph,G1)
                if new_distance <= distance:
                    distance = new_distance
                    fin_subgraph = subgraph
            OEP = nx.optimal_edit_paths(fin_subgraph, G1)[0][0]
            OEP_nodes = nx.optimal_edit_paths(G1, fin_subgraph)[0][0][0]
            print("The final subgraph is with nodes:", nx.nodes(fin_subgraph), "with distance: ", distance, "and edges: ", nx.edges(fin_subgraph))
            initial_placement = [nx.nodes(fin_subgraph), nx.edges(fin_subgraph)]
            print("Optimal edit paths: ", OEP)
            print("Initial placement: ", OEP_nodes)


        # find missing edges

            for edit in OEP[1]:
                if edit[0] is None:
                    n+=1
                    for i in edit[1]:
                        for j in OEP[0]:
                            if i == j[1]:
                                convert_edit = j[0]  #convert the missing edges locations in CG 
                                missing_edges.append(convert_edit)
            missing_edges = np.asarray(missing_edges).reshape((n,2)).tolist()

        #find shortest paths between the nodes of missing edges and total length of shortest paths
            # for edge in missing_edges:
            #     short_paths.append((nx.dijkstra_path(fin_subgraph, edge[0], edge[1])))

            # short_path_len = 0
            # short_path_ind_subgraphs = []
            # short_path_lens = []
            # for short_path in short_paths:
            #     short_path_ind_subgraphs.append(G2.subgraph(short_path))

            # for short_path_sub in short_path_ind_subgraphs:
            #     short_path_len += nx.number_of_edges(short_path_sub)
            #     short_path_lens.append(nx.number_of_edges(short_path_sub))
            # max_len_path = max(short_path_lens)

        # #calculate maximal bound for number of SWAPS:
            max_swaps_bound = twoqgates * (nx.diameter(fin_subgraph)-1)
        
        print(devices_qubits[iter][0])
        data.append([bench_names, no_of_qubits, devices_qubits[iter][0], devices_qubits[iter][1], OEP_nodes, initial_placement, distance, missing_edges, max_swaps_bound]
        )

df =  pd.DataFrame(data, columns =   ["benchmarks names","no of qubits","device name","number of physical qubits", "initial placement nodes","mapped to subgraph","graph distance", "missing edges","upper bound for swaps"]) 
df.to_excel(curdir + "/initial_placement.xlsx", index=False)
