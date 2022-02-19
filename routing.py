import networkx as nx
import numpy as np
import itertools
import matplotlib.pyplot as plt
import gym
import ql
import time

G1=nx.DiGraph()
list_nodes = [1,2,3,4,5,6,7,8,9,10] #### pocisiones [0,1,2,3,4]
acciones = list_nodes
Actionsx1= [2,3]             #### acciones posibles para cada nodo
Actionsx2= [2,5]
Actionsx3= [0,1,3,4,5]
Actionsx4= [0,2,4,6,7]
Actionsx5= [2,3,5,7,8]
Actionsx6= [1,2,4,8,9]
Actionsx7= [3,7]
Actionsx8= [3,4,6,8]
Actionsx9= [4,5,7,9]
Actionsx10= [5,8]

G1.add_nodes_from(list_nodes)
G1.nodes()

#weights = [50,90,50,90,50,50,50,50,50,50,50,50,50,50,50,50,50,50]
weights = [50,90,50,90,50,90,50,50,50,90,50,50,50,50,90,50,50,50]
list_arcs1 = [(1,3,weights[0]), (3,1,weights[0]), (1,4,weights[1]) , (4,1,weights[1]) , (2,3,weights[2]), (3,2,weights[2]) , (2,6,weights[3]) , (6,2,weights[3]) , (3,4,weights[4]) , (4,3,weights[4]) , (3,5,weights[5]) , (5,3,weights[5]) ,  (3,6,weights[6]) , (6,3,weights[6]) ,(4,5,weights[7]), (5,4,weights[7]), (5,6,weights[8]), (6,5,weights[8]), (4,7,weights[9]), (7,4,weights[9]), (4,8,weights[10]), (8,4,weights[10]), (5,8,weights[11]), (8,5,weights[11]), (5,9,weights[12]), (9,5,weights[12]), (6,9,weights[13]), (9,6,weights[13]), (6,10,weights[14]), (10,6,weights[14]), (7,8,weights[15]), (8,7,weights[15]), (8,9,weights[16]), (9,8,weights[16]), (9,10,weights[17]), (10,9,weights[17])]
G1.add_weighted_edges_from(list_arcs1)
G1.edges()

G1.nodes[1]['pos'] = (0,-2)
G1.nodes[2]['pos'] = (0,2)
G1.nodes[3]['pos'] = (2.5,0)
G1.nodes[4]['pos'] = (5,-5)
G1.nodes[5]['pos'] = (6,0)
G1.nodes[6]['pos'] = (5,5)
G1.nodes[7]['pos'] = (10,-9)
G1.nodes[8]['pos'] = (10,-3)
G1.nodes[9]['pos'] = (10,3)
G1.nodes[10]['pos'] = (10,9)

node_pos=nx.get_node_attributes(G1,'pos')
nx.draw_networkx(G1, node_pos,node_size=450)
arc_weight=nx.get_edge_attributes(G1,'weight')
nx.draw_networkx_edge_labels(G1, node_pos, edge_labels=arc_weight)


l=[[1,2,3,4,5,6,7,8,9,10], [7,8,9,10], ['E','R']]
s = list(itertools.product(*l))
bandera=s
print(bandera)

def pesoEnlace(est, a):
    origen = bandera[est][0] 
    destino = a + 1
    for x in range(0,len(list_arcs1)):
        if (list_arcs1[x][0] == origen and list_arcs1[x][1] == destino):
            peso = list_arcs1[x][2]
    return peso 

def randomWeight():
    pesos = np.random.randint(20, 70, size=16)
    return pesos

def reset():
    aleatorio = np.random.randint(0, 80, size=1)
    return aleatorio[0]
def resetTest():
    aleatorio = np.random.randint(0, 16, size=1)
    return aleatorio[0]
def render(col,cond):
    map = []
    for node in G1:
        if node in col and cond == 'R':
            map.append('green')
        elif node in col and cond == 'E':
            map.append('red')
        else:
            map.append('gray')
    #nx.draw(G1, node_color=map, with_labels=True)
    nx.draw_networkx(G1, node_pos,node_size=450,node_color=map)
    nx.draw_networkx_edge_labels(G1, node_pos, edge_labels=arc_weight)
    plt.show()        
    
  
def ActionsXorigen(a1 ,a2 ,a3 ,a4 ,a5, a6, a7, a8, a9, a10 ,origen):  
    if (origen==1):
        return a1
    elif (origen==2):
        return a2
    elif (origen==3):
        return a3
    elif (origen==4):
        return a4
    elif (origen==5):
        return a5
    elif (origen==6):
        return a6
    elif (origen==7):
        return a7
    elif (origen==8):
        return a8
    elif (origen==9):
        return a9
    else:
        return a10
     
def step(s, a, posiblesAcciones, G1, saltos, _s):
    info={}
    imposibles = 0
    for x in range(0,len(posiblesAcciones)):
        if (a == posiblesAcciones[x]):
            imposibles = 1        
    if(imposibles == 0):                        # el destino no s vecino o se queda quieto
        reward = -70
        s_ = s
        done = False
    else:
        if (bandera[s][0] == bandera [s][1]):
            reward = 100
            s_ = s
            done = True
    
        else:
            done = False
            suma = a + 1
            for x in range(0,len(bandera)):
                if (suma == bandera[x][0] and bandera[s][1] == bandera[x][1] and bandera[s][2] == bandera[x][2]):
                    s_ = x
                    break
            #print(bandera[s][0], suma)
            #print (p)
            #if (bandera[s][2]=="E"):   
            if (s_ == _s):
                reward = -130
            else:
                reward = -10* saltos
                if (bandera[s][0] == 1 and bandera[s][1] == 10 and a == 2  or bandera[s][0] == 2 and bandera[s][1] == 7 and a == 2):
                    reward = reward + 3
                if (bandera[s][2] == 'E'):
                    if (pesoEnlace(s, a) > 79):
                        reward = reward - 130
                        
                                    
    #print (bandera[s],a,posiblesAcciones,s_,reward)
    _s = s
    return _s,s_,reward,done,info


if __name__ =="__main__":
    t = time.time()
    alpha = 0.4
    gamma = 0.999
    epsilon = 0.976
    episodes = 400000
    max_steps = 2500
    n_tests = 16
    n_states, n_actions = 80, 10
    agente = ql.QL_agent(alpha, gamma, epsilon, n_states,n_actions) #(alpha, gamma, epsilon, episodes, n_states, n_actions)
    
    episode_rewards = []
    
    for episode in range(episodes):
        print("Episode: {0}".format(episode))
        s = reset() 
        _s = s
        episode_reward = 0
        steps = 0
        done = False
        while steps < max_steps:
            steps += 1    
            a = agente.take_action(s,True)
            o = bandera[s][0]                        #origen
            acc = ActionsXorigen(Actionsx1,Actionsx2,Actionsx3,Actionsx4,Actionsx5,Actionsx6,Actionsx7,Actionsx8,Actionsx9,Actionsx10,o)     #acciones para dicho origen                   
            _s, s_, reward, done, info = step(s,a,acc,G1,steps,_s)
            #print(bandera[s],a,acc,s_,reward)
            episode_reward += reward
            a_ = np.argmax(agente.Q[s_,:])
            agente.updateQ(reward,s,a,a_,s_,done) 
            s, a = s_ , a_
            if done:
                end_ep = time.time()
                episode_rewards.append(episode_reward)
                break   
    print(bandera)
    print(acciones)
    #Test model 
                              
    for test in range(n_tests):
        print("Test #{0}".format(test))
        s = test                              #######################################reset
        _s = s
        done = False
        epsilon = 0
        st=0
        steps = 0
        color=[]
        while True:
            time.sleep(1)
            o = bandera[s][0]                        #origen
            acc = ActionsXorigen(Actionsx1,Actionsx2,Actionsx3,Actionsx4,Actionsx5,Actionsx6,Actionsx7,Actionsx8,Actionsx9,Actionsx10,o)
            #env.render()
            steps += 1
            if(st == 0):
                first_state=False;
            else:
                first_state=True;
            print("Estado actual: {0}".format(bandera[s]))
            color.append(bandera[s][0])
            a = agente.take_action(s,first_state)
            print("Chose action {0} for state {1}".format(a,s))
            #print(_s, s)
            first_state=True
            st=st+1;
            _s, s, reward, done, info = step(s,a,acc,G1,steps,_s)
            print(acc,reward,done)
            if done:
                render(color,bandera[s][2])
                print("Reached goal!")
                color.clear()
                break                
                time.sleep(6)
            
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("")
    plt.plot(episode_rewards,'b')
    plt.legend()
    plt.show()             
    """       
    print(bandera)
    print(acciones)
    
    """
    
    