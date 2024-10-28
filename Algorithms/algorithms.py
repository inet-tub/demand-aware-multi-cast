
import random
import networkx as nx
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import glob
import time
from xml.dom import minidom
import copy
import math

"""## Reading graph from input

"""

def readingGMLFile():
  gml_files = sorted(glob.glob("/Users/pourdamghani/Desktop/Mcast/*.gml"))
  graphs = [nx.read_gml(file, label='id') for file in gml_files]
  return graphs

def get_edge_bandwidths(graph):
    caps = np.zeros((graph.number_of_nodes()+1, graph.number_of_nodes()+1))
    for edge in graph.edges(data=True):
      u,v, data = edge
      caps[u][v] = data.get('bandwidth')/(1000*1000*1000*10)
    return caps

def gettingZooGraphs():
    #path = "/content/drive/MyDrive/zoo/*.xml"
    path = "/Users/pourdamghani/Documents/zoo/*.graphml"
    Zlist = list(glob.glob(path))
    graphs = []
    for zoo in Zlist:
      temp = nx.read_graphml(zoo)
      temp.remove_nodes_from(list(nx.isolates(temp)))
      final = nx.DiGraph()
      tempMax = 0
      for node in temp.nodes:
        final.add_node(int(node))
        tempMax = max(tempMax,int(node))
      for edge in temp.edges():
          edgeTemp = list(edge)
          final.add_edge(int(edgeTemp[0]), int(edgeTemp[1]))
          final.add_edge(int(edgeTemp[1]), int(edgeTemp[0])) #add reverse edge to become undirected
      final.graph['myLabel'] = zoo
      if(nx.is_strongly_connected(final) and tempMax == len(final.nodes)-1):
        graphs.append(final)
    graphs.sort(key=len)
    return graphs

graphs = gettingZooGraphs()
count = 1
for graph in graphs:
  name = "/Users/pourdamghani/Documents/zoo/" + str(count) + ".gml"
  count+=1
  nx.write_gml(graph, name)

string_to_int_map = {}
next_available_id = 0

def mapSNDID(input_string):
    global next_available_id

    if input_string in string_to_int_map:
        return string_to_int_map[input_string]

    string_to_int_map[input_string] = next_available_id
    next_available_id += 1

    return string_to_int_map[input_string]



def gettingSNDLibGraphs():
  path = "/Users/pourdamghani/Documents/sndlib/*.xml"
  Zlist = list(glob.glob(path))
  graphs = []
  for zoo in Zlist:
    read_xml = minidom.parse(zoo)
    node_list = read_xml.getElementsByTagName('node')
    edge_list = read_xml.getElementsByTagName('link')
    demand_list = read_xml.getElementsByTagName('demand')
    newGraph = nx.DiGraph()
    for node in node_list:
      nodeID = node.getAttribute('id')
      newGraph.add_node(mapSNDID(nodeID),demand = 0)
    for edge in edge_list:
      s = edge.getElementsByTagName('source')[0].firstChild.data
      t = edge.getElementsByTagName('target')[0].firstChild.data
      cap = edge.getElementsByTagName('capacity')[0].firstChild.data,
      newGraph.add_edge(u_of_edge=mapSNDID(s),v_of_edge=mapSNDID(t), capacity = cap)
    for demand in demand_list:
      s = demand.getElementsByTagName('source')[0].firstChild.data
      dem = demand.getElementsByTagName('demandValue')[0].firstChild.data
      newGraph.nodes[mapSNDID(s)]['demand']+=float(dem)
  return graphs
gettingSNDLibGraphs()

def getPopTelekom(telekomGraph):
  return [data.get('pop') for u, data in telekomGraph.nodes(data=True)]

def getCapTelekom(graph):
  caps = np.zeros((graph.number_of_nodes()+1, graph.number_of_nodes()+1))
  for edge in graph.edges(data=True):
    u,v, data = edge
    caps[u][v] = data.get('cap')
  return caps

def getingTelekom():
  gml_files = glob.glob("/Users/pourdamghani/Desktop/telekom.gml")
  graphs = [nx.read_gml(file, label='id') for file in gml_files]
  telekomGraph = graphs[0]
  return telekomGraph

"""## Instance Generation



"""

class Instance:

  maxCap = 100*1000
  minCap = 100
  maxBandwidth = 2000
  minBandwidth = 10

  def generateRandCapacites(self,curMin,curMax):
    temp = [[random.randrange(curMin, curMax)	for _ in range(self.numNodes)] for _ in range(self.numNodes)]
    for i in range(self.numNodes):
      for j in range(i,self.numNodes):
        temp[i][j] = temp[j][i]
    return temp

  def createRandSources(self):
    return random.sample(range(0, self.numNodes), random.randint(1,self.numNodes))

  def createRandReceivers(self,s):
    current = random.sample(range(0, self.numNodes), random.randint(1,self.numNodes))
    if(s in current):
      current.remove(s)
    return current

  def createRandDemand(self,curMin,curMax):
    return np.random.choice(range(curMin, curMax), size=self.numNodes)

  def isNotSenOrRec(self,node,s):
    return (not (node == s)) and (not (node in self.Receivers[s]))

  def localNetGen(self,dimension):
    tempGraph = nx.grid_2d_graph(dimension, dimension)
    self.graph = nx.DiGraph()
    for edge in tempGraph.edges:
      u, v = edge
      u1, u2 = u
      v1, v2 = v
      self.graph.add_edge(u1*dimension+u2,v1*dimension+v2)
      self.graph.add_edge(v1*dimension+v2,u1*dimension+u2)
    avegNodeCap = int((Instance.minCap+Instance.maxCap)/2)
    self.EdgeCapacites = self.generateRandCapacites(avegNodeCap,avegNodeCap+1)
    avegNodeBand = int((Instance.minBandwidth+Instance.maxBandwidth)/2)
    self.BandwidthDemand = self.createRandDemand(avegNodeBand,avegNodeBand+1)

  def GravityModel(self):
    self.frictionFactor = 1000*1000*1000*10
    pop = getPopTelekom(self.graph)
    demand = np.zeros((self.numNodes + 1, self.numNodes + 1))
    for i in range(self.numNodes ):
      for j in range(self.numNodes ):
        demand[i][j] = int(pop[i])*int(pop[j])/self.frictionFactor

    outputDemand = np.zeros((self.numNodes + 1))
    for s in self.Sources:
      sum = 0
      count = 0
      for r in self.Receivers[s]:
        sum+= demand[s][r]
        count+=1
      outputDemand[s] = sum/count
      #print(outputDemand[s])
    return outputDemand

  def randNetGen(self,graph,numNodes, typeSim):
    self.graph = graph
    self.numNodes = numNodes
    if(typeSim == "zoo"):
      self.EdgeCapacites = self.generateRandCapacites(Instance.minCap,Instance.maxCap)
      self.BandwidthDemand = self.createRandDemand(Instance.minBandwidth,Instance.maxBandwidth)

    if(typeSim == "iGen"):
      self.BandwidthDemand = self.createRandDemand(Instance.minBandwidth,Instance.maxBandwidth)
      self.EdgeCapacites = get_edge_bandwidths(graph)


  def __init__(self,graph,numNodes,typeSim):
    if (typeSim == "telekom"):
      self.graph = graph
      self.numNodes = len(graph.nodes)
      self.nodes = self.graph.nodes
      self.edges = self.graph.edges
      self.maxValue = self.maxCap*(len(self.nodes)) + 2
      self.Sources = self.createRandSources()
      self.Receivers = {}
      for s in self.Sources:
        self.Receivers[s] = self.createRandReceivers(s)

      self.EdgeCapacites = getCapTelekom(graph)
      self.BandwidthDemand = self.GravityModel()
      return

    if(typeSim == "iGen"):
      self.randNetGen(graph,numNodes,typeSim)
    if(typeSim == "sndlib"):
      self.numNodes = numNodes
      self.graph = graph
      self.edgeCapacites = np.zeros((num_Nodes + 1, num_Nodes + 1))
      for u, v, data in G.edges(data=True):
        self.EdgeCapacites[u][v] = data.get('capacity', 0)
      self.BandwidthDemand = [data['demand'] for node, data in graph.nodes(data=True)]
    if (typeSim=="zoo"):
      self.randNetGen(graph,numNodes,typeSim)

    if (typeSim=="local"):
      self.numNodes = numNodes*numNodes
      self.localNetGen(numNodes)
    self.nodes = self.graph.nodes
    self.edges = self.graph.edges
    self.maxValue = self.maxCap*(len(self.nodes)) + 2
    self.Sources = self.createRandSources()
    self.Receivers = {}
    for s in self.Sources:
      self.Receivers[s] = self.createRandReceivers(s)

"""# Algorithms

## MIP

### Variables
"""

def edgeToInt(edge, n):
  u, v = edge
  return u*n+v

def edgeToUndirect(edge,n):
  u,v = edge
  if (u > v):
    u,v = v,u
  return u*n+v

class Variables:
  weight = {} #l_{(u,v)}
  active = {} #a_{(u,v)}
  blended = {} #b^s_{(u,v)}
  flow = {} #f^{s,r}_{(u,v)}

  dis = {} #dis_{u,v}
  valid = {} # y_{u,v,w}
  env = gp.Env()
  model = gp.Model("Multicast",env=env)
  def __init__(self,inst):
    self.congest = self.model.addVar(lb = -Instance.maxCap, vtype=GRB.INTEGER)
    for edge in inst.edges:
      u,v = edge
      edgeUn = edgeToUndirect((u,v),inst.numNodes)
      edgeNum1 = edgeToInt((u,v),inst.numNodes)
      edgeNum2 = edgeToInt((v,u),inst.numNodes)
      self.weight[edgeUn] = self.model.addVar(lb=1,vtype = GRB.INTEGER) #, name = nameConstructer("weight",[edge])
      self.active[edgeUn] = self.model.addVar(vtype = GRB.BINARY, name = "active["+str(edge)+"]") #, name = nameConstructer("active",[source,edge])

      for s in inst.Sources:
        self.blended[s,edgeNum1] = self.model.addVar(vtype = GRB.BINARY, name = "blended["+str(s)+","+str(edgeNum1)+"]")
        self.blended[s,edgeNum2] = self.model.addVar(vtype = GRB.BINARY, name = "blended["+str(s)+","+str(edgeNum2)+"]")

        for r in inst.Receivers[s]:
          self.flow[s,r,edgeNum1] = self.model.addVar(vtype = GRB.BINARY, name = "flow["+str(s)+","+str(r)+","+str(edgeNum1)+"]")
          self.flow[s,r,edgeNum2] = self.model.addVar(vtype = GRB.BINARY, name = "flow["+str(s)+","+str(r)+","+str(edgeNum2)+"]")


    for u in inst.nodes:
      for v in inst.nodes:
        self.dis[u,v] = self.model.addVar(vtype = GRB.INTEGER)
        for w in inst.nodes:
          self.valid[u,v,w] = self.model.addVar(vtype = GRB.BINARY)

"""### Constraints

"""

def geningOut(u,inst):
  for value in inst.nodes:
      if inst.graph.has_edge(u,value):
        yield value

def geningIn(u,inst):
  for value in inst.nodes:
      edge = (value,u)
      if (inst.graph.has_edge(value,u)):
        yield value


def construct_constraints(inst, myVars):

  for u in inst.nodes:
    myVars.model.addLConstr(myVars.dis[u,u], GRB.EQUAL, 0)
  for edge in inst.edges:
    u,v = edge
    edgeUn = edgeToUndirect(edge,inst.numNodes)
    edgeNum1 = edgeToInt((u,v),inst.numNodes)
    edgeNum2 = edgeToInt((v,u),inst.numNodes)
    myVars.model.addLConstr(myVars.dis[u,v], GRB.LESS_EQUAL, myVars.weight[edgeUn])
    myVars.model.addLConstr(myVars.dis[v,u], GRB.LESS_EQUAL, myVars.weight[edgeUn])
    disEdge = gp.LinExpr()
    disEdge.addTerms(1,myVars.weight[edgeUn])
    disEdge.addTerms(inst.maxValue,myVars.active[edgeUn])
    myVars.model.addLConstr(myVars.dis[u,v], GRB.GREATER_EQUAL, disEdge-inst.maxValue)
    myVars.model.addLConstr(myVars.dis[v,u], GRB.GREATER_EQUAL, disEdge-inst.maxValue)
    myVars.model.addLConstr(myVars.dis[v,u], GRB.EQUAL, myVars.dis[u,v])
    congestSum = gp.LinExpr()
    for s in inst.Sources:
      congestSum.addTerms(inst.BandwidthDemand[s],myVars.blended[s,edgeNum1])
      congestSum.addTerms(inst.BandwidthDemand[s],myVars.blended[s,edgeNum2])
      myVars.model.addLConstr(myVars.blended[s,edgeNum1]+myVars.blended[s,edgeNum2], GRB.LESS_EQUAL, 1)
    myVars.model.addLConstr(myVars.congest, GRB.GREATER_EQUAL, congestSum-inst.EdgeCapacites[u][v])

  for u in inst.nodes:
    for v in inst.nodes:
      sumValidActiveOne = gp.LinExpr()
      myVars.model.addLConstr(myVars.dis[u,v],GRB.EQUAL,myVars.dis[v,u])
      for w in inst.nodes: # check if w can be u or v
        if( w!=u and w!=v):
          sumWithMiddle = gp.LinExpr()
          sumWithMiddle.addTerms(1,myVars.dis[u,w])
          sumWithMiddle.addTerms(1,myVars.dis[w,v])
          myVars.model.addLConstr(myVars.dis[u,v], GRB.LESS_EQUAL,sumWithMiddle)
          sumMax = gp.LinExpr()
          sumMax.addTerms(1,myVars.dis[u,w])
          sumMax.addTerms(1,myVars.dis[w,v])
          sumMax.addTerms(inst.maxValue,myVars.valid[u,v,w])
          myVars.model.addLConstr(myVars.dis[u,v], GRB.GREATER_EQUAL, sumMax-inst.maxValue)
          sumValidActiveOne.addTerms(1,myVars.valid[u,v,w])
      if( (u,v) in inst.edges):
        edgeUn = edgeToUndirect(edge,inst.numNodes)
        sumValidActiveOne.addTerms(1,myVars.active[edgeUn])
      myVars.model.addLConstr(sumValidActiveOne,GRB.GREATER_EQUAL,1)

  for s in inst.Sources:
    for edge in inst.edges:
      u,v = edge
      edgeUn = edgeToUndirect(edge,inst.numNodes)
      edgeNum1 = edgeToInt((u,v),inst.numNodes)
      edgeNum2 = edgeToInt((v,u),inst.numNodes)
      myVars.model.addLConstr(myVars.blended[s,edgeNum1], GRB.LESS_EQUAL, myVars.active[edgeUn])
      myVars.model.addLConstr(myVars.blended[s,edgeNum2], GRB.LESS_EQUAL, myVars.active[edgeUn])
    for r in inst.Receivers[s]:
      for edge in inst.edges:
        u,v = edge
        edgeNum1 = edgeToInt((u,v),inst.numNodes)
        edgeNum2 = edgeToInt((v,u),inst.numNodes)
        myVars.model.addLConstr(myVars.flow[s,r,edgeNum1], GRB.LESS_EQUAL, myVars.blended[s,edgeNum1])
        myVars.model.addLConstr(myVars.flow[s,r,edgeNum2], GRB.LESS_EQUAL, myVars.blended[s,edgeNum2])
        myVars.model.addLConstr(myVars.flow[s,r,edgeNum1]+myVars.flow[s,r,edgeNum2], GRB.LESS_EQUAL, 1)
      myVars.model.addLConstr(gp.quicksum(myVars.flow[s,r,edgeToInt((s,edgeOut),inst.numNodes)] for edgeOut in geningOut(s,inst)), GRB.EQUAL, 1)
      myVars.model.addLConstr(gp.quicksum(myVars.flow[s,r,edgeToInt((edgeIn,r),inst.numNodes)] for edgeIn in geningIn(r,inst)), GRB.EQUAL, 1)
      for u in inst.nodes:
        if (inst.isNotSenOrRec(u,s)):
          myVars.model.addLConstr(gp.quicksum(myVars.flow[s,r,edgeToInt((u,v),inst.numNodes)] for v in geningOut(u,inst)), GRB.EQUAL, gp.quicksum(myVars.flow[s,r,edgeToInt((v,u),inst.numNodes)] for v in geningIn(u,inst)))

"""### Objective"""

def construct_objectives(myVars):
  myVars.model.setObjective(myVars.congest, GRB.MINIMIZE)

"""### Solving MIP

"""

def MIP(inst):
  myVars = Variables(inst)
  construct_constraints(inst, myVars)
  construct_objectives(myVars)
  myVars.model.setParam('OutputFlag', False)   #Quite Opetimzation
  myVars.model.optimize()
  if (myVars.model.SolCount > 0):
    return myVars.model.ObjVal
  else:
    return -1

"""## Overlapping MBSTs

### Finding Congestion Using Weight
"""

def findShortestTree(s,graph,weights,rec):
  tempGraph = graph.copy()
  tempDict = {}
  for edge in graph.edges:
     u,v = edge
     tempDict.update({(u,v):weights[u][v]})
  nx.set_edge_attributes(tempGraph,tempDict,name="weights")
  paths = nx.shortest_path(tempGraph, source=s, target=None, method='dijkstra', weight="weights")
  allEdges = set()
  for d in paths:
    tempPath = paths[d]
    if (d in rec[s] and len(paths[d]) > 1):
      for i in range(len(paths[d])-1):
        allEdges.add((int(paths[d][i]),int(paths[d][i+1])))
  return allEdges

def calcCostByWeights(inst,nowWeights):
  finalCap = np.zeros((inst.numNodes,inst.numNodes))
  for s in inst.Sources:
    tree = findShortestTree(s,inst.graph,nowWeights,inst.Receivers)
    for edge in tree:
      u, v = edge
      finalCap[u][v]+= inst.BandwidthDemand[s]
  congest = -inst.maxValue
  for edge in inst.edges:
    u,v = edge
    congest = max(congest,finalCap[u][v]+finalCap[v][u]-inst.EdgeCapacites[u][v])
  return congest

"""### Minimum Bottelneck Spanning Tree

"""

def findBottelTree(s,nowGraph,capacities,receivers):
  listOfEdges = []
  for edge in nowGraph.edges:
    u,v = edge
    listOfEdges.append((capacities[u][v],edge))
  listOfEdges.sort(reverse=True)
  counter = 0
  validEdges = 0
  tree = nx.Graph()
  tree.add_nodes_from(nowGraph.nodes)
  while (counter < len(listOfEdges)):
    tempCap, tempEdge = listOfEdges[counter]
    counter+=1
    tempU , tempV = tempEdge
    tree.add_edge(tempU , tempV)
    pathChecker = True
    for r in receivers:
      if(not nx.has_path(tree,s,r)):
        pathChecker = False
        break
    if (pathChecker):
      break
  listOfEdgesForReceivers = set()
  for r in receivers:
    path = nx.shortest_path(tree, source=s, target=r)
    listOfEdgesForReceivers.update({(path[i], path[i+1]) for i in range(len(path)-1)})
  return listOfEdgesForReceivers

def byDemandSortSources(inst):
  listAll = []
  for s in inst.Sources:
    listAll.append((inst.BandwidthDemand[s],s))
  listAll.sort(reverse=True)
  return listAll

"""### Overalapping Trees"""

def overlapWegihts(inst):
  sortedByDemand = byDemandSortSources(inst)
  tempCapacities = copy.deepcopy(inst.EdgeCapacites)
  tempGraph = inst.graph.copy()
  hWeights = [[math.inf	for _ in range(inst.numNodes)] for _ in range(inst.numNodes)]
  for pair in sortedByDemand:
    band,s = pair
    tree = findBottelTree(s,tempGraph,tempCapacities,inst.Receivers[s])
    for edge in tree:
      u, v = edge
      if (hWeights[u][v] == math.inf):
        hWeights[u][v] = float(inst.maxCap/band)

  return hWeights

def overlap(inst):
  return calcCostByWeights(inst,overlapWegihts(inst))

"""### Overlapping and removing highly loaded edges"""

def overlapRemoveWeights(inst):
  sortedByDemand = byDemandSortSources(inst)
  tempCapacities = copy.deepcopy(inst.EdgeCapacites)
  tempGraph = inst.graph.copy()
  hWeights = [[0	for _ in range(inst.numNodes)] for _ in range(inst.numNodes)]
  for pair in sortedByDemand:
    band,s = pair
    tree = findBottelTree(s,tempGraph,tempCapacities,inst.Receivers[s])
    for edge in tree:
      u, v = edge
      if (hWeights[u][v] == 0):
        hWeights[u][v] = float(1/band) * Instance.maxBandwidth
      tempCapacities[u][v]-=band
      if (tempCapacities[u][v] < 0):
        tempGraph.remove_edge(u,v)
        if(not nx.is_strongly_connected(tempGraph)):
          tempGraph.add_edge(u,v)
  return hWeights

def overlapRemove(inst):
  resNow = overlapRemoveWeights(inst)
  return calcCostByWeights(inst,overlapRemoveWeights(inst))

"""## 1/Cap"""

def oneOverCap(inst):
  myWeights = [[0	for _ in range(inst.numNodes)] for _ in range(inst.numNodes)]
  for edge in inst.edges:
      u, v = edge
      myWeights[u][v] = float(1/inst.EdgeCapacites[u][v]) * Instance.maxCap
  return calcCostByWeights(inst,myWeights)

"""## Post Processings

### Cobmined 1/Cap and MBST
"""

def overlapPlusOneOverCap(inst):
  currentWeights = overlapWegihts(inst)
  congestAlg = overlap(inst)
  congestTrad = oneOverCap(inst)
  multFactor = 1
  for edge in inst.edges:
      u, v = edge
      currentWeights[u][v] += float(1/inst.EdgeCapacites[u][v])*Instance.maxCap*multFactor
  return calcCostByWeights(inst,currentWeights)

"""### Changing trees based on cap

# Comparing Algorithms
"""

# For all senders, starting with the one with lowest bandwidth requirement:
def pickyAlg(inst):
  sortedSources = [] #sort source by their demand
  for s in inst.Sources:
    sortedSources.append((inst.BandwidthDemand[s],s))
  sortedSources.sort()
  capacities = copy.deepcopy(inst.EdgeCapacites)

  PWeights = [[math.inf	for _ in range(inst.numNodes)] for _ in range(inst.numNodes)]


  tree = nx.DiGraph()
  tree.add_nodes_from(inst.nodes)
  for pair in sortedSources:
    bandwidthNow, s = pair
    listOfEdges = []
    receivers = inst.Receivers[s]
    for edge in inst.edges:
      u,v = edge
      if (capacities[u][v] >= bandwidthNow): #Only consider edges if there is enough capacity
        listOfEdges.append((capacities[u][v],(u,v)))
      if (capacities[v][u] >= bandwidthNow): #Only consider edges if there is enough capacity
        listOfEdges.append((capacities[v][u],(v,u)))
    listOfEdges.sort()
    counter = 0


    while (counter < len(listOfEdges)):
      tempCap, tempEdge = listOfEdges[counter]
      counter+=1
      pathChecker = True
      for r in receivers:
        if(not nx.has_path(tree,s,r)):
          pathChecker = False
          break
      if (pathChecker == True):
        break
      tempU , tempV = tempEdge
      tree.add_edge(tempU , tempV, weights = float(1/inst.EdgeCapacites[u][v]) * Instance.maxCap )
    pathChecker = True
    for r in receivers:
      if(not nx.has_path(tree,s,r)):
        pathChecker = False
        break
    if (pathChecker == False):
      return oneOverCap(inst)

    allEdgesNow = set()
    for r in receivers:
      path = nx.shortest_path(tree, source=s, target=r,  method='dijkstra', weight="weights")
      for i in range(len(path)-1):
        u = path[i]
        v = path[i+1]
        allEdgesNow.add((u,v))
    for nowEdge in allEdgesNow:
      u,v = nowEdge
      capacities[u][v] -= bandwidthNow
      PWeights[v][u] = float(1/inst.EdgeCapacites[u][v]) * Instance.maxCap
  overCapcityCognestion = oneOverCap(inst)
  ourCongestion = calcCostByWeights(inst,PWeights)

  if (ourCongestion > overCapcityCognestion):
    return overCapcityCognestion

  return ourCongestion



numAlgs = 10
def computeAlgs(graph,lenNodes,typeSim):
  instance = Instance(graph,lenNodes,typeSim)
  congestAlgs = [0]*numAlgs
  timingAlgs = [0]*numAlgs

  start_time = time.time()
  congestAlgs[4] = pickyAlg(instance)
  end_time = time.time()
  timingAlgs[4] = end_time-start_time

  start_time = time.time()
  congestAlgs[0] = oneOverCap(instance)
  end_time = time.time()
  timingAlgs[0] = end_time-start_time


  start_time = time.time()
  congestAlgs[1] =  overlap(instance)
  end_time = time.time()
  timingAlgs[1] = end_time-start_time

  start_time = time.time()
  congestAlgs[2] =  overlapPlusOneOverCap(instance)
  end_time = time.time()
  timingAlgs[2] = end_time-start_time

  if(lenNodes < 40):
    start_time = time.time()
    congestAlgs[3] =  MIP(instance)
    end_time = time.time()
    timingAlgs[3] = end_time-start_time

  return [congestAlgs,timingAlgs]