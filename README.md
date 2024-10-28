In this repository, you can find the code for our project on demand-aware multi-source IP-multicast.

# Running the code

## Installing and Running Gurobi with Python
To be able to run our code, you need to install Gurobi first.

1. Download Gurobi from the [Gurobi website](https://www.gurobi.com/downloads/gurobi-software/).
2. Follow installation instructions for your OS.
3. Activate the appropriate license using
```console
   grbgetkey YOUR_LICENSE_KEY
```
4. Install Gurobi Python Package:
```console
  pip3 install gurobipy
```

## Selecting Network Topology
In this project, we cover topologies from a few sources that you can select in [algorithms.py](Algorithms/algorithms.py).
1. [The Internet Topology Zoo](https://topology-zoo.org/): You can download topologies from their website, and then include .graphml files in an appropriate folder.
2. [iGen](https://igen.sourceforge.net/): After installing the tool mentioned in the website, you can create graphs with desired properties described in the paper.
3. Campus network: we hardcoded the creation of this type of network in our code.
4. National ISP network: unfortunately, we can not provide this network publicly.
5. SNDlib networks: the code also can work with networks from [SNDlib](https://sndlib.put.poznan.pl/home.action), however, we didn't include results from those networks in our current version of the paper.

Then based on each topology, we create the source and destination pairs, demand between demand, and capacity of edges accordingly, if needed. However, you might still need to set the size of the graphs that you want to use in the code.

## Visualization

After setting up what is discussed above, you can run the visualization that you prefer and mention it in the visualization folder.
