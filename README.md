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

## Selecting appropriate network topology
In this project, we cover topologies from a few sources:
1. [The Internet Topology Zoo](https://topology-zoo.org/): You can download topologies from their website, and then include .graphml files in an appropriate folder.
2. [iGen](https://igen.sourceforge.net/): After installing the tool mentioned in the website, you can create graphs with desired properties described in the paper.
3. Campus network: we hardcoded creating of this type of graphs in our code.
4. National ISP network: unforutnatly, we can not provide this network publicly.
5. SNDlib networks: the code also can work with networks from [SNDlib](https://sndlib.put.poznan.pl/home.action), however, we didn't include result from those networks in our current version of the paper.


## Selecting appropriate network topology

