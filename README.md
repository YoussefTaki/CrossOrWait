# Is the pedestrian going to cross or wait? Predicting Pedestrian
Crossing Intention with Graph Neural Network


<br />
We propose a Graphs Convolutional Network (GCN), which collects information about the pedestrians, Driver, and the environment in a real driving scenario  by focusing on extracting the largest number of information and creating a graph,  Our results show an improvement over the state of art by 8% on Accuracy.



#### Model
<div align='center'>
  <img src="images/models.png" width="700" height="400" >
</div>

<br />
GCNN model consists of 2 building phases: <br />
1- Graph Construction: At this point, we build our graph based on the collected information, using a graph with 3 nodes (driver, pedestrian, environment) and then calculate the adjacency matrix to feed them together into our GNN model.


<br />
<div align='center'>
<img src="images/graph.png"  width="500" height="400" ></img>
</div>


2- GNN: This model consists of three graph convolution layers followed by a linear classifier.
Each employs ReLU activation functions after processing. The final layer conducts global mean pooling to
aggregate node embeddings, followed by the dropout Layer and a linear layer as a classifier.


<br />
<div align='center'>
<img src="images/Gnn2.png"  width="600" height="300" ></img>
</div>


### Setup: 
The code was written using Python and Pytorch 
The following libraries are the minimal to run the code: 
```python
import PyTorch
import networkx
import sklearn
import torch_geometric
```
or you can have everything set up by running: 
```bash
pip install -r requirements.txt
```


To train and test the GNN model fas in the paper, simply run:
```bash
Graph_main.py
```
