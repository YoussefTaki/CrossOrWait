# STMGCN: A Spatio-Temporal Multi-Graph Convolutional Neural Network for Human Trajectory Prediction
### Taki Youssef

### Graph Model

<br />
we propose a Graphs Convolutional Network (GCN), which collects information about the social interactions of pedestrians , Driver, and Environement in a real driving scenario  by focusing on extracting the largest number of information and creating a graph,  Our results show an improvement over the state of art by 8% on the Accuracy.



#### Model
<br />
Social-STGCNN model consists of 2 building blocks: <br />
1- ST-GCNN: A Spatio-Tempral Graph CNN that creates a spatio-temporal graph embedding representing the previous pedestrians trajectories. <br />
2- TXP-CNN: A Time-Extrapolator CNN that utilizes the spatio-temporal graph embedding to predict future trajectories.<br />


### Setup: 
The code was written using python 3.6. 
The following libraries are the minimal to run the code: 
```python
import pytorch
import networkx
import numpy
import tqdm
```
or you can have everything set up by running: 
```bash
pip install -r requirements.txt
```
### Using the code:
To use the pretrained models at `checkpoint/` and evaluate the models performance run:
```bash
test.py
```

To train a model for each data set with the best configuration as in the paper, simply run:
```bash
./train.sh  
```
