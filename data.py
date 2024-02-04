import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import torch
import torch_geometric.data as gdata
import networkx as nx
from sklearn import preprocessing
import imblearn
from imblearn.over_sampling import SMOTE
import random
label_encoder = preprocessing.LabelEncoder() 
from imblearn.over_sampling import SMOTE

smote = SMOTE()


def data_au(X,y):
    X_au,y_au=smote.fit_resample(X, y)
    return X_au,y_au

def get_ratio(df):
    ratio=[]
    for i in range(len(df)):
        list_=[]
        if df["gender_p"][i]==0:
            list_.append(2)
        else: list_.append(1)
        if df["l"][i]==0 or df["l"][i]==2:
            list_.append(2)
        else: list_.append(1)
        if df["location"][i]==1:
            list_.append(2)
        else: list_.append(1)
        list_.append(df["g"][i])
        
        if (df["l"][i]==0 or df["l"][i]==2) and df["location"][i]==1:
            list_.append(2)
        elif (df["l"][i]==1 or df["l"][i]==3) and df["location"][i]==2:
            list_.append(-1)
        
        ratio.append(np.sum(list_))
    return ratio


def get_distance(df):
    Distance1=[]
    Distance2=[]
    for i in range(len(df)):
        Distance1.append(df["g"][i]*13.4 )#if data_plus["location"][i]==1 #else data_plus["g"][i]*10)
        Distance2.append(4.55 if (df['l'][i]==0 or df['l'][i]==0) else 5.2 )
    return Distance1,Distance2
    
def get_data(path):
    df=pd.read_excel(path)
    df=df.drop(["Row","outcome", 'conditionNo', 'runNo', 'trialNo'], axis=1)
    
    df['gender_D']= label_encoder.fit_transform(df['gender_D']) 
    df['gender_p']= label_encoder.fit_transform(df['gender_p']) 
    df['SVO_D_A']= label_encoder.fit_transform(df['SVO_D_A']) 
    df['SVO_P_A']= label_encoder.fit_transform(df['SVO_P_A']) 
    df['l']= label_encoder.fit_transform(df['l'])
    df['delta_sss_cat']= label_encoder.fit_transform(df['delta_sss_cat'])
    
    df["svo/sss_d"]=((df["SVO_d"]/df["SSS_d"]))
    df["svo/sss_p"]=((df["SVO_p"]/df["SSS_p"]))
    df["svo_sss_d"]=(df["SSS_d"]-df["SVO_d"])
    df["svo_sss_p"]=(df["SSS_p"]-df["SVO_p"])
    df["distance_d"],df["distance_p"]=get_distance(df)
    df["delta_dis"]=df["distance_d"]-df["distance_p"]
    df["ratio"]=get_ratio(df)

    
    
    
    
    data_base=df[["subjectNo",'t', 'g', 'gender_p', 'age_p', 'location', 'cross', 'delta_svo','delta_sss', ]].copy()
    
    data=df[['t', 'g', 'l','SVO_D_A', 'SVO_P_A', 'gender_D', 'gender_p', 'age_p', 'age_d', 'SVO_d',
            'SVO_p', 'location', 'cross', 'SSS_p', 'SSS_d', 'delta_svo',
            'delta_sss', 'delta_sss_cat']].copy()
    
    data_plus=df[['t', 'g', 'l','SVO_D_A', 'SVO_P_A', 'gender_D', 'gender_p', 'age_p', 'age_d', 'SVO_d',
            'SVO_p', 'location', 'cross', 'SSS_p', 'SSS_d', 'delta_svo',
            'delta_sss', 'delta_sss_cat','distance_d',"distance_p", 'svo/sss_d',"svo_sss_p","delta_dis",'svo_sss_d',"ratio"]].copy()
    
    data_realistic=df[['gender_p', 'age_p','SVO_p', 'location', 'cross', 'SSS_p','distance_d',
                        'svo/sss_d', 'svo_sss_d',"ratio"]].copy()
    
    
    return data_base, data, data_plus, data_realistic


def GNN_data(data):
        Y=data["cross"]
        X=data.drop("cross", axis=1)
        XX, YY = data_au(X,Y)
        graph_dataset = []
        for idx, data in XX.iterrows():
            G = nx.Graph()
            
            driver_features = data[["distance_d",'svo/sss_d',"svo_sss_d",'SVO_D_A', 'gender_D', 'age_d', 'SVO_d', 'SSS_d']]
            pedestrian_features =data[["distance_p",'svo/sss_d', "svo_sss_p" ,'SVO_P_A', 'gender_p', 'age_p', 'SVO_p', 'SSS_p']]
            vehicle_features = data[["ratio","g","l","delta_dis",'location','t','delta_svo', 'delta_sss']]

            G.add_node(0, type='driver', features=driver_features.to_dict())
            G.add_node(1, type='pedestrian', features=pedestrian_features.to_dict())
            G.add_node(2, type='vehicle', features=vehicle_features.to_dict())

            if data["location"] == 1:
                G.add_edge(1, 2)
            else:
                G.add_edge(0, 2)
                
            if ((data["distance_d"] < 30 ) and (data["l"] == "0" or data["l"] == "1" or data["l"] == "2")):
                G.add_edge(0, 1)
            # elif (data["Distance"] < 40 and (data["l"] == "2" or data["l"] == "3")):
            #     G.add_edge(0, 1)
            elif ((data["distance_d"] >= 30) and (data["l"] == "0")):
                G.add_edge(0, 1)
            # G.name(data["cross"])
                
            graph_dataset.append(G)

    # Convert NetworkX graph dataset to PyTorch Geometric Data objects
        data_list = []
        for i, graph in enumerate(graph_dataset):

            node_features = []
            for node in graph.nodes():
                features = graph.nodes[node]['features']
                node_features.append([v for k, v in features.items()])
            edge_index = torch.tensor(list(graph.edges)).t().contiguous()

            x = torch.tensor(node_features, dtype=torch.float)  # Node features as a tensor
            # x = torch.ones(graph.number_of_nodes(), 5) 
            edge_index = edge_index
            y = torch.tensor(YY[i])  # Assuming a single label for each graph (modify as needed)

            data = gdata.Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
            random.shuffle(data_list)

            train_dataset = data_list[:1378]
            test_dataset = data_list[1378:]
            
        return train_dataset,test_dataset
    