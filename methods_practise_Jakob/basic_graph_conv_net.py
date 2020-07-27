import numpy as np
from networkx import karate_club_graph, to_numpy_matrix
import matplotlib.pyplot as plt


# Defining and building GCN


def deg_mat(A):
    D = np.array(np.sum(A, axis=0))[0]
    D = np.matrix(np.diag(D))
    return D


zkc = karate_club_graph()
order = sorted(list(zkc.nodes()))
A = to_numpy_matrix(zkc, nodelist=order)
I = np.eye(zkc.number_of_nodes())

A_hat = A + I
D_hat = deg_mat(A_hat)

W_1 = np.random.normal(
    loc=0, scale=1, size=(zkc.number_of_nodes(), 4))
W_2 = np.random.normal(
    loc=0, size=(W_1.shape[1], 2))

def ReLU(x):
    return np.maximum(x, 0.)

def gcn_layer(A, D, X, W):
    return ReLU(np.diag(np.diag(D)**-.5) * A * np.diag(np.diag(D)**-.5) * X * W)

H_1 = gcn_layer(A_hat, D_hat, I, W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)

output = H_2

feature_reps = {node: np.array(output)[node] for node in zkc.nodes()}

out_x = np.array(output[:,0]).flatten().reshape(-1)
out_y = np.array(output[:,1]).flatten().reshape(-1)


#Loading data

from collections import namedtuple
from networkx import read_edgelist, set_node_attributes, to_numpy_matrix, degree_centrality, betweenness_centrality, shortest_path_length
from pandas import read_csv, Series

DataSet = namedtuple(
    'DataSet',
    field_names=['X_train', 'y_train', 'X_test', 'y_test', 'network']
)

def load_karate_club():
    network = read_edgelist(
        './karate.edgelist',
        nodetype=int)

    attributes = read_csv(
        './karate.attributes.csv',
        index_col=['node'])

    for attribute in attributes.columns.values:
        set_node_attributes(
            network,
            values=Series(
                attributes[attribute],
                index=attributes.index).to_dict(),
            name=attribute
        )

    X_train, y_train = map(np.array, zip(*[
        ([node], data['role'] == 'Administrator')
        for node, data in network.nodes(data=True)
        if data['role'] in {'Administrator', 'Instructor'}
    ]))
    X_test, y_test = map(np.array, zip(*[
        ([node], data['community'] == 'Administrator')
        for node, data in network.nodes(data=True)
        if data['role'] == 'Member'
    ]))
    
    return DataSet(
        X_train, y_train,
        X_test, y_test,
        network)


zkc = load_karate_club()

A = to_numpy_matrix(zkc.network)
print(A)
A = np.ndarray((A.shape[0],A.shape[1]),buffer=A)

X_train = zkc.X_train.flatten()
y_train = zkc.y_train
X_test = zkc.X_test.flatten()
y_test = zkc.y_test


#Defining model
import torch
from torch.nn import Module, Sequential, ReLU, Linear, Sigmoid, Tanh


class Layer(Module):
    
    def __init__(self, A, dim_in, dim_out, activation="relu", **kwargs):
        
        super().__init__()
        
        I = np.eye(*A.shape)
        A_hat = A.copy() + I

        D = np.sum(A_hat, axis=0)
        D_inv = D**-.5
        D_diag = np.diag(D_inv)
        
        A_hat = D_diag * A_hat * D_diag
        A_hat = torch.from_numpy(np.matrix.astype(A_hat,dtype=np.float32))
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self.A_hat = A_hat
        self.W = Linear(dim_in,dim_out)
        if activation == 'relu':
            self.activation = ReLU()
        elif activation == 'tanh':
            self.activation = Tanh()
        else:
            self.activation = lambda X: X
        
    def forward(self, X):
        return self.activation(self.W(torch.mm(self.A_hat, X)))
        
    
        
def define_model(A, X):
    model = Sequential(
        Layer(A, X.shape[1], 4, 'tanh'),
        Layer(A, 4, 2, 'tanh'),
        Linear(2, 1, ' '),
        Sigmoid()
    )
    
    return model


model = define_model(A, np.eye(*A.shape))
X_1 = np.eye(*A.shape,dtype=np.float32)
input_tensor = torch.from_numpy(X_1)

print(X_train)
print(model(input_tensor))
#Training GCN

def train(model, X, X_train, y_train, epochs, print_progress = True):
    
    cross_entropy = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=1)
    
    feature_reps = [model(X).detach().numpy()]
    
    for e in range(1, epochs + 1):
        cum_loss = 0
        cum_preds = []
        
        for i, x in enumerate(X_train):
            optimizer.zero_grad()
            y = np.array(y_train)[i]
            
            preds = model(X)[x]
            loss = cross_entropy(preds, torch.tensor(float(y)))
            
            loss.backward()
            optimizer.step()
            
            cum_loss += loss
            cum_preds.append(preds.detach().numpy())
            
        feature_reps.append(model(X).detach().numpy())
        if (e % (epochs//10)) == 0 and print_progress == True:
            print(f"Epoch {e}/{epochs} -- Loss: {cum_loss.detach().numpy(): .4f}")
            print([cum_preds[0][0],cum_preds[1][0]])
            
    return feature_reps, model


feature_representations, model = train(model, input_tensor, X_train, y_train, 1000)

print(X_train)
print(y_train)
print(y_test)

print(feature_representations[-1])

from sklearn.metrics import classification_report

def predict(model, X, nodes):
    preds = model(X).detach().numpy()[nodes]
    return np.where(preds >= 0.5, 1, 0)

preds = predict(model,input_tensor,X_test)
print(classification_report(y_test,preds))


X_2 = np.zeros((A.shape[0], 2),dtype=np.float32)
node_distance_instructor = shortest_path_length(zkc.network, target=33)
node_distance_administrator = shortest_path_length(zkc.network, target=0)

for node in zkc.network.nodes():
    X_2[node][0] = node_distance_administrator[node]
    X_2[node][1] = node_distance_instructor[node]
    
print(X_1.shape)
print(X_2.shape)
X_2 = np.concatenate((X_1, X_2),axis=1)
print(X_2)
model_2 = define_model(A, X_2)
input_tensor_2 = torch.from_numpy(X_2)
model_2(input_tensor_2)

num_tries = 10

def init_weights(m):
    if type(m) == Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

for i in range(num_tries):
    model_2.apply(init_weights)
    feature_representations, model = train(model_2, input_tensor_2, X_train, y_train, epochs=500, print_progress=False)
    y_pred_2 = predict(model, input_tensor_2, X_test)
    print(classification_report(y_test, y_pred_2))