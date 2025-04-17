import torch
from torch_geometric.nn import MessagePassing, global_mean_pool
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential

class SmMessagePassing(MessagePassing):
    def __init__(self, hidden_dim=64):
        super().__init__(aggr='add') # select aggregation method ('add', 'mean', 'max')
        self.message_mlp = Sequential(Linear(2 * hidden_dim, hidden_dim), ReLU()) # About edge features
        self.update_mlp = Sequential(Linear(hidden_dim, hidden_dim), ReLU()) # About node features

    def forward(self, x, edge_index, edge_attr):
        updated_node_features = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr) # In propagate(), message() -> aggregate() -> update()
        return updated_node_features 
        
    def message(self, x_j, edge_attr):
        # x_j: node features of the source nodes (neighbors)
        # edge_attr: edge features
        # Concatenate node features and edge features
        #messages = self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1)) # 이건 왜 안 되는 걸까?
        messages = self.message_mlp(torch.cat([x_j, edge_attr], dim=-1)) # [num_edges, 2 * hidden_dim] -> [num_edges, hidden_dim]
        return messages

    def update(self, aggr_out):
        #updated_node_features = self.update_mlp(torch.cat([x_i, aggr_out], dim=-1)) 
        updated_node_features = self.update_mlp(torch.cat([aggr_out], dim=-1)) 
        return updated_node_features
    

class SmMPNN(nn.Module):
    def __init__(self, node_dim=1, edge_dim=1, hidden_dim=64):
        super().__init__()
        self.node_encoder = Linear(node_dim, hidden_dim) # convert node feature to hidden_dim
        self.edge_encoder = Linear(edge_dim, hidden_dim) # convert edge feature to hidden_dim
        self.message_passing_layer = SmMessagePassing(hidden_dim) # Message passing layer
        self.output_layer = Linear(hidden_dim, 1) # One of readout layers # convert hidden_dim to output dimension (1 for regression, num_classes for classification)

    def forward(self, data):
        x, edge_index, edge_attr, batch= data.x, data.edge_index, data.edge_attr, data.batch

        # node/edge feature encoding
        node_features = self.node_encoder(x) 
        edge_features = self.edge_encoder(edge_attr) 

        # Message passing & Update
        updated_node_features = self.message_passing_layer(x=node_features, edge_index=edge_index, edge_attr=edge_features)

        # Aggregate node features to graph-level features # Readout Phase
        avg_features = global_mean_pool(updated_node_features, batch) # Average the same features of nodes belonging to the same graph → Vector of the average of each feature
        
        output = self.output_layer(avg_features).squeeze(-1) # Convert to 1D vector 
        return output
