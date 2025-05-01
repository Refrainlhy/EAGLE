import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

def compute_ap_score(pred_pos, pred_neg, neg_samples):
        y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu().detach()
        y_true = torch.cat([torch.ones_like(pred_pos), torch.zeros_like(pred_neg)], dim=0).cpu().detach()
        size = len(y_pred)
        ap = average_precision_score(y_true, y_pred)

        true_binary_label= torch.zeros(size)
        pred_binary_label = torch.argmax(torch.cat([pred_pos,pred_neg]),dim=1).cpu().detach()

        if neg_samples > 1:
            auc = torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0)
            auc = 1 / (auc+1)
        else:
            auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(true_binary_label, pred_binary_label)
        return ap, auc, acc
    
# GraphMixer time encoder
class TimeEncode(nn.Module):
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))
        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        #output = torch.cos(self.w(t.reshape((-1, 1))))
        output = torch.cos(self.w(t))
        return output

# apply mean to GraphMixer time encodings and calculate edge probability via inner product
# not a learning-based solution
class TimeSketch(nn.Module):
    def __init__(self, dim, ignore_zero):
        super(TimeSketch, self).__init__()
        self.ignore_zero = ignore_zero
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        # [0,1]
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        # print(np.linspace(0, 9, self.dim, dtype=np.float32)) # [0,9]
        # ? wrong time encoding function here? ** means 阶乘
        # ? all the papers use ** why??
        # print(torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32)))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))
        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        
        num_edge = t.shape[0]//3  # [batch_size, topk, 1]
        if self.ignore_zero:      
            index = (t==0).nonzero()
            #print(t.shape,'   ',index.shape)

        t=torch.unsqueeze(t,-1)
        x = torch.cos(self.w(t))  # [batch_size, topk, time_dim]

        if self.ignore_zero:
            x[index[:,0],index[:,1]]=0
        x = torch.mean(x,dim=1) # [batch_size, time_dim]
        x= torch.nn.functional.normalize(x,dim=1)
        src = x[:num_edge]  # [num_edge,time_dim]
        dst = x[num_edge:2*num_edge]
        neg = x[2*num_edge:]

        # ! inner product of the time encoding
        pos = torch.sum(src.mul(dst),dim=1).unsqueeze(-1)
        neg = torch.sum(src.mul(neg),dim=1).unsqueeze(-1)
        return pos, neg

# apply mean to GraphMixer time encodings
# not a learning-based solution
class MLPTime(nn.Module):
    def __init__(self, dim):
        super(MLPTime, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.layernorm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim,dim)    
        self.edge_predictor = EdgePredictor_per_node(dim)
        self.combiner = Combiner(10)
        self.reset_parameters()
        
    def reset_parameters(self, ):
        # [0,1]
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        # print(np.linspace(0, 9, self.dim, dtype=np.float32)) # [0,9]
        # ? wrong time encoding function here? ** means 阶乘
        # ? all the papers use ** why??
        # print(torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32)))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))
        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
        self.edge_predictor.reset_parameters()
        self.combiner.reset_parameters()
    
    #@torch.no_grad()
    def forward(self, t):
        
        # num_edge = t.shape[0]//3  # [batch_size, topk, 1]
        # ignore_zero by default
        with torch.no_grad():
            
            '''
            if self.ignore_zero:      
                index = (t==0).nonzero()
            '''
            index = (t==0).nonzero()

            t=torch.unsqueeze(t,-1)
            x = torch.cos(self.w(t))  # [batch_size, topk, time_dim]
            
            '''
            if self.ignore_zero:
                x[index[:,0],index[:,1]]=0
            '''

            x[index[:,0],index[:,1]]=0
            x = self.layernorm(x)
            x = torch.mean(x, dim=1).squeeze(dim=1)
            x = self.mlp_head(x)
            #print('in model head')
            #print(x.shape)
        
        x = self.edge_predictor(x)
        #print('predictor')
        #print(x[0].shape)
        return x

"""
Module: MLP-Mixer
"""

# [batch_size, graph_size, hidden_dim]
# self.token_forward = FeedForward(per_graph_size, token_expansion_factor, dropout, use_single_layer)
class FeedForward(nn.Module):
    """
    2-layer MLP with GeLU (fancy version of ReLU) as activation
    """
    def __init__(self, dims, expansion_factor, dropout=0, use_single_layer=False):
        super().__init__()

        self.dims = dims
        self.use_single_layer = use_single_layer
        
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        if use_single_layer:
            self.linear_0 = nn.Linear(dims, dims)
        else:
            self.linear_0 = nn.Linear(dims, int(expansion_factor * dims))
            self.linear_1 = nn.Linear(int(expansion_factor * dims), dims)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear_0.reset_parameters()
        if self.use_single_layer==False:
            self.linear_1.reset_parameters()

    def forward(self, x):
        x = self.linear_0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.use_single_layer==False:
            x = self.linear_1(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class MixerBlock(nn.Module):
    """
    out = X.T + MLP_Layernorm(X.T)     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing
    """
    def __init__(self, per_graph_size, dims, 
                 token_expansion_factor=0.5, 
                 channel_expansion_factor=4, 
                 dropout=0, 
                 module_spec=None, use_single_layer=False):
        super().__init__()
        
        if module_spec == None:
            self.module_spec = ['token', 'channel']
        else:
            self.module_spec = module_spec.split('+')


        if 'token' in self.module_spec:
            self.token_layernorm = nn.LayerNorm(dims)
            self.token_forward = FeedForward(per_graph_size, token_expansion_factor, dropout, use_single_layer)
            
        if 'channel' in self.module_spec:
            self.channel_layernorm = nn.LayerNorm(dims)
            self.channel_forward = FeedForward(dims, channel_expansion_factor, dropout, use_single_layer)
        

    def reset_parameters(self):
        if 'token' in self.module_spec:
            self.token_layernorm.reset_parameters()
            self.token_forward.reset_parameters()

        if 'channel' in self.module_spec:
            self.channel_layernorm.reset_parameters()
            self.channel_forward.reset_parameters()
        
    def token_mixer(self, x):
        # after permute [batch_size, hidden_dim, graph_size]
        x = self.token_layernorm(x).permute(0, 2, 1)

        # after permute [batch_size, graph_size, hidden_dim]
        x = self.token_forward(x).permute(0, 2, 1)
        return x
    
    def channel_mixer(self, x):
        # after permute [batch_size, graph_size, hidden_dim]
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x

    def forward(self, x):
        if 'token' in self.module_spec:
            x = x + self.token_mixer(x)

        if 'channel' in self.module_spec:
            x = x + self.channel_mixer(x)
        return x

class MLPMixer(nn.Module):
    def __init__(self, per_graph_size, time_channels,
                 num_layers=2, dropout=0.5,
                 token_expansion_factor=0.5, 
                 channel_expansion_factor=4, 
                 module_spec=None, use_single_layer=False,
                 device = 'cpu'
                ):
        super().__init__()

        self.per_graph_size = per_graph_size # max_edge
        self.num_layers = num_layers
        self.time_encoder = TimeEncode(time_channels)
        self.feat_encoder = nn.Linear(time_channels, time_channels) 
        self.layernorm = nn.LayerNorm(time_channels)
        self.mlp_head = nn.Linear(time_channels,time_channels)
        self.mixer_blocks = torch.nn.ModuleList()
        self.time_channels = time_channels
        self.device = device

        for ell in range(num_layers):
            # module_spec is None by default
            if module_spec is None:
                self.mixer_blocks.append(
                    MixerBlock(per_graph_size, time_channels, 
                               token_expansion_factor, 
                               channel_expansion_factor, 
                               dropout, module_spec=None, 
                               use_single_layer=use_single_layer).to(self.device)
                    )
            else:
                self.mixer_blocks.append(
                    MixerBlock(per_graph_size, time_channels, 
                               token_expansion_factor, 
                               channel_expansion_factor, 
                               dropout, module_spec=module_spec[ell], 
                               use_single_layer=use_single_layer).to(self.device)
                    )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mixer_blocks:
            layer.reset_parameters()
        self.time_encoder.reset_parameters()
        self.feat_encoder.reset_parameters()
        self.layernorm.reset_parameters()
        self.mlp_head.reset_parameters()


    def forward(self, delta_times, inds, batch_size):
        # input: no [0...0] gaps in delta_times
        time_encodings = self.time_encoder(delta_times) 
        time_encodings = self.feat_encoder(time_encodings)

        x = torch.zeros((batch_size * self.per_graph_size, self.time_channels)).to(self.device)

        x[inds] = time_encodings # add 0 gaps if node N's num_nei < topk, add to max_edge

        x = torch.split(x, self.per_graph_size) # -> tuples

        x = torch.stack(x) # [batch_size=num(3types nodes), graph_size=max_edge, hidden_dim]

        for i in range(self.num_layers):
            x = self.mixer_blocks[i](x)
            
        x = self.layernorm(x)
        x = torch.mean(x, dim=1)
        x = self.mlp_head(x)
        return x

class Combiner(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.linear1 = torch.nn.Linear(2, dim)
        self.linear2 = torch.nn.Linear(dim, 1)
        self.reset_parameters()
        
    def reset_parameters(self,):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, h):
        num_edge = h.shape[0] // 2
        x = self.linear1(h)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x[:num_edge], x[num_edge:]

'''
class Combiner(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.linear1 = torch.nn.Linear(2, 1)
        self.reset_parameters()
        
    def reset_parameters(self,):
        self.linear1.reset_parameters()

    def forward(self, h):
        num_edge = h.shape[0] // 2
        x = self.linear1(h)
        x = torch.nn.functional.relu(x)
        #x = self.linear2(x)
        return x[:num_edge], x[num_edge:]
'''    

class EdgePredictor_per_node(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.src_fc = torch.nn.Linear(dim, 100)
        self.dst_fc = torch.nn.Linear(dim, 100)
        self.out_fc = torch.nn.Linear(100, 1)
        self.reset_parameters()
        
    def reset_parameters(self,):
        self.src_fc.reset_parameters()
        self.dst_fc.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, h, num_neg=1):
        num_edge = h.shape[0] // (num_neg + 2)
        
        # encode pos embeds
        h_src = self.src_fc(h[:num_edge])

        # encode pos and neg dest embeds
        h_pos_dst = self.dst_fc(h[num_edge:2*num_edge])
        # h_neg_dst = self.dst_fc(h[2 * num_edge:])

        h_neg_dst = self.dst_fc(h[2*num_edge:(num_neg+2)*num_edge]) # [num_neg*num_edge, embed_dim]
        h_src_repeated = h_src.repeat(num_neg, 1) # [num_neg*num_edge, embed_dim]

        # encode summed source and dest embeds
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        # h_neg_edge = torch.nn.functional.relu(h_src + h_neg_dst)
        h_neg_edge = torch.nn.functional.relu(h_src_repeated + h_neg_dst)

        # [batch_size, 1]
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)
    

'''
mixer_configs = {
    'per_graph_size'  : args.topk, 
    'time_channels'   : 100, 
    'num_layers'      : args.num_layers,
    'use_single_layer' : False
}
'''

class Mixer_per_node(nn.Module):
    """
    Wrapper of MLPMixer and EdgePredictor
    """
    def __init__(self, mlp_mixer_configs, edge_predictor_configs):
        super(Mixer_per_node, self).__init__()

        self.dim = edge_predictor_configs['dim']
        self.edge_predictor = EdgePredictor_per_node(**edge_predictor_configs)
        self.base_model = MLPMixer(**mlp_mixer_configs)
        self.combiner = Combiner(10)
        self.reset_parameters()            

    def reset_parameters(self):
        self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()
        self.combiner.reset_parameters()

    '''
    # no padding in this version
    def forward(self, model_inputs):  
        # [batch size, topk, 1]
        model_inputs = model_inputs.unsqueeze(-1)
        x = self.base_model(model_inputs)
        pred_pos, pred_neg = self.edge_predictor(x)
        return pred_pos, pred_neg
    '''

    # add padding in this version
    def forward(self, delta_times, all_inds, batch_size, num_neg=1):  
        x = self.base_model(delta_times, all_inds, batch_size) # [600,100]
        pred_pos, pred_neg = self.edge_predictor(x, num_neg)
        return pred_pos, pred_neg

"""
Module: Link Predictor
"""
class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()
    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=2)
    h = self.act(self.fc1(x))
    return self.fc2(h).mean(dim=0)

class EdgePredictor(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.src_fc = torch.nn.Linear(dim, 100)
        self.dst_fc = torch.nn.Linear(dim, 100)
        self.out_fc = torch.nn.Linear(100, 1)
        self.norm = torch.nn.LayerNorm(100)
        self.reset_parameters()
        
    def reset_parameters(self,):
        self.src_fc.reset_parameters()
        self.dst_fc.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, h):

        num_edge = h.shape[0] // 3
        h_src = self.src_fc(h[:num_edge])
        dst = self.dst_fc(h[num_edge:])
        h_pos_dst = dst[:num_edge]
        h_neg_dst = dst[num_edge:]

        # print('src: ',h_src.shape) # [200,100]

        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_src + h_neg_dst)

        # print('h_pos_edge: ',h_pos_edge.shape) # [200,100]
  
        # * normalization indeed helps here
        #h_edge = torch.concat((h_pos_edge,h_neg_edge),dim=0)
        #h_edge = self.norm(h_edge)
        #h_pos_edge = h_edge[:num_edge]
        #h_neg_edge = h_edge[num_edge:]
             
        h_pos_edge = self.norm(h_pos_edge)
        h_neg_edge = self.norm(h_neg_edge)

        #print('h_pos_edge: ',h_pos_edge.shape) # [200,100]
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)

    '''
    def forward(self, h):

        num_edge = h.shape[1] // 3
        
        #print(f'num_edge {num_edge}')
        # encode pos embeds
        h_src = self.src_fc(h[:,:num_edge])
        dst = self.dst_fc(h[:,num_edge:])
        h_pos_dst = dst[:,:num_edge]
        h_neg_dst = dst[:,num_edge:]

        # encode pos and neg dest embeds
        #h_pos_dst = self.dst_fc(h[:,num_edge:2 * num_edge])
        #h_neg_dst = self.dst_fc(h[:,2 * num_edge:])
        

        # encode summed source and dest embeds
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_src + h_neg_dst)
        
        # * normalization indeed helps here
        # [tppr_size, batch_size, k]
        h_pos_edge = self.norm(h_pos_edge)
        h_neg_edge = self.norm(h_neg_edge)

        # let's try some 
        # let's add try combiner here..
        # add the ReLU layer here
        # * no sigmoid function applied here
        # [tppr_size, batch_size, 1]  => [batch_size, 1]
        return self.out_fc(h_pos_edge).mean(dim=0), self.out_fc(h_neg_edge).mean(dim=0)
    '''

"""
Module: Node classifier
"""
class NodeClassificationModel(nn.Module):

    def __init__(self, dim_in, dim_hid, num_class):
        super(NodeClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, dim_hid)
        self.fc2 = torch.nn.Linear(dim_hid, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x
