import torch
import torch.nn as nn
class SocialLSTM(nn.Module):
    def __init__(self, pos_embedding_size = 64, grid_embedding_size = 64, 
                 hidden_dim = 128,grid_size=4, dropout=0.0, max_agents=20):
        super().__init__()
        self.pos_embedding_size = pos_embedding_size
        self.grid_embedding_size = grid_embedding_size
        self.hidden_dim = hidden_dim
        self.grid_size =grid_size
        self.dropout =dropout
        self.max_agents= max_agents

        self.position_embedding = nn.Linear(2, self.pos_embedding_size)
        self.social_grid_embedding = nn.Linear(self.grid_size*self.grid_size*self.hidden_dim,self.grid_embedding_size)
        self.lstm_cell = nn.LSTMCell(self.pos_embedding_size+self.grid_embedding_size,self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, 2)
        self.relu_layer = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def compute_social_tensor(self, grid_tensor, hidden_states):
        '''
        grid_tensor for a particular sequence, [max_no_agents,grid_size*grid_size,max_no_agents ]
        hidden_state = [max_no_agents,hidden_dim]
        return social_tensor : [max_no_agents,grid_size*grid_size*hidden_dim]

        '''
        num_active = grid_tensor.size(0)
        hidden_dim = hidden_states.size(1)
        social_tensor = torch.zeros(num_active,self.grid_size*self.grid_size*hidden_dim)
        # print(grid_tensor[0].shape)
        # print(hidden_states.shape)
        for i in range(num_active):
            social_tensor[i] = torch.mm(grid_tensor[i],hidden_states).view(-1) # view for flattening the tensor
        
        return social_tensor

    def forward(self, input_tensor, grid_tensor, hidden_states, cell_states,mask_tensor):
        sequence_length , max_no_agents, no_coordinates = input_tensor.shape
        outputs = []
        
        for t in range(sequence_length):
            agent_positions = input_tensor[t]
            agent_mask = mask_tensor[t]

            active_agents = agent_mask.nonzero(as_tuple = True)[0]

        # handling sequnce with zero agents
            if len(active_agents) == 0:
                outputs.append(torch.zeros_like(agent_positions))
                continue
            pos_embedding_tensor = self.relu_layer(self.position_embedding(agent_positions[active_agents]))
            pos_embedding_tensor_dropped = self.dropout_layer(pos_embedding_tensor)
            hidden_active = hidden_states[active_agents]
            #grid_t = grid_tensor[t][active_agents]
            grid_t = grid_tensor[t][active_agents][:, :, active_agents] 
            social_tensor  = self.compute_social_tensor(grid_t,hidden_active)
            
            embedded_social_tensor = self.relu_layer(self.social_grid_embedding(social_tensor))
            embedded_social_tensor_dropped = self.dropout_layer(embedded_social_tensor)
            concat_input = torch.cat([pos_embedding_tensor_dropped,embedded_social_tensor_dropped],dim=1)
            hidden, cell = self.lstm_cell(concat_input,(hidden_states[active_agents],cell_states[active_agents]))
            hidden_states[active_agents] = hidden
            cell_states[active_agents] = cell
            pred = self.output_layer(hidden)
            step_output = torch.zeros(max_no_agents, 2, device=input_tensor.device)
            step_output[active_agents] = pred
            outputs.append(step_output)
            
        return torch.stack(outputs), hidden_states, cell_states



