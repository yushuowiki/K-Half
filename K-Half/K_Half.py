import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import SpGraphAttentionLayer


class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention
        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)


        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nheads * nhid, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )

    def forward(self, entity_embeddings, relation_embed,
                edge_list, edge_type, edge_embed):
        x = entity_embeddings
        x = torch.cat([att(x, edge_list, edge_embed)
                       for att in self.attentions], dim=1)
        x = self.dropout_layer(x)

        out_relation_1 = relation_embed.mm(self.W)

        edge_embed = out_relation_1[edge_type]

        x = F.elu(self.out_att(x, edge_list, edge_embed))
        return x, out_relation_1


class K_Half(nn.Module):
    def __init__(self, initial_entity_emb, entity_out_dim, h_dim, num_ents, nheads_GAT,
                 drop_GAT=0.3, alpha=0.2, relation_dict=None):
        super(K_Half, self).__init__()
        self.h_dim = h_dim
        self.num_ents = num_ents
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]
        self.drop_GAT = drop_GAT
        self.alpha = alpha
        self.relation_dict = relation_dict

        self.weight_t2 = nn.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.Parameter(torch.randn(1, h_dim))


        self.entity_embeddings = nn.Parameter(initial_entity_emb)


        self.sparse_gat_1 = SpGAT(self.num_ents, self.entity_in_dim, self.entity_out_dim_1,
                                  self.entity_out_dim_1, self.drop_GAT, self.alpha, self.nheads_GAT_1)

        self.W_entities = nn.Parameter(torch.zeros(size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

    def get_time_embedding(self, t2):
        time_embedding = torch.cos(self.weight_t2 * t2 + self.bias_t2).repeat(self.num_ents, 1)
        return time_embedding

    def forward(self, Corpus_, batch_inputs, edge_list, edge_type):


        edge_embed = torch.stack([self.relation_dict[rel_id.item()] for rel_id in edge_type])


        out_entity_1, _ = self.sparse_gat_1(self.entity_embeddings, edge_embed, edge_list, edge_type, edge_embed)


        self.entity_embeddings.data = F.normalize(self.entity_embeddings.data, p=2, dim=1).detach()

        his_temp_embs = []
        for i, fact in enumerate(batch_inputs):
            t2 = fact[3]
            h_t = self.get_time_embedding(t2)

            his_temp_embs.append(h_t[batch_inputs[:, 0].long()])


        mask_indices = torch.unique(batch_inputs[:, 2]).to(self.entity_embeddings.device).long()
        mask = torch.zeros(self.entity_embeddings.shape[0], device=self.entity_embeddings.device)
        mask[mask_indices] = 1.0

        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        out_entity_1 = entities_upgraded + mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)

        return out_entity_1, his_temp_embs