from torch import nn
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import knn_graph


class GCN_multi_feature(nn.Module):
    def __init__(self, mv_feature,residual_feature,partition_feature,dcavg_feature,qp_feature):
        super().__init__()

        self.mvgcn1 = GCNConv(mv_feature,mv_feature)
        self.mvgcn2 = GCNConv(mv_feature,mv_feature)
        

        self.residualgcn1 = GCNConv(residual_feature,residual_feature)
        self.residualgcn2 = GCNConv(residual_feature,residual_feature)
        

        self.partitiongcn1 = GCNConv(partition_feature,partition_feature)
        self.partitiongcn2 = GCNConv(partition_feature,partition_feature)
        

        self.dcavggcn1 = GCNConv(dcavg_feature,dcavg_feature)
        self.dcavggcn2 = GCNConv(dcavg_feature,dcavg_feature)
        

        self.qpgcn1 = GCNConv(qp_feature,qp_feature)
        self.qpgcn2 = GCNConv(qp_feature,qp_feature)


        self.a = nn.Linear(3328,3328)
        self.b = nn.Linear(3328,1024)

        self.gcn4 = GCNConv(1024,512)
        self.gcn5 = GCNConv(512,256)
        self.fc1 = nn.Linear(256,128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mvData,residualData,partitionData,dcavgData,qpData):
        mv, mv_edge_index = mvData.x, mvData.edge_index
        residual, residual_edge_index = residualData.x, residualData.edge_index
        partition, partition_edge_index = partitionData.x, partitionData.edge_index
        dcavg, dcavg_edge_index = dcavgData.x, dcavgData.edge_index
        qp, qp_edge_index = qpData.x, qpData.edge_index

        mv_out = self.mvgcn1(mv,mv_edge_index)
        mv_out = mv_out+mv
        mv_edge_index=knn_graph(mv_out, k=30, cosine=True)
        mv_out = self.mvgcn2(mv_out,mv_edge_index)
        
        
        residual_out = self.residualgcn1(residual,residual_edge_index)
        residual_out = residual_out+residual
        residual_edge_index=knn_graph(residual_out, k=30, cosine=True)
        residual_out = self.residualgcn2(residual_out,residual_edge_index)
        
        partition_out = self.partitiongcn1(partition,partition_edge_index)
        partition_out = partition_out+partition
        partition_edge_index=knn_graph(partition_out, k=30, cosine=True)
        partition_out = self.partitiongcn2(partition_out,partition_edge_index)
        

        dcavg_out = self.dcavggcn1(dcavg,dcavg_edge_index)
        dcavg_out = dcavg_out+dcavg
        dcavg_edge_index=knn_graph(dcavg_out, k=30, cosine=True)
        dcavg_out = self.dcavggcn2(dcavg_out,dcavg_edge_index)
        

        qp_out = self.qpgcn1(qp,qp_edge_index)
        qp_out = qp_out+qp
        qp_edge_index=knn_graph(qp_out, k=30, cosine=True)
        qp_out = self.qpgcn2(qp_out,qp_edge_index)


        all = torch.cat((mv_out,residual_out,partition_out,dcavg_out,qp_out),dim=1)
        all = self.a(all)
        all = self.relu(all)
        all = self.b(all)
        all = self.relu(all)

        out_edge_index = knn_graph(all, k=30, cosine=True)
        out = self.gcn4(all,out_edge_index)
        out_edge_index = knn_graph(out, k=30, cosine=True)
        out = self.gcn5(out,out_edge_index)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(1,-1)
        return out