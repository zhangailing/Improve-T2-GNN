import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch_geometric.nn import SAGEConv

class Teacher_F(nn.Module):
    def __init__(self, num_nodes, in_size, hidden_size, out_size, num_layers, dropout):
        # in_size是输入特征的维度
        super(Teacher_F, self).__init__()
        if num_layers == 1:
            hidden_size = out_size
        # 图中节点的重要特征矩阵
        self.imp_feat = nn.Parameter(torch.empty(size=(num_nodes, in_size)))
        # 用一个正态分布初始化，激活函数为ReLU
        nn.init.xavier_normal_(self.imp_feat.data, gain=1.414)

        self.fm1 = nn.Linear(in_size, hidden_size, bias=True)
        self.fm2 = nn.Linear(hidden_size, out_size, bias=True)
        # 正则化技术防止过平滑
        self.dropout = dropout
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 均匀分布初始化
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, feature):
        feature = torch.where(torch.isnan(feature), self.imp_feat, feature)
        middle_representations = []

        h = self.fm1(feature)
        middle_representations.append(h)
        # 防止过平滑
        h = F.dropout(h, self.dropout, training=self.training)
        # 引入非线形，可学习更复杂的模式
        h = F.relu(h)
        h = self.fm2(h)
        middle_representations.append(h)

        return h, middle_representations

class Teacher_S(nn.Module):
    def __init__(self, num_nodes, in_size, hidden_size, out_size,dropout, device, num_layers):
        super(Teacher_S, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_size, hidden_size))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_size, hidden_size))
        self.layers.append(SAGEConv(hidden_size, out_size))

        self.residual_layers = nn.ModuleList()
        self.residual_layers.append(nn.Linear(in_size, hidden_size))
        for _ in range(num_layers - 2):
            self.residual_layers.append(nn.Linear(hidden_size, hidden_size))
        self.residual_layers.append(nn.Linear(hidden_size, out_size))

        self.dropout = dropout
        self.linear = nn.Linear(num_nodes, in_size, bias=True)
        self.pe_feat = torch.FloatTensor(torch.eye(num_nodes)).to(device)

    def forward(self, adj):
        middle_representations = []
        pe = self.linear(self.pe_feat)
        h = pe
        for i, (layer, residual_layer) in enumerate(zip(self.layers, self.residual_layers)):
            h_prev = h
            h = layer(h, adj)
            if i != len(self.layers) - 1:
                h = F.leaky_relu(h)
                h = F.dropout(h, self.dropout, training=self.training)
            h += residual_layer(h_prev)  # Adjust the size of h_prev before adding
            middle_representations.append(h)

        return h, middle_representations

#Student
class  GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nhid_feat, nhid_stru, tau=0.5):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        # 温度参数
        self.tau = tau
        self.feat2stu = torch.nn.Linear(nhid_feat, nhid)
        self.stru2stu = torch.nn.Linear(nhid_stru, nhid)

    def forward(self, adj, x):
        #imp[0],用0替换特征矩阵中NaN的部分
        imp = torch.zeros([x.shape[0], x.shape[1]])
        imp = imp.to(x.device)
        x = torch.where(torch.isnan(x), imp, x)
        middle_representations = []
        h = self.gc1(x, adj)
        middle_representations.append(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.gc2(h, adj)
        middle_representations.append(h)
        return h, middle_representations

    # 计算余弦相似度
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
 
    # 让相似的更相似，不相似的更不相似
    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    # z1是学生的输出，z2是特征教师的输出，z3是结构教师的输出
    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor,mean: bool = True):
        R_stu_1 = z1[0]
        # z2[0]&z3[0]是特征教师和结构教师隐藏层的输出，和z1[0]学生隐藏层的输出对比的时候，需要转换成相同维度,R_fea_1 torch.Size([节点数, 64]),R_str_1 torch.Size([节点数, 64])
        R_fea_1 = self.feat2stu(z2[0])
        R_str_1 = self.stru2stu(z3[0])
        fea_stu_1 = self.semi_loss(R_stu_1, R_fea_1)
        str_stu_1 = self.semi_loss(R_stu_1, R_str_1)
        # mean true 则 mean ，mean false 则 sum
        fea_stu_1 = fea_stu_1.mean() if mean else fea_stu_1.sum()
        str_stu_1 = str_stu_1.mean() if mean else str_stu_1.sum()
        # 学生、教师的输出都是类别的维度
        R_stu_2 = z1[1]
        R_fea_2 = z2[1]
        R_str_2 = z3[1]
        fea_stu_2 = self.semi_loss(R_stu_2, R_fea_2)
        str_stu_2 = self.semi_loss(R_stu_2, R_str_2)
        fea_stu_2 = fea_stu_2.mean() if mean else fea_stu_2.sum()
        str_stu_2 = str_stu_2.mean() if mean else str_stu_2.sum()
        loss_mid_fea = fea_stu_1 + fea_stu_2
        loss_mid_str = str_stu_1 + str_stu_2

        return loss_mid_fea, loss_mid_str

