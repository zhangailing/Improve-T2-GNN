import os
import random
import torch
import dgl
import scipy.sparse as sp
import numpy as np

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    from tensorflow.keras.utils import to_categorical
except ImportError as e:
    print("Import failed:", e)

# 设置随机数生成器种子，确保使得输出可预测和重现
def setup_seed(seed, cuda):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda is True:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_sparse(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# 分类任务的常用指标，acc=正确预测数/总的预测数，output是模型预测未归一化的原始输出[每个节点属于5个类别的概率]，labels是真实标签,打印多次labels，是因为一个epoch中可能包含多个batch的数据
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# 随机掩蔽特定比例的特征，先用0&1生成个同型张量，再比，然后将mask变成true&false
def feature_mask(features, missing_rate):
    mask = torch.rand(size=features.size())
    mask = mask <= missing_rate
    return mask

# 将mask为true的地方用nan替换
def apply_feature_mask(features, mask):
    features[mask] = float('nan')

# 随机删除一些边，enforce_connected结果图是否连通，
def edge_delete(prob_del, adj, enforce_connected=False):
    rnd = np.random.RandomState(1234)
    adj= adj.toarray()
    del_adj = np.array(adj, dtype=np.float32)
    smpl = rnd.choice([0., 1.], p=[prob_del, 1. - prob_del], size=adj.shape) * np.triu(np.ones_like(adj), 1)
    smpl += smpl.transpose()
    # 构造对称的删除矩阵，删除i到j的同时，也删除j到i的
    del_adj *= smpl
    if enforce_connected:
        add_edges = 0
        for k, a in enumerate(del_adj):
            if not list(np.nonzero(a)[0]):
                prev_connected = list(np.nonzero(adj[k, :])[0])
                other_node = rnd.choice(prev_connected)
                del_adj[k, other_node] = 1
                del_adj[other_node, k] = 1
                add_edges += 1
    del_adj= sp.csr_matrix(del_adj)
    return del_adj

# 这里的repeat指的是加载分片数据集的，rate是随机掩蔽特征或边的
def load_raw_data(dataset, repeat, device, rate):
    path = './data/{}/'.format(dataset)
    # 特征
    f = np.loadtxt(path + '{}.feature'.format(dataset), dtype=float)
    # 标签
    l = np.loadtxt(path + '{}.label'.format(dataset), dtype=int)

    if repeat is None:
        test = np.loadtxt(path + 'test.txt')
        train = np.loadtxt(path + 'train.txt')
        val = np.loadtxt(path + 'val.txt')
    else:
        # 分片batch
        test = np.loadtxt(path + '{}test.txt'.format(repeat), dtype=int)
        train = np.loadtxt(path + '{}train.txt'.format(repeat), dtype=int)
        val = np.loadtxt(path + '{}val.txt'.format(repeat), dtype=int)

    # 先转成稀疏矩阵再转成密集张量
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense())).to(device)
    # 用指定的rate生成掩蔽矩阵，再对特征矩阵进行掩闭，将掩蔽的特征设置为NaN
    mask = feature_mask(features, rate)
    apply_feature_mask(features, mask)

    # 分别是训练集、测试集和验证集的索引列表，这些索引指向 features 和 label 中的对应样本
    idx_test = test.tolist()
    idx_train = train.tolist()
    idx_val = val.tolist()

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)

    # 标签变成张量和独热编码
    label = torch.LongTensor(np.array(l)).to(device)
    label_oneHot = torch.FloatTensor(to_categorical(l)).to(device)

    # 加载边的数据，随机删除一些边
    struct_edges = np.genfromtxt(path + '{}.edge'.format(dataset), dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    # 构造邻接矩阵
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                         shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # 删除一定比例的边，模拟边缺失
    sadj = edge_delete(rate, sadj)

    # 三个版本的邻接矩阵
    # ppr_input:A（邻接矩阵）+I（单位矩阵），结构教师的输入
    ttadj = sadj + sp.eye(sadj.shape[0])
    ttadj = torch.FloatTensor(ttadj.todense()).to(device)
    # A
    tadj = torch.FloatTensor(sadj.todense()).to(device)
    # stu_input，归一化后的（邻接矩阵+单位矩阵），学生的输入
    sadj = normalize_sparse(sadj + sp.eye(sadj.shape[0]))
    nsadj = torch.FloatTensor(np.array(sadj.todense())).to(device)

    return ttadj, tadj, nsadj, features, label, label_oneHot, idx_train, idx_val, idx_test
