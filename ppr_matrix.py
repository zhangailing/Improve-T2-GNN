import numba
import numpy as np
import scipy.sparse as sp

# 计算一个特点节点的PPR值,在后续的三个函数中均有调用
@numba.njit(cache=True,
            locals={
                '_val': numba.float32,
                'res': numba.float32,
                'res_vnode': numba.float32
            })
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
  alpha_eps = alpha * epsilon
  f32_0 = numba.float32(0)
  # 给出图中所有节点，针对特定起始节点的个性化排名，inode指定ppr的起始节点，indptr指定图中每个节点的邻居节点的索引，indices指定图中每个节点的邻居节点，deg指定图中每个节点的度，alpha指定随机游走的概率，epsilon指定停止条件
  # p存储每个节点的ppr值，r存储每个节点的残差（要给邻居节点分配的ppr数量），q存储待处理的节点
  p = {inode: f32_0}
  r = {}
  r[inode] = alpha
  q = [inode]
  while len(q) > 0:
    unode = q.pop()

    res = r[unode] if unode in r else f32_0
    if unode in p:
      p[unode] += res
    else:
      p[unode] = res
    r[unode] = f32_0
    for vnode in indices[indptr[unode]:indptr[unode + 1]]:
      _val = (1 - alpha) * res / deg[unode]
      if vnode in r:
        r[vnode] += _val
      else:
        r[vnode] = _val

      res_vnode = r[vnode] if vnode in r else f32_0
      if res_vnode >= alpha_eps * deg[vnode]:
        if vnode not in q:
          q.append(vnode)
  # 返回图中节点及其相应的PPR值
  return list(p.keys()), list(p.values())

# 对给定的一组节点调用_calc_ppr_node函数，计算它们的PPR值
@numba.njit(cache=True)
def calc_ppr(indptr, indices, deg, alpha, epsilon, nodes):
  # indptr指定图中每个节点的邻居节点的索引，indices指定图中每个节点的邻居节点，deg指定图中每个节点的度，alpha指定随机游走的概率，epsilon指定停止条件，nodes指定计算ppr的节点
  js = []
  vals = []
  for i, node in enumerate(nodes):
    j, val = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
    js.append(j)
    vals.append(val)
  # js和vals分别存储了图中所有节点及其相应的PPR值
  return js, vals

# 对一组节点计算PPR值，但是它只保留每个节点的top-k个PPR邻居和它们的权重
@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, topk):
  js = [np.zeros(0, dtype=np.int64)] * len(nodes)
  vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
  for i in numba.prange(len(nodes)):
    j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
    j_np, val_np = np.array(j), np.array(val)
    idx_topk = np.argsort(val_np)[-topk:]
    js[i] = j_np[idx_topk]
    vals[i] = val_np[idx_topk]
    # js top-k PPR 邻居，vals PPR 邻居相对于节点的权重
  return js, vals

# 与calc_ppr_topk_parallel类似，但是在计算PPR值时，它只考虑keep_nodes中的节点
@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel_keep(indptr, indices, deg, alpha, epsilon, nodes,keep_nodes, topk):
  # keep_nodes 计算PPR时需要考虑的节点
  """Keep only certain nodes"""
  js = [np.zeros(0, dtype=np.int64)] * len(nodes)
  vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
  for i in numba.prange(len(nodes)):
    j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
    for k in range(len(j)):
      if j[k] in keep_nodes:
        pass
      else:
        val[k] = 0
    j_np, val_np = np.array(j), np.array(val)
    idx_topk = np.argsort(val_np)[-topk:]
    js[i] = j_np[idx_topk]
    vals[i] = val_np[idx_topk]
  return js, vals

# 根据给定的邻居和权重信息构建一个稀疏矩阵，在ppr_topk函数中调用
def construct_sparse(neighbors, weights, shape):
  i = np.repeat(np.arange(len(neighbors)),
                np.fromiter(map(len, neighbors), dtype=np.int64))
  j = np.concatenate(neighbors)
  return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)

# 根据是否提供keep_nodes参数，选择调用calc_ppr_topk_parallel或calc_ppr_topk_parallel_keep函数。然后，它使用返回的邻居和权重信息构建一个稀疏矩阵
def ppr_topk(adj_matrix, alpha, epsilon, nodes, topk, keep_nodes=None):
  """Calculate the PPR matrix approximately using Anderson."""
  # adj_matrix邻接矩阵，alpha随机游走的概率，epsilon停止条件，nodes计算ppr的节点，topk返回的top节点的数量，keep_nodes打算计算的节点数
  out_degree = np.sum(adj_matrix > 0, axis=1).A1
  nnodes = adj_matrix.shape[0]

  if keep_nodes:
    keep_nodes = set(keep_nodes)
    neighbors, weights = calc_ppr_topk_parallel_keep(adj_matrix.indptr,adj_matrix.indices,out_degree,numba.float32(alpha),numba.float32(epsilon),nodes, keep_nodes, topk)
  else:
    neighbors, weights = calc_ppr_topk_parallel(adj_matrix.indptr,adj_matrix.indices, out_degree,numba.float32(alpha),numba.float32(epsilon), nodes,topk)

  return construct_sparse(neighbors, weights, (len(nodes), nnodes))

# 首先调用ppr_topk函数获取top-k PPR邻居的稀疏矩阵，然后根据normalization参数的值对矩阵进行归一化处理，这是这个文件最终暴露出来的函数
def topk_ppr_matrix(adj_matrix,alpha,eps,idx,topk,normalization='row',
keep_nodes=None):
  """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""

  topk_matrix = ppr_topk(adj_matrix,
                         alpha,
                         eps,
                         idx,
                         topk,
                         keep_nodes=keep_nodes).tocsr()

  if normalization == 'sym':
    # Assume undirected (symmetric) adjacency matrix
    deg = adj_matrix.sum(1).A1
    deg_sqrt = np.sqrt(np.maximum(deg, 1e-12))
    deg_inv_sqrt = 1. / deg_sqrt

    row, col = topk_matrix.nonzero()
    # assert np.all(deg[idx[row]] > 0)
    # assert np.all(deg[col] > 0)
    topk_matrix.data = deg_sqrt[idx[row]] * topk_matrix.data * deg_inv_sqrt[col]
  elif normalization == 'col':
    # Assume undirected (symmetric) adjacency matrix
    deg = adj_matrix.sum(1).A1
    deg_inv = 1. / np.maximum(deg, 1e-12)

    row, col = topk_matrix.nonzero()
    # assert np.all(deg[idx[row]] > 0)
    # assert np.all(deg[col] > 0)
    topk_matrix.data = deg[idx[row]] * topk_matrix.data * deg_inv[col]
  elif normalization == 'row':
    pass
  else:
    raise ValueError(f"Unknown PPR normalization: {normalization}")

  return topk_matrix
