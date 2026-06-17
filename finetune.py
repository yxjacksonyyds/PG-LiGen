import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# 1. 将 SMILES 转换为 PyG 图对象
# ------------------------------
def atom_features(atom):
    """提取原子特征，返回长度为 84 的向量"""
    atomic_num = atom.GetAtomicNum()
    atomic_onehot = [0.0] * 60
    if 1 <= atomic_num <= 60:
        atomic_onehot[atomic_num - 1] = 1.0

    degree = atom.GetDegree()
    degree_onehot = [0.0] * 6
    if degree <= 5:
        degree_onehot[degree] = 1.0

    formal_charge = atom.GetFormalCharge()
    charge_onehot = [0.0] * 5
    if -2 <= formal_charge <= 2:
        charge_onehot[formal_charge + 2] = 1.0

    chiral = [1.0] if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else [0.0]

    hybrid = atom.GetHybridization()
    hybrid_onehot = [0.0] * 6
    if hybrid == Chem.rdchem.HybridizationType.SP:
        hybrid_onehot[0] = 1.0
    elif hybrid == Chem.rdchem.HybridizationType.SP2:
        hybrid_onehot[1] = 1.0
    elif hybrid == Chem.rdchem.HybridizationType.SP3:
        hybrid_onehot[2] = 1.0
    elif hybrid == Chem.rdchem.HybridizationType.SP3D:
        hybrid_onehot[3] = 1.0
    elif hybrid == Chem.rdchem.HybridizationType.SP3D2:
        hybrid_onehot[4] = 1.0
    else:
        hybrid_onehot[5] = 1.0

    aromatic = [1.0] if atom.GetIsAromatic() else [0.0]

    h_count = atom.GetTotalNumHs()
    h_onehot = [0.0] * 5
    if h_count <= 4:
        h_onehot[h_count] = 1.0

    return atomic_onehot + degree_onehot + charge_onehot + chiral + hybrid_onehot + aromatic + h_onehot

def bond_features(bond):
    """提取键特征，返回长度为 9 的向量"""
    bt = bond.GetBondType()
    bond_onehot = [0.0] * 4
    if bt == Chem.rdchem.BondType.SINGLE:
        bond_onehot[0] = 1.0
    elif bt == Chem.rdchem.BondType.DOUBLE:
        bond_onehot[1] = 1.0
    elif bt == Chem.rdchem.BondType.TRIPLE:
        bond_onehot[2] = 1.0
    elif bt == Chem.rdchem.BondType.AROMATIC:
        bond_onehot[3] = 1.0

    conj = [1.0] if bond.GetIsConjugated() else [0.0]
    ring = [1.0] if bond.IsInRing() else [0.0]
    stereo = [0.0, 0.0, 0.0]
    stereo_type = bond.GetStereo()
    if stereo_type == Chem.rdchem.BondStereo.STEREOZ:
        stereo[0] = 1.0
    elif stereo_type == Chem.rdchem.BondStereo.STEREOE:
        stereo[1] = 1.0
    else:
        stereo[2] = 1.0

    return bond_onehot + conj + ring + stereo   # 4+1+1+3 = 9

def mol_to_graph(smiles):
    """将 SMILES 转换为 PyG Data 对象"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # 节点特征
    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    # 边索引和边特征
    edges = []
    edge_feats = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
        bf = bond_features(bond)
        edge_feats.append(bf)
        edge_feats.append(bf.copy())
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)   # [E, 9]
    else:
        # 无键分子（如单个原子）: 边特征维度必须为 9
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 9), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ------------------------------
# 2. GIN 模型定义（5 层，MLP 隐藏维度 256）
# ------------------------------
class GINEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=256, num_layers=5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            else:
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        x = self.pool(x, batch)
        return x

# ------------------------------
# 3. 数据集类
# ------------------------------
class TripletDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        required = ['anchor_smiles', 'positive_smiles', 'negative_smiles']
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"CSV 必须包含列: {col}")
        self.anchors = self.df['anchor_smiles'].tolist()
        self.positives = self.df['positive_smiles'].tolist()
        self.negatives = self.df['negative_smiles'].tolist()

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        a_smi = self.anchors[idx]
        p_smi = self.positives[idx]
        n_smi = self.negatives[idx]
        a_graph = mol_to_graph(a_smi)
        p_graph = mol_to_graph(p_smi)
        n_graph = mol_to_graph(n_smi)
        if a_graph is None or p_graph is None or n_graph is None:
            return None
        return (a_graph, p_graph, n_graph)

def collate_fn(batch):
    """过滤无效的三元组，并返回列表形式的图对象"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    anchors, positives, negatives = zip(*batch)
    return list(anchors), list(positives), list(negatives)

def pyg_collate(anchors, positives, negatives):
    """将图对象列表合并成 PyG Batch 对象"""
    a_batch = Batch.from_data_list(anchors)
    p_batch = Batch.from_data_list(positives)
    n_batch = Batch.from_data_list(negatives)
    return a_batch, p_batch, n_batch

# ------------------------------
# 4. InfoNCE 损失（使用 batch 内所有负样本）
# ------------------------------
def infonce_loss(anchor_emb, positive_emb, negative_emb, temperature=0.1):
    """
    anchor_emb:   [B, D]
    positive_emb: [B, D]
    negative_emb: [B, D]   # 每个 anchor 对应的一个负样本
    分母包含 batch 内所有 positive 和 negative（共 2B 个候选）
    """
    B = anchor_emb.size(0)
    all_candidates = torch.cat([positive_emb, negative_emb], dim=0)  # [2B, D]
    sim = torch.matmul(anchor_emb, all_candidates.T) / temperature   # [B, 2B]
    labels = torch.arange(B, device=anchor_emb.device)
    loss = F.cross_entropy(sim, labels)
    return loss

# ------------------------------
# 5. 训练与验证函数
# ------------------------------
def train_one_epoch(model, dataloader, optimizer, device, temperature):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue
        anchors, positives, negatives = batch
        if len(anchors) == 0:
            continue
        a_batch, p_batch, n_batch = pyg_collate(anchors, positives, negatives)
        a_batch = a_batch.to(device)
        p_batch = p_batch.to(device)
        n_batch = n_batch.to(device)

        anchor_emb = model(a_batch)
        positive_emb = model(p_batch)
        negative_emb = model(n_batch)

        loss = infonce_loss(anchor_emb, positive_emb, negative_emb, temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches if num_batches > 0 else float('inf')

def validate(model, dataloader, device, temperature):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if batch is None:
                continue
            anchors, positives, negatives = batch
            if len(anchors) == 0:
                continue
            a_batch, p_batch, n_batch = pyg_collate(anchors, positives, negatives)
            a_batch = a_batch.to(device)
            p_batch = p_batch.to(device)
            n_batch = n_batch.to(device)

            anchor_emb = model(a_batch)
            positive_emb = model(p_batch)
            negative_emb = model(n_batch)

            loss = infonce_loss(anchor_emb, positive_emb, negative_emb, temperature)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else float('inf')

# ------------------------------
# 主程序
# ------------------------------
if __name__ == "__main__":
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 50
    TEMPERATURE = 0.1
    PATIENCE = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # 请修改为你的 CSV 文件路径
    CSV_PATH = "triplets.csv"
    dataset = TripletDataset(CSV_PATH)

    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    # 获取输入特征维度（实际为84）
    sample_graph = mol_to_graph("CC")  # 乙烷
    if sample_graph is None:
        raise RuntimeError("无法生成样例图，请检查 RDKit 安装及 SMILES 有效性")
    in_dim = sample_graph.x.size(1)
    print(f"Input feature dimension: {in_dim}")

    model = GINEncoder(in_dim=in_dim, hidden_dim=256, out_dim=256, num_layers=5).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, TEMPERATURE)
        val_loss = validate(model, val_loader, DEVICE, TEMPERATURE)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_gin_model.pth")
            print("  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print("Training finished. Best model saved as best_gin_model.pth")