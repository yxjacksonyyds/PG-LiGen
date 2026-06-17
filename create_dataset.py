import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics.pairwise import cosine_similarity

def smiles_to_embedding(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = mfpgen.GetFingerprint(mol)
    return np.array(fp, dtype=np.float32)

def build_triplet_dataset(csv_path, k=10):
    df = pd.read_csv(csv_path)
    smiles_list = df['SMILES'].tolist()
    logp_list = df['xlogp'].tolist()

    embeddings = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        emb = smiles_to_embedding(smi)
        if emb is not None:
            embeddings.append(emb)
            valid_indices.append(i)

    if len(embeddings) < k + 1:
        raise ValueError(f"有效分子数量不足 {k+1}")

    embeddings = np.array(embeddings)
    valid_smiles = [smiles_list[i] for i in valid_indices]
    valid_logp = [logp_list[i] for i in valid_indices]
    n = len(valid_smiles)

    norm_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim_mat = np.dot(norm_emb, norm_emb.T)

    triplets = []
    for i in range(n):
        sim = sim_mat[i]
        neighbor_idx = np.argsort(sim)[::-1]
        neighbor_idx = [idx for idx in neighbor_idx if idx != i][:k]
        if len(neighbor_idx) < k:
            continue
        neighbor_logps = [valid_logp[idx] for idx in neighbor_idx]
        pos_idx = neighbor_idx[np.argmin(neighbor_logps)]
        neg_idx = neighbor_idx[np.argmax(neighbor_logps)]
        if pos_idx == neg_idx:
            continue
        triplets.append({
            'anchor_smiles': valid_smiles[i],
            'positive_smiles': valid_smiles[pos_idx],
            'negative_smiles': valid_smiles[neg_idx],
            'anchor_logp': valid_logp[i],
            'positive_logp': valid_logp[pos_idx],
            'negative_logp': valid_logp[neg_idx]
        })

    triplet_df = pd.DataFrame(triplets)
    triplet_df.to_csv('triplets.csv', index=False)
    print(f"生成 {len(triplets)} 个三元组 -> triplets.csv")
    return triplet_df

if __name__ == '__main__':
    build_triplet_dataset('train.csv', k=10)