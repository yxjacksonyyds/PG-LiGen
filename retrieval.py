import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from run import MolecularEmbedder

def clean_smiles_column(df, smiles_col='best_ligand_smiles'):
    """
    清理 smiles 列：转换为字符串，过滤掉 NaN、None、非字符串或空值。
    返回清理后的 DataFrame 和被删除的索引列表。
    """
    # 保存原始长度
    original_len = len(df)
    # 转换为字符串，NaN 会变成 'nan' 字符串，但我们可以根据原始类型判断
    # 更好的方式：先复制一列，标记无效
    mask_invalid = pd.Series([False] * len(df))
    
    # 找出 NaN 或 None
    mask_nan = df[smiles_col].isna()
    # 找出非字符串类型（例如 float）
    mask_non_str = df[smiles_col].apply(lambda x: not isinstance(x, str))
    # 字符串中空值或仅空白
    mask_empty = df[smiles_col].apply(lambda x: isinstance(x, str) and len(x.strip()) == 0)
    
    mask_invalid = mask_nan | mask_non_str | mask_empty
    
    # 删除无效行
    df_clean = df[~mask_invalid].copy()
    # 确保 SMILES 列为字符串（去除首尾空格）
    df_clean[smiles_col] = df_clean[smiles_col].astype(str).str.strip()
    
    removed_indices = df.index[mask_invalid].tolist()
    print(f"清理 {smiles_col} 列：原始 {original_len} 行，无效 {len(removed_indices)} 行，保留 {len(df_clean)} 行")
    return df_clean, removed_indices

def build_train_embeddings(train_csv_path, embedder, smiles_col='smiles', batch_size=128):
    """
    读取训练集 SMILES，清洗无效值，批量计算嵌入，返回嵌入矩阵和对应的 SMILES 列表
    """
    df_train = pd.read_csv(train_csv_path)
    if smiles_col not in df_train.columns:
        raise ValueError(f"train.csv 中未找到列 '{smiles_col}'")
    
    # 清洗训练集 SMILES
    df_train_clean, removed = clean_smiles_column(df_train, smiles_col)
    if removed:
        print(f"训练集中移除了 {len(removed)} 行无效 SMILES，可查看输出日志。")
    
    smiles_list = df_train_clean[smiles_col].tolist()
    print(f"训练集有效分子数量: {len(smiles_list)}")
    
    embeddings = []
    valid_smiles = []
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="计算训练集嵌入"):
        batch_smiles = smiles_list[i:i+batch_size]
        emb_batch = embedder.embed(batch_smiles, return_numpy=True)  # (n_valid, 256)
        if len(emb_batch) > 0:
            embeddings.append(emb_batch)
            # 注意：embedder.embed 会自动跳过无效 SMILES，我们需要对齐有效 SMILES
            # 方法：对 batch 中的每个 SMILES 单独测试有效性（效率稍低但准确）
            valid_batch = [s for s in batch_smiles if embedder.embed_single(s) is not None]
            valid_smiles.extend(valid_batch)
    if not embeddings:
        raise RuntimeError("训练集中没有有效 SMILES")
    embeddings = np.vstack(embeddings)
    print(f"最终有效训练集分子数: {len(valid_smiles)}, 嵌入矩阵形状: {embeddings.shape}")
    return embeddings, valid_smiles

def retrieve_top_k(query_smiles, query_embedder, train_embeddings, train_smiles, k=3):
    """
    对单个查询 SMILES 检索 top-k 相似分子
    返回相似分子 SMILES 列表（长度为 k，若不足则用 None 填充）
    """
    # 查询 SMILES 有效性已在外部保证，这里再加一层保护
    if not isinstance(query_smiles, str) or len(query_smiles.strip()) == 0:
        return [None] * k
    emb = query_embedder.embed_single(query_smiles, return_numpy=True)
    if emb is None:
        return [None] * k
    similarities = cosine_similarity(emb.reshape(1, -1), train_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:k]
    top_smiles = [train_smiles[idx] for idx in top_indices]
    if len(top_smiles) < k:
        top_smiles.extend([None] * (k - len(top_smiles)))
    return top_smiles

def main():
    # 文件路径（请根据实际情况修改）
    TARGET_CSV = "best_ligand.csv"           # 包含 uniprotid, chemblid, best_ligand_smiles
    TRAIN_CSV = "train.csv"              # 包含 smiles 列
    MODEL_PATH = "best_gin_model.pth"
    OUTPUT_CSV = "retrieved_results.csv"
    ERROR_CSV = "invalid_smiles_log.csv" # 保存被过滤的查询记录
    K = 3

    # 1. 加载嵌入模型
    embedder = MolecularEmbedder(MODEL_PATH)

    # 2. 构建训练集的嵌入索引
    train_embeddings, train_smiles = build_train_embeddings(TRAIN_CSV, embedder, smiles_col='smiles')

    # 3. 读取查询目标文件并清洗
    df_target = pd.read_csv(TARGET_CSV)
    required_cols = ['uniprotid', 'chemblid', 'best_ligand_smiles']
    for col in required_cols:
        if col not in df_target.columns:
            raise ValueError(f"目标 CSV 缺少列 '{col}'")
    
    # 清洗查询 SMILES
    df_target_clean, removed_target = clean_smiles_column(df_target, 'best_ligand_smiles')
    if removed_target:
        # 保存被过滤的行到错误日志
        df_error = df_target.loc[removed_target]
        df_error.to_csv(ERROR_CSV, index=False)
        print(f"发现 {len(removed_target)} 个无效查询 SMILES，已保存至 {ERROR_CSV}")
        print("这些查询将被跳过，不进行检索。")
    
    if len(df_target_clean) == 0:
        print("没有有效的查询 SMILES，程序退出。")
        return

    # 4. 对每个有效查询分子检索 top-k
    top_smiles_list = []
    for idx, row in tqdm(df_target_clean.iterrows(), total=len(df_target_clean), desc="检索"):
        query_smi = row['best_ligand_smiles']
        top_k = retrieve_top_k(query_smi, embedder, train_embeddings, train_smiles, k=K)
        top_smiles_list.append(top_k)

    # 将 top_k 拆分为三列
    df_target_clean['top1_smiles'] = [t[0] for t in top_smiles_list]
    df_target_clean['top2_smiles'] = [t[1] for t in top_smiles_list]
    df_target_clean['top3_smiles'] = [t[2] for t in top_smiles_list]

    # 5. 保存结果
    df_target_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"结果已保存至: {OUTPUT_CSV}")
    if removed_target:
        print(f"无效查询记录已保存至: {ERROR_CSV}")

if __name__ == "__main__":
    main()