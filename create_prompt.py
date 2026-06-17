import json
import csv
from typing import Dict

# ================== 配置参数 ==================
jsonl_path = "normal_prompt.jsonl"          # 输入的JSONL文件路径
csv_path = "retrieved_results.csv"             # 包含靶点信息的CSV文件路径
output_path = "prompt.jsonl"         # 输出的JSONL文件路径

# 要优化的目标属性（可根据需要修改）
property_name = "ring_count"         # 例如: ring_count, MW, xlogP, F_Count, TPSA
direction = "lower"                  # 例如: lower, higher, 或区间描述如"between 2 and 4"

# 要替换的prompt在列表中的索引（0-based，第4项对应索引3）
PROMPT_INDEX_TO_REPLACE = 3

# ================== 读取CSV文件，构建chemblid到信息的映射 ==================
csv_mapping: Dict[str, dict] = {}
with open(csv_path, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    # 假设CSV列名为: uniprotid, chemblid, best_ligand_smiles, top1_smiles, top2_smiles, top3_smiles
    for row in reader:
        chemblid = row['chemblid'].strip()
        csv_mapping[chemblid] = {
            'uniprotid': row.get('uniprotid', '').strip(),
            'best_ligand_smiles': row.get('best_ligand_smiles', '').strip(),
            'top1_smiles': row.get('top1_smiles', '').strip(),
            'top2_smiles': row.get('top2_smiles', '').strip(),
            'top3_smiles': row.get('top3_smiles', '').strip(),
        }

# ================== 构建新提示词的函数 ==================
def build_property_guided_prompt(
    uniprotid: str,
    chemblid: str,
    best_ligand_smiles: str,
    top1_smiles: str,
    top2_smiles: str,
    top3_smiles: str,
    property_name: str,
    direction: str
) -> str:
    """
    根据论文PG-TaLiRAGen的风格，构造一个property-guided分子生成提示词。
    """
    prompt_template = f"""你是一位经验丰富的药物化学专家，擅长基于结构相似的示例分子进行启发式分子设计。

【任务】
给定一个目标靶点及其已知的结合配体（种子配体），同时提供三个与该种子配体结构相似、但在目标属性上表现更优的示例分子（exemplars）。请你：
1. 分析这三个示例分子相比于种子配体，在结构上存在哪些共同的改造模式（例如：特定官能团的引入/删除、环系简化、杂原子替换、碳链缩短等）。
2. 基于这些改造模式，结合靶点信息，生成 **3个全新的、化学上有效的 SMILES 字符串**。这些新分子应保留与靶点结合的关键骨架，同时体现出从示例分子中学到的属性优化策略。
3. 不要直接输出示例分子本身，也不要输出种子配体。

【输入信息】
- 靶点 Uniprot ID: {uniprotid}
- 靶点 ChEMBL ID: {chemblid}
- 种子配体 SMILES（与靶点结合最好的已知分子）: {best_ligand_smiles}
- 结构相似且目标属性更优的示例分子（共3个）:
  - Exemplar 1: {top1_smiles}
  - Exemplar 2: {top2_smiles}
  - Exemplar 3: {top3_smiles}

【目标属性】
- 属性名称: {property_name}
- 优化方向: {direction}

【输出格式要求】
请严格按照以下格式输出，不要添加额外解释：

## Structural Analysis of Exemplars
（用1-2句话总结三个示例分子相比于种子配体的共同结构改造策略）

## Generated Molecules
SMILES1
SMILES2
SMILES3
"""
    return prompt_template.strip()

# ================== 处理JSONL文件 ==================
with open(jsonl_path, mode='r', encoding='utf-8') as infile, \
     open(output_path, mode='w', encoding='utf-8') as outfile:

    for line_num, line in enumerate(infile, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"警告：第{line_num}行JSON解析失败，跳过。错误：{e}")
            continue

        chemblid = data.get('chemblid')
        if not chemblid:
            print(f"警告：第{line_num}行缺少'chemblid'字段，跳过。")
            continue

        # 从CSV映射中获取信息
        if chemblid not in csv_mapping:
            print(f"警告：chemblid '{chemblid}' 在CSV中未找到，跳过该行。")
            # 可以选择保留原样或跳过，这里选择跳过
            continue

        info = csv_mapping[chemblid]
        uniprotid = info['uniprotid']
        best_ligand = info['best_ligand_smiles']
        top1 = info['top1_smiles']
        top2 = info['top2_smiles']
        top3 = info['top3_smiles']

        # 检查必要字段是否为空
        if not best_ligand or not top1 or not top2 or not top3:
            print(f"警告：chemblid '{chemblid}' 的exemplar信息不完整，跳过。")
            continue

        # 构造新提示词
        new_prompt = build_property_guided_prompt(
            uniprotid=uniprotid,
            chemblid=chemblid,
            best_ligand_smiles=best_ligand,
            top1_smiles=top1,
            top2_smiles=top2,
            top3_smiles=top3,
            property_name=property_name,
            direction=direction
        )

        # 替换prompt列表中的指定项
        prompt_list = data.get('prompt', [])
        if not isinstance(prompt_list, list):
            print(f"警告：第{line_num}行的'prompt'不是列表，跳过。")
            continue

        # 确保列表长度足够
        if len(prompt_list) <= PROMPT_INDEX_TO_REPLACE:
            # 如果列表太短，先补齐到目标索引+1
            prompt_list.extend([""] * (PROMPT_INDEX_TO_REPLACE + 1 - len(prompt_list)))

        prompt_list[PROMPT_INDEX_TO_REPLACE] = new_prompt

        # 更新data中的prompt
        data['prompt'] = prompt_list

        # 写入新文件
        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"处理完成！结果已保存至：{output_path}")