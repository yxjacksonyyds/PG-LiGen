import json
import re
from rdkit import Chem
from deepseek import chat

MAX_RETRIES = 10

def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def extract_smiles(text):
    matches = re.findall(r'<SMILES>(.*?)</SMILES>', text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None

def process_line(line, outfile):
    obj = json.loads(line)
    prompt_list = obj['prompt']
    chembl_id = obj.get('chemblid', 'UNKNOWN')   # 提取 chemblid，防止缺失

    for _ in range(MAX_RETRIES):
        answer = chat(prompt_list)
        smiles = extract_smiles(answer)
        if smiles and is_valid_smiles(smiles):
            outfile.write(f"{smiles} || {chembl_id}\n")  # 修改：按格式写入
            outfile.flush()
            return
    print(f"警告：经过 {MAX_RETRIES} 次尝试仍无法生成合法 SMILES，跳过 {chembl_id}。")

if __name__ == '__main__':
    input_file = 'prompt.jsonl'
    output_file = 'generated_smiles.txt'

    with open(output_file, 'a', encoding='utf-8') as fout:
        with open(input_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                process_line(line, fout)