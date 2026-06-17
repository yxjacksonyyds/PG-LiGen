1.Dataset:
Crossdocked2020:包含uniprotid、chemblid
chebi：csv_folder文件夹下train.csv

2.trained model:
model文件夹下

3.csv_folder文件夹其他文件：中间的数据处理过程

4.how to run？
###4.1创建property数据集
执行create_dataset.py 得到triplets.csv数据集
###4.2训练模型
执行finetune.py 得到model/best…….pth
###4.3检索靶点结构最相似分子
执行retrieval.py得到"retrieved_results.csv"文件
###4.4构造prompt
执行create_prompt.py得到prompt.jsonl
###4.5生成分子
执行generate.py文件得到smiles.txt结果
###4.6指标评估
rdkit、autodock的pypi即可做到

5.环境：settings.txt文件
