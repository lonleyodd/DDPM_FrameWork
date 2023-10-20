import json
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import numpy as np
import os
from tqdm import tqdm

def PretrainProcess(file_path, data_name=None, tokenizer_name=None):
    # tokenizer=AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer = ChatGLMTokenizer(vocab_file=r"./chatglm_tokenizer/tokenizer.model")
    file_tokens = []
    nums=0
    if data_name=="wikipedia":
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        for line in tqdm(data):
            text = line["completion"]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.special_tokens['<eos>'])
            if len(tokens) < 20:
                continue
            nums+=1
            file_tokens += tokens
    elif data_name=="medical":
        with open(file_path,'r',encoding="utf-8") as f:
            data=f.read().strip().split('\n')
        for line in tqdm(data):
            line=json.loads(line)
            text=line["text"]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.special_tokens['<eos>'])
            if len(tokens) < 20:
                continue
            nums+=1
            file_tokens += tokens
    elif data_name=="baidu":
        with open(file_path, 'r', encoding="utf-8") as f:
            data = f.read().strip().split('\n')
        for line in tqdm(data):
            text=''
            line=json.loads(line)
            try:
                text+=line["title"]+':'+line["summary"]
            except:
                continue
            for section in line["sections"]:
                text+=section["title"]+":"+section["content"]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.special_tokens['<eos>'])
            nums+=1
            if len(tokens) < 20:
                continue
            file_tokens += tokens
    arr = np.array(file_tokens, dtype=np.uint16)
    file_name = data_name if data_name else os.path.basename(file_path.split('.')[0])
    file_name = file_name + ".bin"
    print(f"{file_name} valid data is: {nums}")
    with open(os.path.join('./cache', file_name), 'wb') as f:
        f.write(arr.tobytes())
    return arr

def PretrainDataset_cat(data_list):
    data_array=[]
    for bin_file in data_list:
        print(f"process {bin_file}")
        with open(bin_file,'rb') as f:
            data=np.fromfile(f,dtype=np.uint16)
            data_array.append(data)
    arr = np.concatenate(data_array)
    print("tokens nums:",arr.shape)
    with open('./data/pretrain_data.bin','wb') as f:
        f.write(arr.tobytes())

if __name__ == "__main__":
    # data_map={
    #     "wikipedia":"/home/wangh/dataset/nlp_data/wikipedia-cn-20230720-filtered.json",
    #     "medical":"/home/wangh/dataset/nlp_data/medical/pretrain/train_encyclopedia.json",
    #     "baidu":"/home/wangh/dataset/nlp_data/baidubaike.json", 
    # }
    # for name,data in data_map.items():
    #     PretrainProcess(data,name)
    data_list=[
        "./cache/baidu.bin",
        "./cache/medical.bin",
        "./cache/wikipedia.bin",
    ]
    PretrainDataset_cat(data_list)