'''
hotpotqa distractor train + validation context -> corpus -> parquet로 저장 
경로 : hotpotqa -> corpus_distractor.parquet
-원본 데이터 개수: 973367
-중복 제거 후 개수: 509315
'''
CORPUS_PATH = "/aix23604/hotpotqa/corpus_distractor.parquet"
ds_path = "/aix23604/hotpotqa/distractor"

import pandas as pd
from utils import load_datasets

def corpus_generate(datasets):
    raw_data = []
    
    for split in datasets.keys():
        for data in datasets[split]:
            context = data['context'] # dict_keys(['title', 'sentences'])
            titles = context['title'] # list[str]
            sent_lists = context['sentences'] # list[list[str]]
            
            for title, sents in zip(titles, sent_lists):
                paragraph = " ".join(sents)
                raw_data.append({"title": title, "text": paragraph, "source_id": data["id"]})

    df = pd.DataFrame(raw_data) 
    print(f"원본 데이터 개수: {len(df)}")

    df_clean = df.drop_duplicates(subset=['title', 'text'], keep='first')
    print(f"중복 제거 후 개수: {len(df_clean)}") 

    return df_clean

def save_corpus(df, save_path):
    df.to_parquet(save_path, index=False)
    print(f"저장 완료: {save_path}")


ds_dataset = load_from_disk(ds_path)

df_corpus = corpus_generate(ds_dataset)
save_corpus(df_corpus, CORPUS_PATH)

