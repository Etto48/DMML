import pandas as pd
import json

def preprocess_csv(path: str) -> tuple[list[list[int]],dict[int,str]]:
    df = pd.read_csv(path)
    transactions = []
    item_id = 0
    items: dict[str,int] = {}
    for t in df["Product"]:
        # t is a valid json array of strings
        t: str
        t = t.replace("'", "\"")
        try:
            t = json.loads(t)
        except json.decoder.JSONDecodeError:
            print(f"Error decoding {t}")
            continue
        for i in t:
            if i not in items:
                items[i] = item_id
                item_id += 1
        transactions.append([items[i] for i in t])
    
    items_reverse: dict[int,str] = {v: k for k, v in items.items()}
    return transactions, items_reverse
    

if __name__ == '__main__':
    path = f"{__file__}/../../../datasets/labelled_retail.csv"
    print(preprocess_csv(path)[1])
    