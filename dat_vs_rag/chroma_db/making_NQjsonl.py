'''
создает jsonl файл из датасета natural_questions, который будет использоваться для создания базы данных в ChromaDB. 
запускать в терминале с открытой директорией места сохранения будущего файла (в нашем случае dat_vs_rag/chroma_db/data)
'''

import json
from datasets import load_dataset

from dat_vs_rag.utils.load_params import get_params


PARAMS = get_params()



def load_NQjsonl(OUTPUT_PATH=PARAMS["paths"]["datasets"]["natural_questions_path"], LIMIT=300):
    ds = load_dataset("natural_questions", split="train", streaming=True)


    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if i >= LIMIT:
                break

            question = row["question"]["text"]

            tokens = row["document"]["tokens"]["token"]
            is_html = row["document"]["tokens"]["is_html"]

            clean_tokens = [tok for tok, html_flag in zip(tokens, is_html) if not html_flag]
            text = " ".join(clean_tokens).strip()

            if not text:
                continue

            sample = {
                "id": f"nq_{i}",
                "question": question,
                "text": text
            }

            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

            if i % 100 == 0:
                print(f"saved {i}")

    print(f"done -> {OUTPUT_PATH}")

