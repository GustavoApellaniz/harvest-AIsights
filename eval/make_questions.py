"""Gera perguntas SINTÉTICAS (a partir de modelos de frase) com a linha esperada, para testar a recuperação.

Uso: python -m eval.make_questions --csv data/agritel_quotes.csv --out eval/questions.jsonl --n 40

Atenção: como as perguntas são montadas com os próprios campos da linha (produto, mercado, contrato),
elas são mais fáceis que perguntas reais de usuários. O hit@k obtido com elas tende a SUPERESTIMAR
o desempenho real; escreva também perguntas à mão para uma avaliação mais honesta.
"""

import argparse
import csv
import json
import random
from pathlib import Path

from app.services.tabular import PRODUCT_PT, normalize_contract


def build_questions(csv_path: Path, n: int, seed: int = 42) -> list[dict]:
    source = csv_path.name
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        header = f.readline()
        f.seek(0)
        rows = list(csv.DictReader(f, delimiter=max(";,\t", key=header.count)))

    candidates = []
    for line_no, row in enumerate(rows, start=2):
        if row.get("record_type") == "physical" and row.get("description"):
            question = f"Qual o preço final de: {row['description'].strip()}?"
        elif row.get("record_type") == "futures" and row.get("contract"):
            product = PRODUCT_PT.get(row["product"], row["product"])
            contract = normalize_contract(row["contract"]).split(" (")[0]
            question = f"Qual o preço final do contrato de {product} para {contract} no mercado {row['market']}?"
        else:
            continue
        candidates.append({"question": question, "expected_source": source, "expected_row": line_no})

    random.Random(seed).shuffle(candidates)
    return sorted(candidates[:n], key=lambda c: c["expected_row"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()
    questions = build_questions(args.csv, args.n)
    args.out.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\n", encoding="utf-8")
    print(f"{len(questions)} perguntas escritas em {args.out}")


if __name__ == "__main__":
    main()
