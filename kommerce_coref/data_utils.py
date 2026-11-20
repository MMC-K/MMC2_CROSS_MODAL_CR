"""Reproduce independent, gold-agnostic mention permutations."""
import random,hashlib
SEED=42

def get_coreference_prompt(dialog_history, mentions_pool):
    history_str = "\n".join(dialog_history)
    pool_str = "\n".join([f"{m['id']}: [Turn {m['turn_idx']}] {m['content']} ({m['type']})" for m in mentions_pool])
    
    return f"""Task: Coreference Resolution (상호참조 해결 - 멘션 클러스터링)
제공된 대화 히스토리에서 동일한 패션 아이템(엔티티)을 지칭하는 멘션들을 그룹화하세요.

[대화 히스토리]
{history_str}

[대상 멘션 목록]
{pool_str}

* 지침:
1. 위 목록의 멘션 ID를 사용하여 동일한 대상을 가리키는 멘션들을 하나의 리스트로 묶으세요.
2. 각 리스트(클러스터)는 하나의 고유한 패션 아이템을 의미합니다.
3. 이미지 멘션의 경우 좌표 [x, y, w, h] 정보가 포함되어 있습니다.

응답 형식: JSON
{{
  "clusters": [
    ["M1", "M2"], 
    ["M3", "M4", "M5"]
  ]
}}
"""
