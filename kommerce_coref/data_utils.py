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

def rng_for(dialogue_id,role):
    return random.Random(int.from_bytes(hashlib.sha256(f'{SEED}:{dialogue_id}:{role}'.encode()).digest()[:16],'big'))

def clusters_equal(a,b):return {frozenset(c) for c in a}=={frozenset(c) for c in b}

def transform(row):
    original=row['mention_pool'];old_ids=[m['id'] for m in original]
    new_ids=[f'M{i}' for i in range(len(original))]
    rng_for(row['dialogue_id'],'id_assignment').shuffle(new_ids)
    mapping=dict(zip(old_ids,new_ids));inverse={v:k for k,v in mapping.items()}
    display=list(range(len(original)))
    rng_for(row['dialogue_id'],'display_order').shuffle(display)
    pool=[{**original[i],'id':mapping[original[i]['id']]} for i in display]
    gold=[sorted((mapping[m] for m in c),key=lambda m:int(m[1:])) for c in row['gold']]
    gold.sort(key=lambda c:tuple(int(m[1:]) for m in c))
    assert clusters_equal([[inverse[m] for m in c] for c in gold],row['gold'])
    assert len(mapping)==len(inverse)==len(original)
    assert {m['id'] for m in pool}==set(new_ids)
    original_by_id={m['id']:m for m in original}
    for m in pool:assert {**m,'id':inverse[m['id']]}==original_by_id[inverse[m['id']]]
    # Restore the existing history verbatim, then render the same task template.
    head='[대화 히스토리]\n';sep='\n\n[대상 멘션 목록]\n'
    history=row['prompt'].split(head,1)[1].split(sep,1)[0].split('\n')
    assert get_coreference_prompt(history,original)==row['prompt']
    changed={**row,'mention_pool':pool,'gold':gold,'prompt':get_coreference_prompt(history,pool)}
    assert changed['image_urls']==row['image_urls']
    assert changed['prompt'].count('[IMAGE]')==len(row['image_urls'])
    # Regenerate only from sample identity and seed; no gold membership enters the RNG.
    assert len(pool)==len(original)
    return changed,dict(old_to_new=mapping,display_original_ids=[old_ids[i] for i in display])
