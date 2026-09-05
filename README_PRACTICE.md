# Part 5: DFlash Parallel Block Drafting

실제 `Qwen/Qwen3-4B` Target과 pretrained `z-lab/Qwen3-4B-DFlash-b16` drafter를 사용한다.
학생은 **drafter forward 1회 → block hidden states → Target LM head → 동시 argmax**만 구현한다.
Target prefill, context feature 추출, mask embedding과 position 준비, verification,
acceptance, bonus token, KV cache 관리는 기존 `dflash_generate()`가 담당한다.
실습 경로는 original DFlash의 `temperature=0`이다.

## Colab setup

런타임을 GPU로 선택하고 다음 셀을 실행한다.

```python
%cd /content
!git clone -q https://github.com/kimjongjip/dflash-practice.git

import sys
sys.path.insert(0, "/content/dflash-practice")

import torch
import transformers
from dflash.model import dflash_generate
from dflash.practice import (
    load_practice_models,
    stop_token_ids,
    apply_practice_chat_template,
)
print(torch.__version__, transformers.__version__)
```

Colab의 기존 `torch`, `transformers`, `huggingface_hub`, `safetensors`와 그 기본
의존성을 사용한다. `pip install dflash`나 `dflash[local]`은 필요 없다.
실습 helper는 `benchmark.py`와 `datasets`를 import하지 않으며 benchmark용
`tqdm`·`requests`를 별도로 설치하지 않는다. Transformers 자체의 간접 의존성은 유지한다.

## 모델 load

```python
target, draft, tokenizer = load_practice_models()
print(target.device, target.dtype, draft.block_size)
```

두 모델은 같은 CUDA device에서 SDPA와 `eval()`을 사용한다. BF16 지원 GPU에서는
`torch.bfloat16`, T4처럼 지원하지 않는 GPU에서는 `torch.float16`을 자동 선택한다.
BF16 에뮬레이션을 지원으로 보고하는 PyTorch도 고려해 CUDA compute capability가
8 이상인지 함께 확인한다.
학생 함수가 미완성이어도 import와 모델 loading은 성공한다.

이 BF16 checkpoint의 context projection은 FP16에서 overflow할 수 있다.
helper는 FP16 로딩 시 `fc`와 이어지는 normalization 계산만 FP32로 수행한 뒤
FP16으로 돌려준다. 공식 model architecture·parameter 이름·shape는 유지하며
decoder와 KV cache는 FP16을 사용한다. 추가 메모리는 약 63 MiB다.

## Student function과 monkey patch

아래 셀의 TODO를 구현한다. 입력은 모두 공식 generation 코드가 준비한다.
`verify_size`에는 block의 첫 anchor token이 포함되므로 제안할 token 수는
`verify_size - 1`이다. 반환값은 입력과 같은 device의 `torch.long` tensor이며
shape는 `[1, verify_size - 1]`이다.

```python
import dflash.student_block as student_block

def parallel_block_draft(
    model,
    target_hidden,
    noise_embedding,
    draft_position_ids,
    past_key_values_draft,
    output_head,
    verify_size,
):
    ##################### 실습 코드 #####################
    # TODO: prepared inputs로 drafter를 한 번 forward한다 (use_cache=True).
    # TODO: 마지막 verify_size - 1개 hidden state를 선택한다.
    # TODO: model.compute_logits와 output_head로 모든 위치의 logits을 구한다.
    # TODO: 모든 위치의 greedy token을 동시에 선택한다.
    raise NotImplementedError("Parallel Block Drafting을 구현하세요.")
    ################### 실습 코드 끝 ###################
    # return draft_tokens

student_block.parallel_block_draft = parallel_block_draft
```

이 셀을 다시 실행하면 다음 generation부터 새 함수가 적용된다.
`importlib.reload`나 저장소 파일 수정은 필요 없다. 첫 greedy draft block에
도달했을 때 함수가 미완성이면 `NotImplementedError`가 발생한다.

## Full DFlash generation

학생 함수를 완성한 뒤 실행한다. Qwen3의 thinking은 기본적으로 끈다.

```python
messages = [{"role": "user", "content": "Explain why the sky is blue."}]
text = apply_practice_chat_template(tokenizer, messages)
input_ids = tokenizer.encode(
    text, add_special_tokens=False, return_tensors="pt",
).to(target.device)
eos_ids = stop_token_ids(target, tokenizer)

result = dflash_generate(
    model=draft,
    target=target,
    input_ids=input_ids,
    max_new_tokens=128,
    stop_token_ids=eos_ids,
    temperature=0.0,
    block_size=16,
    return_stats=True,
)
print(tokenizer.decode(
    result.output_ids[0, result.num_input_tokens:], skip_special_tokens=True,
))
```

## Block size sweep

모델은 한 번만 load하고 같은 prompt로 비교한다. 각 호출의 KV cache는 공식
generation 함수가 새로 만든다. 다음 셀은 각 크기를 한 번 warmup한 뒤 측정한다.

```python
import statistics

rows = []
for block_size in [2, 4, 8, 12, 16]:
    kwargs = dict(
        model=draft, target=target, input_ids=input_ids,
        max_new_tokens=128, stop_token_ids=eos_ids,
        temperature=0.0, block_size=block_size,
    )
    dflash_generate(**kwargs)  # warmup
    result = dflash_generate(**kwargs, return_stats=True)
    average = (statistics.mean(result.acceptance_lengths)
               if result.acceptance_lengths else 0.0)
    row = {
        "block_size": block_size,
        "average_acceptance_length": average,
        "TPS": 1.0 / result.time_per_output_token,
        "num_output_tokens": result.num_output_tokens,
    }
    rows.append(row)
    print(row)
```

`acceptance_lengths`는 공식 통계 그대로 **round마다 실제로 전진한 token 수**다.
보통 accepted draft prefix 길이 + 1이며 EOS와 출력 길이 제한에서 줄어들 수 있다.
`time_per_output_token`은 prefill을 제외한 decode 시간 / 전체 출력 token 수이고,
그 역수가 위 TPS다. 첫 prefill token도 분모에 포함되는 공식 정의를 유지한다.
Block size가 커질 때 제안 token 수, acceptance, verification 비용이 함께
TPS에 어떤 영향을 주는지 관찰한다. 짧은 출력의 시간은 변동이 크므로 여러 번 비교한다.

## 검증

CPU unit test에는 checkpoint 다운로드나 pytest 설치가 필요 없다.

```bash
python -m unittest discover -s tests -v
```

강사용 실제 checkpoint smoke test에는 reference 구현이 들어 있다.

```bash
python scripts/practice_smoke_test.py --block-sizes 2 4 8 12 16
# T4와 같은 dtype 경로를 다른 GPU에서도 확인:
python scripts/practice_smoke_test.py --dtype float16
```

Transformers 4.56.1에서 재현한 `DynamicCache.activate_past_recording` 부재와
`crop(0)` 의미 차이는 API 존재 여부를 확인해 처리한다. 최신 cache의 recording과
crop 호출은 유지한다. 실제 Colab 이미지의 패키지 구성은 바뀔 수 있으므로 오류가
발생하면 위 버전 출력과 traceback을 먼저 확인한다. 버전 재설치를 setup에 넣지 않는다.

로컬 검증: RTX 3090, PyTorch 2.5.1+cu121 / Transformers 5.16.1에서 실제 두 모델의
BF16·FP16 전체 sweep과 각 16 token 생성을 확인했다. PyTorch 2.6.0+cu124 /
Transformers 4.56.1에서도 BF16 block size 16 생성이 성공했다. 두 환경 모두
unit test 12개와 import smoke가 통과했다. Colab/T4 실기기 검증은 수행하지 않았다.
