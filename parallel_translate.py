import os
import re
import time
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types

# --------------------------------------------------
# 환경 로드
# --------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수를 설정해주세요.")

MODEL_NAME = "gemini-2.5-flash"
SOURCE_DIR = "source_mds"
TRANSLATED_DIR = "translated_mds"
PROMPT_EXAMPLE_DIR = "prompt_examples"
MAX_RETRIES = 2
RETRY_WAIT_SECONDS = 60

DEFAULT_MAX_CAP = 200

# --------------------------------------------------
# 시스템 인스트럭션
# --------------------------------------------------

def create_system_instruction(example_en_texts: List[str], example_ko_texts: List[str]) -> str:
    header = """
You are a specialized translator for academic papers, translating Markdown documents from English to Korean.
Follow these rules meticulously:
1.  Translate the main content into professional, natural-sounding Korean.
2.  **Crucially, do not translate technical English terms.** Keep terms like 'Transformer', 'DETR', 'cross attention', 'encoder', 'decoder', etc., in their original English form.
3.  **Ensure that figure links are preserved.** Do not omit any figure URLs or accompanying descriptions.
4.  **Include references formatted as [1], but when encountering footnote markers like [^1], remove only the [^1] marker while keeping the adjacent explanation text intact.**
5.  **Use the provided examples as a reference for tone, translation style, and formatting.** This includes matching whitespace, line breaks, and paragraph divisions exactly as shown.
6.  **Do not create any new titles or headings; translate only the content you’ve been given.**
7.  **Enhance readability by using bold formatting for important terms or key phrases, as in the examples.**

Now, here are multiple examples:
"""
    examples = ""
    for en, ko in zip(example_en_texts, example_ko_texts):
        examples += f"""
---
### Translation Example ###
[English Original]
{en}

[Korean Translation]
{ko}
"""
    footer = """
---
Now, please translate the following English text into Korean based on all the rules and the examples provided.
"""
    return header + examples + footer

# --------------------------------------------------
# 예시 로드
# --------------------------------------------------

def load_examples() -> Tuple[List[str], List[str]]:
    example_bases = ["small_example"]
    example_en_texts: List[str] = []
    example_ko_texts: List[str] = []

    def split_and_clean(text: str) -> List[str]:
        chunks = re.split(r'(^(?:#+ ).*$)', text, flags=re.MULTILINE)
        return [c.strip() for i, c in enumerate(chunks) if i % 2 == 0 and c.strip()]

    for base in example_bases:
        en_path = os.path.join(PROMPT_EXAMPLE_DIR, f"{base}_en.md")
        ko_path = os.path.join(PROMPT_EXAMPLE_DIR, f"{base}_ko.md")
        if not (os.path.exists(en_path) and os.path.exists(ko_path)):
            raise FileNotFoundError(f"예시 파일 누락: {base}_en.md 또는 {base}_ko.md")
        with open(en_path, 'r', encoding='utf-8') as f_en, open(ko_path, 'r', encoding='utf-8') as f_ko:
            en_pars = split_and_clean(f_en.read())
            ko_pars = split_and_clean(f_ko.read())
            for en_par, ko_par in zip(en_pars, ko_pars):
                example_en_texts.append(en_par)
                example_ko_texts.append(ko_par)
    if not example_en_texts:
        raise RuntimeError("유효한 예시 텍스트를 찾지 못했습니다.")
    return example_en_texts, example_ko_texts

# --------------------------------------------------
# 분할 & 번역 유틸
# --------------------------------------------------

def split_markdown(content: str) -> List[str]:
    return re.split(r'(^(?:#+ ).*$)', content, flags=re.MULTILINE)

def needs_translation(i: int, chunk: str) -> bool:
    if i % 2 == 1:  # 헤더
        return False
    return bool(chunk.strip())

# --------------------------------------------------
# 실제 단락 번역 (재시도 포함)
# --------------------------------------------------

def translate_chunk(client: genai.Client, model_config: types.GenerateContentConfig, idx: int, text: str):
    attempt = 0
    while attempt <= MAX_RETRIES:
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=text,
                config=model_config
            )
            return idx, "\n\n" + resp.text + "\n\n"
        except Exception as e:
            attempt += 1
            if attempt > MAX_RETRIES:
                print(f"   [실패] 청크 {idx} -> 원문 유지. 에러: {e}")
                return idx, text
            print(f"   [재시도] 청크 {idx} ({attempt}/{MAX_RETRIES}) 에러: {e} -> {RETRY_WAIT_SECONDS}s 대기")
            time.sleep(RETRY_WAIT_SECONDS)

# --------------------------------------------------
# 파일 단위 처리 (파일은 순차, 내부 단락은 '전부 동시에 제출')
# --------------------------------------------------

def translate_file_all_at_once(path: str, client: genai.Client, model_config: types.GenerateContentConfig):
    name = os.path.basename(path)
    print(f"📄 '{name}' 번역 시작 (모든 단락 동시 제출)")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = split_markdown(content)
    targets = [(i, c) for i, c in enumerate(chunks) if needs_translation(i, c)]
    print(f"   - 번역 대상 청크 수: {len(targets)} (헤더/공백 제외)")
    if not targets:
        print("   - 번역 대상 없음 -> 그대로 저장")

    # 동시 스레드 수: 대상 청크 수 또는 상한 중 작은 값
    max_workers = min(len(targets), DEFAULT_MAX_CAP) if targets else 1

    results: Dict[int, str] = {}
    submitted = 0

    # 한 번에 모두 future 제출
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(translate_chunk, client, model_config, i, c) for i, c in targets]
        for fut in as_completed(futures):
            idx, translated = fut.result()
            results[idx] = translated
            submitted += 1
            if submitted % 5 == 0 or submitted == len(targets):
                print(f"      진행률 {submitted}/{len(targets)} ({submitted/len(targets)*100 if targets else 100:.1f}%)")

    # 재조합
    final_parts: List[str] = []
    for i, original in enumerate(chunks):
        final_parts.append(results.get(i, original))
    final_text = "".join(final_parts)

    os.makedirs(TRANSLATED_DIR, exist_ok=True)
    base, ext = os.path.splitext(name)
    out = os.path.join(TRANSLATED_DIR, f"{base}_translate{ext}")
    with open(out, 'w', encoding='utf-8') as f:
        f.write(final_text)
    print(f"✅ 완료: {out}\n")

# --------------------------------------------------
# 메인
# --------------------------------------------------

def main():
    if not os.path.exists(SOURCE_DIR):
        raise FileNotFoundError(f"'{SOURCE_DIR}' 폴더가 없습니다.")

    en_examples, ko_examples = load_examples()
    system_instruction = create_system_instruction(en_examples, ko_examples)

    client = genai.Client(api_key=GOOGLE_API_KEY)
    model_config = types.GenerateContentConfig(
        temperature=0,
        system_instruction=system_instruction,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.md')]
    files.sort()
    if not files:
        print("번역할 파일이 없습니다.")
        return

    print(f"총 {len(files)}개 파일. 파일은 순차, 각 파일 내부 단락은 전부 동시에 제출 (상한 {DEFAULT_MAX_CAP}).")

    for fname in files:
        translate_file_all_at_once(os.path.join(SOURCE_DIR, fname), client, model_config)

if __name__ == "__main__":
    main()
