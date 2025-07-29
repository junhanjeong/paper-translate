from google import genai
from google.genai import types
import os
import re
import time
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# Gemini API 설정
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수를 설정해주세요.")

# --- 설정 변수 ---
MODEL_NAME = "gemini-2.5-flash"
SOURCE_DIR = "source_mds"
TRANSLATED_DIR = "translated_mds"
PROMPT_EXAMPLE_DIR = "prompt_examples"
REQUESTS_PER_MINUTE = 10
SECONDS_TO_WAIT = 60

# --- 프롬프트 생성 함수 (업데이트됨) ---
def create_system_instruction(example_en_texts, example_ko_texts):
    """번역 작업을 위한 시스템 지침(System Instruction)을 생성합니다."""
    # 사용자 맞춤 지시사항 반영 (공통 헤더)
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
    # 여러 예시 병합
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
    # 마무리 안내
    footer = """
---
Now, please translate the following English text into Korean based on all the rules and the examples provided.
"""
    return header + examples + footer

# --- 메인 번역 함수 (업데이트됨) ---
def translate_markdown_file(filepath, client, model_config):
    """Markdown 파일을 읽고, 단락별로 번역한 후 저장합니다."""
    print(f"📄 '{os.path.basename(filepath)}' 파일 번역을 시작합니다...")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"   [오류] 파일을 찾을 수 없습니다: {filepath}")
        return

    # 헤더(#)를 기준으로 텍스트 분할
    chunks = re.split(r'(^(?:#+ ).*$)', content, flags=re.MULTILINE)
    print(f"   - 총 {len(chunks)//2}개의 단락으로 분할되었습니다.")
    
    translated_content = []
    request_count = 0
    start_time = time.time()

    for i, chunk in enumerate(chunks):
        is_header = i % 2 == 1
        is_blank = not chunk.strip()
        
        if is_header or is_blank:
            translated_content.append(chunk)
            continue

        # API 속도 제어
        if request_count >= REQUESTS_PER_MINUTE:
            elapsed_time = time.time() - start_time
            if elapsed_time < SECONDS_TO_WAIT:
                sleep_time = SECONDS_TO_WAIT - elapsed_time + 2 # 여유 시간을 두기 위해 2초 추가
                print(f"   ⏳ API 속도 제어를 위해 {sleep_time:.1f}초 대기합니다...")
                time.sleep(sleep_time)
            request_count = 0
            start_time = time.time()

        # API 호출 및 번역
        print(f"   - 단락 {i//2 + 1} 번역 중...")
        max_retries = 2
        retries = 0
        translated = False
        while not translated and retries < max_retries:
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=chunk,
                    config=model_config
                )
                translated_text = "\n\n" + response.text + "\n\n"
                translated_content.append(translated_text)
                request_count += 1
                translated = True
            except Exception as e:
                retries += 1
                print(f"   [오류] 단락 {i//2 + 1} 번역 중 에러 발생({retries}/{max_retries}): {e}")
                if retries < max_retries:
                    print("   -> 60초 대기 후 재시도합니다...")
                    time.sleep(60)
                else:
                    print(f"   -> {max_retries}회 시도했으나 실패하여 다음 단락으로 넘어갑니다. 해당 단락은 원본으로 대체합니다.")
                    translated_content.append(chunk)

    # 번역된 내용을 합쳐서 파일로 저장
    final_text = "".join(translated_content)
    
    base, ext = os.path.splitext(os.path.basename(filepath))
    new_filename = f"{base}_translate{ext}"
    output_path = os.path.join(TRANSLATED_DIR, new_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_text)
        
    print(f"✅ 번역 완료! 저장된 파일: {output_path}\n")

# --- 스크립트 실행 ---
if __name__ == "__main__":
    # 폴더 존재 여부 확인 및 생성
    if not os.path.exists(TRANSLATED_DIR):
        os.makedirs(TRANSLATED_DIR)
    if not os.path.exists(SOURCE_DIR):
        print(f"오류: '{SOURCE_DIR}' 폴더를 찾을 수 없습니다. 폴더를 생성하고 번역할 파일을 넣어주세요.")
        exit()
    if not all(os.path.exists(os.path.join(PROMPT_EXAMPLE_DIR, f)) for f in ["small_example_en.md", "small_example_ko.md"]):
         print(f"오류: '{PROMPT_EXAMPLE_DIR}' 폴더 안에 'small_example_en.md'와 'small_example_ko.md' 파일이 필요합니다.")
         exit()

    # 모델 초기화 (GenerationConfig 포함)
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # --- 프롬프트 예시 파일 미리 읽기 (업데이트됨) ---
    # 사용할 예시 파일 베이스 이름을 리스트로 지정합니다.
    example_bases = ["small_example"]  # 예: ["small_example", "another_example"]
    example_en_texts = []
    example_ko_texts = []
    for base in example_bases:
        en_path = os.path.join(PROMPT_EXAMPLE_DIR, f"{base}_en.md")
        ko_path = os.path.join(PROMPT_EXAMPLE_DIR, f"{base}_ko.md")
        if not os.path.exists(en_path) or not os.path.exists(ko_path):
            print(f"   [오류] 예시 파일이 누락되었습니다: {base}_en.md 또는 {base}_ko.md")
            exit()
        with open(en_path, 'r', encoding='utf-8') as f_en:
            en_content = f_en.read()
        with open(ko_path, 'r', encoding='utf-8') as f_ko:
            ko_content = f_ko.read()
        # 헤더(#) 기준 분할 및 빈문단 제거
        def split_and_clean(text):
            chunks = re.split(r'(^(?:#+ ).*$)', text, flags=re.MULTILINE)
            pars = []
            for i, chunk in enumerate(chunks):
                # 짝수 인덱스(본문)만, 공백 아닌 것만
                if i % 2 == 0 and chunk.strip():
                    pars.append(chunk.strip())
            return pars
        en_pars = split_and_clean(en_content)
        ko_pars = split_and_clean(ko_content)
        # 영어-한국어 단락 쌍 추가
        for en_par, ko_par in zip(en_pars, ko_pars):
            example_en_texts.append(en_par)
            example_ko_texts.append(ko_par)
    if not example_en_texts:
        print(f"   [오류] '{PROMPT_EXAMPLE_DIR}' 폴더에 유효한 예시 파일이 없습니다.")
        exit()
    # 시스템 지침(System Instruction) 생성
    system_instruction = create_system_instruction(example_en_texts, example_ko_texts)

    model_config = types.GenerateContentConfig(
    temperature=0,
    system_instruction=system_instruction,
    thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
    )
    
    source_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".md")]
    
    if not source_files:
        print(f"'{SOURCE_DIR}' 폴더에 번역할 Markdown 파일(.md)이 없습니다.")
    else:
        for filename in source_files:
            filepath = os.path.join(SOURCE_DIR, filename)
            translate_markdown_file(filepath, client, model_config)