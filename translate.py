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
MODEL_NAME = "gemini-2.5-flash-lite"
SOURCE_DIR = "source_mds"
TRANSLATED_DIR = "translated_mds"
PROMPT_EXAMPLE_DIR = "prompt_examples"
REQUESTS_PER_MINUTE = 15
SECONDS_TO_WAIT = 60

# --- 프롬프트 생성 함수 (업데이트됨) ---
def create_system_instruction(example_en_text, example_ko_text):
    """번역 작업을 위한 시스템 지침(System Instruction)을 생성합니다."""
    # 사용자 맞춤 지시사항 반영
    return f"""
You are a specialized translator for academic papers, translating Markdown documents from English to Korean.
Follow these rules meticulously:
1.  Translate the main content into professional, natural-sounding Korean.
2.  **Crucially, do not translate technical English terms.** Keep terms like 'Transformer', 'DETR', 'cross attention', 'encoder', 'decoder', etc., in their original English form.
3.  **Ensure that figure links are preserved.** Do not omit any figure URLs or accompanying descriptions.
4.  **Include references formatted as [1], but when encountering footnote markers like [^1], remove only the [^1] marker while keeping the adjacent explanation text intact.**
5.  **Use the provided example as a reference for tone, translation style, and formatting.** This includes matching whitespace, line breaks, and paragraph divisions exactly as shown in the example.
6.  **Do not create any new titles or headings; translate only the content you’ve been given.**
7.  **In the translated text, enhance readability by using bold formatting for important terms or key phrases, as shown in the [Korean Translation] of the translation example.**

---
### Translation Example ###

[English Original]
{example_en_text}

[Korean Translation]
{example_ko_text}
---

Now, please translate the following English text into Korean based on all the rules and the example provided.
"""

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

    # 프롬프트 예시 파일 미리 읽기
    try:
        with open(os.path.join(PROMPT_EXAMPLE_DIR, "small_example_en.md"), 'r', encoding='utf-8') as f_en:
            example_en = f_en.read()
        with open(os.path.join(PROMPT_EXAMPLE_DIR, "small_example_ko.md"), 'r', encoding='utf-8') as f_ko:
            example_ko = f_ko.read()
    except FileNotFoundError:
        print(f"   [오류] '{PROMPT_EXAMPLE_DIR}' 폴더에서 예시 파일을 찾을 수 없습니다.")
        
    # 시스템 지침(System Instruction)을 한 번만 생성
    system_instruction = create_system_instruction(example_en, example_ko)

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