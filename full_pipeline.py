import os
import re
import time
import requests
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types

"""
full_pipeline.py

단일 실행으로 다음을 수행:
1. source_mds 내 모든 .md 파일 반복
2. 각 파일에서 '## references' 또는 '## reference' (대소문자 무시) 헤딩 섹션만 제거 (다음 헤딩 전까지만)
3. (참조 제거된) 원본으로부터 동시에 다음 3 작업 병렬 수행
    - 번역 (기존 parallel_translate.py 와 유사한 청크 병렬 번역)
    - 메타데이터 생성 (Gemini 2.5 Pro 모델, 지정된 시스템 프롬프트)
    - 인용(MLA) 검색 (가장 첫 번째 # 헤딩을 논문 제목으로 사용, SerpAPI Google Scholar -> MLA)
4. 결과를 하나의 최종 Markdown으로 결합 후 translated_mds/<원본이름>_final.md 로 저장

최종 출력 포맷:
---
(YAML Front Matter 메타데이터)
---

> (MLA 인용)

(번역 본문)

주의: 기존 코드 파일 수정 없음. 새로운 파일만 생성.
"""

# --------------------------------------------------
# 환경 로드
# --------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수를 설정해주세요.")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  # 인용 생성을 위한 SerpAPI 키 (없으면 인용 생략)

# 모델 이름
TRANSLATION_MODEL = "gemini-2.5-flash"
METADATA_MODEL = "gemini-2.5-pro"

SOURCE_DIR = "source_mds"
OUTPUT_DIR = "translated_mds"
PROMPT_EXAMPLE_DIR = "prompt_examples"
DEFAULT_MAX_CAP = 200
MAX_RETRIES = 2
RETRY_WAIT_SECONDS = 60

# --------------------------------------------------
# 공통 유틸
# --------------------------------------------------

def read_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


# --------------------------------------------------
# 번역용 예시 로드 & 시스템 인스트럭션 (parallel_translate.py 로직 차용)
# --------------------------------------------------

def split_and_clean_example(text: str) -> List[str]:
    chunks = re.split(r'(^(?:#+ ).*$)', text, flags=re.MULTILINE)
    return [c.strip() for i, c in enumerate(chunks) if i % 2 == 0 and c.strip()]


def load_examples() -> Tuple[List[str], List[str]]:
    example_bases = ["small_example"]
    example_en_texts: List[str] = []
    example_ko_texts: List[str] = []
    for base in example_bases:
        en_path = os.path.join(PROMPT_EXAMPLE_DIR, f"{base}_en.md")
        ko_path = os.path.join(PROMPT_EXAMPLE_DIR, f"{base}_ko.md")
        if not (os.path.exists(en_path) and os.path.exists(ko_path)):
            raise FileNotFoundError(f"예시 파일 누락: {base}_en.md 또는 {base}_ko.md")
        en_pars = split_and_clean_example(read_file(en_path))
        ko_pars = split_and_clean_example(read_file(ko_path))
        for en_par, ko_par in zip(en_pars, ko_pars):
            example_en_texts.append(en_par)
            example_ko_texts.append(ko_par)
    if not example_en_texts:
        raise RuntimeError("유효한 예시 텍스트를 찾지 못했습니다.")
    return example_en_texts, example_ko_texts


def create_translation_system_instruction(example_en: List[str], example_ko: List[str]) -> str:
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
    examples = []
    for en, ko in zip(example_en, example_ko):
        examples.append(f"""
---
### Translation Example ###
[English Original]
{en}

[Korean Translation]
{ko}
""")
    footer = """
---
Now, please translate the following English text into Korean based on all the rules and the examples provided.
"""
    return header + "".join(examples) + footer

# --------------------------------------------------
# 메타데이터 시스템 프롬프트
# --------------------------------------------------
METADATA_SYSTEM_PROMPT = """
마크다운 형식의 논문 파일을 입력으로 받아 블로그 게시물에 사용할 메타데이터를 작성하세요. 출력 형식은 다음과 같습니다:

---
title: '<게시글 제목>'
description: '<게시글 설명>'
date: '2025-08-01'
tags: ['<관련 태그 1>', '<관련 태그 2>']
excerpt: '<게시글 설명>'
featuredImage: '<가장 핵심적인 이미지 주소>'
---

**목표 및 역할:**

* 사용자가 제공한 마크다운 논문 파일에서 블로그 게시물에 필요한 메타데이터를 추출하고 생성합니다.
* 제시된 형식에 맞춰 'title', 'description', 'date', 'tags', 'excerpt', 'featuredImage' 필드를 정확하게 채웁니다.
* 'title'은 논문의 제목을 기반으로 블로그 게시물에 적합하게 생성합니다.
* 'description'은 논문의 핵심 내용을 요약하여, 사용자가 논문을 읽기전에 description을 통해 전체적인 내용을 바로 쉽게 파악할 수 있도록 작성합니다. 마지막에는 논문 제목: <논문 제목> 형식으로 영어 원어 그대로 논문 제목을 덧붙여주세요.
* 'date'는 항상 '2025-08-01'로 설정합니다.
* 'tags'는 논문의 핵심 키워드나 주제를 파악하여 1~2개의 관련 태그를 배열 형식으로 생성합니다. (영어로 작성)
* 'excerpt'는 'description'과 동일한 내용을 작성합니다.
* 'featuredImage'는 마크다운 파일 내의 이미지 주소 중 논문의 핵심 내용을 가장 잘 대표하거나 시각적으로 중요한 이미지를 찾아 해당 URL을 제공합니다. 만약 적절한 이미지가 없으면 해당 필드를 비워둡니다.
* AI 논문의 전문적인 용어(예: Transformer, Contrastive loss, Cross attention 등)는 영어 원어를 유지합니다. 가능한, 용어는 영어원어를 유지해주세요.

**행동 및 규칙:**

1. **입력 처리:** 사용자가 마크다운 논문 파일을 입력하면, 해당 파일을 분석하여 필요한 정보를 추출합니다.
2. **메타데이터 생성:** 추출된 정보를 바탕으로 위에서 정의된 형식에 맞춰 각 메타데이터 필드의 내용을 생성합니다.
* 'title': 논문 제목을 직관적으로 작성합니다. 독자들이 제목을 보자마자 무슨 논문인지 알 수 있도록 작성합니다.
* 'description': 논문의 핵심 요약을 제공하여 독자의 흥미를 유발합니다. (한글로 적되, 전문적인 영어 용어는 번역하지 않고 원어 그대로 유지, 마지막 부분의 논문 제목은 영어 원어 그대로 유지)
* 'tags': 논문의 주요 개념, 연구 분야, 방법론 등을 기반으로 관련성 높은 태그를 선정합니다. (영어로 작성)
* 'excerpt': 'description'과 동일한 내용을 작성합니다.
* 'featuredImage': 마크다운 내의 모든 이미지 URL을 검토하여, 논문의 주제와 가장 밀접하게 관련된 시각 자료를 대표하는 이미지를 선택합니다. 이 필드는 반드시 따옴표로 묶인 URL 문자열이어야 합니다.
3. **출력 형식 준수:** 생성된 메타데이터를 정확히 제시된 YAML Front Matter 형식(---로 시작하고 끝남)으로 출력합니다. 모든 문자열 값은 작은따옴표로 묶어야 합니다.
4. **응답 상세화:** 메타데이터 생성 시, 각 필드가 왜 그렇게 채워졌는지에 대한 간략한 설명을 추가할 필요는 없으며, 오직 최종 메타데이터 형식만 출력합니다.

**출력 예시:**

---
title: 'Flamingo: a Visual Language Model for Few-Shot Learning'
description: 'Flamingo는 이미지와 텍스트가 혼합된 입력을 처리할 수 있으며, few-shot 학습 환경에서도 높은 성능을 보이는 Visual Language Model (VLM)이다. Flamingo는 pretrained된 vision-only 및 language-only 모델을 효과적으로 연결하고, 임의의 순서로 interleaved된 이미지 및 텍스트 시퀀스를 처리할 수 있도록 설계되었다. 이 모델은 이미지와 텍스트가 섞인 대규모 웹 데이터로 학습되며, in-context few-shot 학습 능력을 통해 다양한 multimodal task (예: visual question answering, image captioning 등)에 빠르게 적응하는 성능을 보여준다. 논문 제목: Flamingo: a Visual Language Model for Few-Shot Learning'
date: '2025-08-01'
tags: ['Visual Language Model', 'Few-shot Learning']
excerpt: 'Flamingo는 이미지와 텍스트가 혼합된 입력을 처리할 수 있으며, few-shot 학습 환경에서도 높은 성능을 보이는 Visual Language Model (VLM)이다. Flamingo는 pretrained된 vision-only 및 language-only 모델을 효과적으로 연결하고, 임의의 순서로 interleaved된 이미지 및 텍스트 시퀀스를 처리할 수 있도록 설계되었다. 이 모델은 이미지와 텍스트가 섞인 대규모 웹 데이터로 학습되며, in-context few-shot 학습 능력을 통해 다양한 multimodal task (예: visual question answering, image captioning 등)에 빠르게 적응하는 성능을 보여준다. 논문 제목: Flamingo: a Visual Language Model for Few-Shot Learning'
featuredImage: 'https://cdn.mathpix.com/cropped/2025_07_26_7c316185968e7585aacbg-02.jpg?height=2242&width=1403&top_left_y=180&top_left_x=361'
---
"""

# --------------------------------------------------
# 레퍼런스 제거
# --------------------------------------------------

def remove_references_section(md: str) -> str:
    """'## references' 또는 '## reference' 헤딩으로 시작하는 섹션만 제거.

    - 대소문자 무시
    - 헤딩 라인과 그 *다음 첫 번째 헤딩(임의의 #) 직전까지*의 내용만 제거
    - 뒤에 Appendix 등 다른 헤딩 블록은 보존
    - 여러 개 존재하면 모두 제거 (안전성 위해 반복 처리)
    """
    lines = md.splitlines()
    out: List[str] = []
    i = 0
    ref_header_pattern = re.compile(r'^\s{0,3}#{2,6}\s*references?\s*$', re.IGNORECASE)
    any_header_pattern = re.compile(r'^\s{0,3}#+\s')
    while i < len(lines):
        line = lines[i]
        if ref_header_pattern.match(line):
            # Skip this references header line
            i += 1
            # Skip until next heading (any level)
            while i < len(lines) and not any_header_pattern.match(lines[i]):
                i += 1
            continue  # do not append skipped lines
        out.append(line)
        i += 1
    return "\n".join(out).rstrip() + "\n"

# --------------------------------------------------
# 첫 메인 제목(# ) 추출
# --------------------------------------------------

def extract_main_title(md: str) -> Optional[str]:
    for line in md.splitlines():
        if line.strip().startswith('# '):
            return line.strip()[2:].strip()
    return None

# --------------------------------------------------
# 인용 (SerpAPI Google Scholar -> MLA)
# --------------------------------------------------

def get_mla_citation(title: str) -> Optional[str]:
    """주어진 논문 제목으로 Google Scholar 검색 후 MLA 인용문 반환.

    흐름:
      1) google_scholar 엔진으로 검색 → 첫 organic result 추출
      2) result_id 기반 google_scholar_cite 호출 (없으면 inline_links.serpapi_cite_link 사용)
      3) citations 배열에서 title == 'MLA' 인 snippet 반환
    SERPAPI_API_KEY 가 없거나 오류 발생 시 None 반환.
    """
    if not SERPAPI_API_KEY:
        return None
    try:
        search_params = {
            "engine": "google_scholar",
            "q": title,
            "hl": "en",
            "api_key": SERPAPI_API_KEY,
        }
        r = requests.get("https://serpapi.com/search", params=search_params, timeout=30)
        r.raise_for_status()
        data = r.json()
        organic = data.get("organic_results", [])
        if not organic:
            return None
        top = organic[0]
        result_id = top.get("result_id")
        if result_id:
            cite_params = {
                "engine": "google_scholar_cite",
                "q": result_id,
                "hl": "en",
                "api_key": SERPAPI_API_KEY,
            }
            r2 = requests.get("https://serpapi.com/search", params=cite_params, timeout=30)
        else:
            cite_link = (top.get("inline_links") or {}).get("serpapi_cite_link")
            if not cite_link:
                return None
            connector = "&" if "?" in cite_link else "?"
            r2 = requests.get(f"{cite_link}{connector}api_key={SERPAPI_API_KEY}", timeout=30)
        r2.raise_for_status()
        cite_data = r2.json()
        for c in cite_data.get("citations", []):
            if (c.get("title") or "").strip().upper() == "MLA":
                snippet = (c.get("snippet") or "").strip()
                return snippet or None
        return None
    except Exception:
        return None

# --------------------------------------------------
# 번역 관련 분할
# --------------------------------------------------

def split_markdown(content: str) -> List[str]:
    return re.split(r'(^(?:#+ ).*$)', content, flags=re.MULTILINE)


def needs_translation(i: int, chunk: str) -> bool:
    if i % 2 == 1:  # 헤더
        return False
    return bool(chunk.strip())

# --------------------------------------------------
# 번역 청크 호출
# --------------------------------------------------

def translate_chunk(client: genai.Client, model_config: types.GenerateContentConfig, idx: int, text: str):
    attempt = 0
    while attempt <= MAX_RETRIES:
        try:
            resp = client.models.generate_content(
                model=TRANSLATION_MODEL,
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
# 파일 단위 번역 (청크 동시 제출)
# --------------------------------------------------

def translate_content_all_at_once(md: str, client: genai.Client, model_config: types.GenerateContentConfig) -> str:
    chunks = split_markdown(md)
    targets = [(i, c) for i, c in enumerate(chunks) if needs_translation(i, c)]
    if not targets:
        return md
    max_workers = min(len(targets), DEFAULT_MAX_CAP) if targets else 1
    results: Dict[int, str] = {}
    submitted = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(translate_chunk, client, model_config, i, c) for i, c in targets]
        for fut in as_completed(futures):
            idx, translated = fut.result()
            results[idx] = translated
            submitted += 1
            if submitted % 5 == 0 or submitted == len(targets):
                print(f"      번역 진행률 {submitted}/{len(targets)} ({submitted/len(targets)*100:.1f}%)")
    final_parts: List[str] = []
    for i, original in enumerate(chunks):
        final_parts.append(results.get(i, original))
    return "".join(final_parts)

# --------------------------------------------------
# 메타데이터 생성
# --------------------------------------------------

def generate_metadata(client: genai.Client, markdown: str) -> str:
    try:
        resp = client.models.generate_content(
            model=METADATA_MODEL,
            contents=markdown + "\n\n메타데이터 작성해줘",
            config=types.GenerateContentConfig(
                temperature=1.0,
                system_instruction=METADATA_SYSTEM_PROMPT,
                thinking_config=types.ThinkingConfig(thinking_budget=-1)
            )
        )
        meta = resp.text.strip()
        # front matter 보정
        if '---' not in meta:
            meta = f"---\n{meta}\n---"
        else:
            # 시작과 끝이 --- 로 감싸져있는지 확인
            if not meta.startswith('---'):
                meta = '---\n' + meta
            if not re.search(r'\n---\s*$', meta):
                meta = meta.rstrip() + '\n---'
        return meta
    except Exception as e:
        print(f"[메타데이터 실패] {e}")
        return "---\ntitle: 'N/A'\ndescription: 'N/A'\ndate: '2025-08-01'\ntags: []\nexcerpt: 'N/A'\nfeaturedImage: ''\n---"

# --------------------------------------------------
# 개별 파일 처리
# --------------------------------------------------

def process_file(path: str, translation_client: genai.Client, translation_config: types.GenerateContentConfig):
    name = os.path.basename(path)
    base, ext = os.path.splitext(name)
    print(f"\n📄 처리 시작: {name}")
    original_md = read_file(path)
    cleaned_md = remove_references_section(original_md)
    main_title = extract_main_title(original_md) or base

    # 병렬: 번역 / 메타데이터 / 인용
    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_translation = ex.submit(translate_content_all_at_once, cleaned_md, translation_client, translation_config)
        fut_metadata = ex.submit(generate_metadata, translation_client, cleaned_md)
        fut_citation = ex.submit(get_mla_citation, main_title)

        translated_text = fut_translation.result()
        metadata_text = fut_metadata.result()
        citation_text = fut_citation.result()

    if not citation_text:
        citation_text = f"Citation not found for title: {main_title}"

    final_output = f"{metadata_text}\n\n> {citation_text}\n{translated_text}".rstrip() + "\n"

    out_path = os.path.join(OUTPUT_DIR, f"{base}_final{ext}")
    write_file(out_path, final_output)
    print(f"✅ 완료: {out_path}")

# --------------------------------------------------
# 메인
# --------------------------------------------------

def main():
    if not os.path.exists(SOURCE_DIR):
        raise FileNotFoundError(f"'{SOURCE_DIR}' 폴더가 없습니다.")

    en_examples, ko_examples = load_examples()
    translation_system_instruction = create_translation_system_instruction(en_examples, ko_examples)

    translation_client = genai.Client(api_key=GOOGLE_API_KEY)
    translation_config = types.GenerateContentConfig(
        temperature=0,
        system_instruction=translation_system_instruction,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.md')]
    files.sort()
    if not files:
        print("번역할 파일이 없습니다.")
        return

    print(f"총 {len(files)}개 파일 처리. (번역 청크 동시 상한 {DEFAULT_MAX_CAP})")

    for fname in files:
        process_file(os.path.join(SOURCE_DIR, fname), translation_client, translation_config)


if __name__ == '__main__':
    main()
