# Paper Translate

Google Gemini API를 사용한 학술 논문 Markdown 번역 도구

## 폴더 구조

```
paper_translate/
├── README.md
├── translate.py              # 메인 번역 스크립트
├── prompt_examples/          # 번역 예시 파일들
│   ├── example_en.md         # 영어 예시
│   ├── example_ko.md         # 한국어 예시
│   ├── small_example_en.md   # 작은 영어 예시
│   └── small_example_ko.md   # 작은 한국어 예시
├── source_mds/               # 번역할 원본 마크다운 파일
├── source_mds_complete/      # 완료된 원본 파일 보관
└── translated_mds/           # 번역 완료된 파일 저장소
```

## 사용법

1. 필요한 라이브러리 설치
   ```bash
   pip install -q -U google-genai
   pip install python-dotenv
   ```

2. `.env` 파일에 Google API 키 설정
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

3. 번역할 마크다운 파일을 `source_mds/` 폴더에 넣기

4. 번역 실행
   ```bash
   python translate.py
   ```

5. 번역된 파일은 `translated_mds/` 폴더에 저장됨

## 특징

- 학술 논문 전용 번역 (기술 용어는 영어 유지)
- 헤더 기준으로 단락 분할하여 번역
- API 속도 제한 자동 제어
- 번역 예시를 참고하여 일관된 스타일 유지
- **Google Gemini API 무료 사용** (일일 제한 내에서)

## 적용 계획

이 번역 방법을 [junhan.blog](https://junhan.blog) 사이트에 차차 적용할 예정입니다.
현재 일부 포스트는 이미 적용되어 있으며, 나머지 포스트들도 순차적으로 번역하여 업데이트할 계획입니다.