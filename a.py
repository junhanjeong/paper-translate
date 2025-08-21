import requests
from urllib.parse import quote

def get_mla_citation(title):
    # 1) 논문 제목으로 Crossref API 검색 (최대 1개 결과)
    search_url = f"https://api.crossref.org/works?query.bibliographic={quote(title)}&rows=1"
    res = requests.get(search_url)
    res.raise_for_status()
    data = res.json()

    items = data.get("message", {}).get("items", [])
    if not items:
        return None

    doi = items[0].get("DOI")
    if not doi:
        return None

    # 2) DOI로 MLA 형식 인용문 받아오기
    cite_url = f"https://citation.doi.org/format?doi={doi}&style=modern-language-association&lang=en-US"
    res2 = requests.get(cite_url, headers={"Accept": "text/x-bibliography; charset=utf-8"})
    res2.raise_for_status()
    return res2.text.strip()

# 사용 예시
title = "GPT-4 Technical Report"
citation = get_mla_citation(title)
print(citation)
