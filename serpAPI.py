import os
from dotenv import load_dotenv
import requests
from typing import Optional

load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  # 환경변수로 관리 권장

def get_mla_citation(title: str) -> Optional[str]:
    """
    주어진 논문 제목을 Google Scholar 상단 결과로 검색하고,
    해당 결과의 MLA 인용문을 반환한다.
    SerpAPI 사용.
    """
    try:
        # 1) 제목으로 Google Scholar 검색 → 첫 결과의 result_id 추출
        search_params = {
            "engine": "google_scholar",
            "q": title,
            "hl": "en",
            "api_key": SERPAPI_API_KEY,
            # "no_cache": "true",  # 필요 시 캐시 비활성화
        }
        r = requests.get("https://serpapi.com/search", params=search_params, timeout=30)
        r.raise_for_status()
        data = r.json()

        organic = data.get("organic_results", [])
        if not organic:
            return None

        top = organic[0]
        result_id = top.get("result_id")

        # 2) result_id로 Google Scholar Cite API 호출 → MLA snippet 추출
        if result_id:
            cite_params = {
                "engine": "google_scholar_cite",
                "q": result_id,
                "hl": "en",
                "api_key": SERPAPI_API_KEY,
            }
            r2 = requests.get("https://serpapi.com/search", params=cite_params, timeout=30)
        else:
            # fallback: inline_links.serpapi_cite_link가 있으면 그대로 호출
            cite_link = (top.get("inline_links") or {}).get("serpapi_cite_link")
            if not cite_link:
                return None
            # cite_link는 api_key를 포함하지 않으므로 추가
            connector = "&" if "?" in cite_link else "?"
            r2 = requests.get(f"{cite_link}{connector}api_key={SERPAPI_API_KEY}", timeout=30)

        r2.raise_for_status()
        cite_data = r2.json()

        for c in cite_data.get("citations", []):
            if (c.get("title") or "").strip().upper() == "MLA":
                snippet = (c.get("snippet") or "").strip()
                return snippet if snippet else None

        return None
    except Exception:
        return None

# 사용 예시
title = "Object-Centric Learning with Slot Attention"
citation = get_mla_citation(title)
print(citation)