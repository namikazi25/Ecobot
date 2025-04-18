import requests
from typing import Dict, Optional
import re
import time

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "EcoBot/1.0 (https://github.com/namikazi25/Ecobot; contact@ecobot.org)"
}

def search_wikipedia(query: str, sentences: int = 3) -> Dict:
    """Search Wikipedia with exponential backoff and proper error handling"""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 3,
        "srprop": "size|wordcount|timestamp",
        "srinfo": "totalhits|suggestion"
    }
    
    for attempt in range(3):
        try:
            response = requests.get(WIKIPEDIA_API, params=params, headers=HEADERS, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('query', {}).get('search'):
                return {"error": "No results found", "status": 404}
                
            best_match = data['query']['search'][0]
            return get_page_details(best_match['pageid'], sentences)
            
        except (requests.exceptions.RequestException, KeyError) as e:
            if attempt == 2:
                return {"error": f"Wikipedia API Error: {str(e)}", "status": 500}
            time.sleep(2 ** attempt)
    
    return {"error": "Unknown error", "status": 500}

def get_page_details(pageid: int, sentences: int) -> Dict:
    """Get detailed page information with section awareness"""
    params = {
        "action": "query",
        "pageids": pageid,
        "prop": "extracts|info|revisions",
        "exsentences": sentences,
        "explaintext": True,
        "inprop": "url",
        "rvprop": "timestamp",
        "format": "json"
    }
    
    try:
        response = requests.get(WIKIPEDIA_API, params=params, headers=HEADERS)
        data = response.json()
        page = data['query']['pages'][str(pageid)]
        
        return {
            "title": page["title"],
            "summary": clean_text(page.get("extract", "")),
            "url": page["fullurl"],
            "pageid": pageid,
            "last_updated": page["revisions"][0]["timestamp"],
            "wordcount": len(page.get("extract", "").split())
        }
    except Exception as e:
        return {"error": str(e), "status": 500}

def fetch_full_page(title: str) -> Dict:
    """Get full page content with table of contents"""
    params = {
        "action": "parse",
        "page": title,
        "prop": "text|sections",
        "format": "json",
        "disabletoc": 1
    }
    
    try:
        response = requests.get(WIKIPEDIA_API, params=params, headers=HEADERS)
        data = response.json()
        return {
            "content": clean_html(data["parse"]["text"]["*"]),
            "sections": [s["line"] for s in data["parse"]["sections"]],
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        }
    except Exception as e:
        return {"error": str(e), "status": 500}

def clean_html(html: str) -> str:
    """Basic HTML cleaning while preserving structure"""
    return re.sub(r'<[^>]+>', '', html)

def clean_text(text: str) -> str:
    """Clean text for GPT consumption"""
    return re.sub(r'\s+', ' ', text).strip()