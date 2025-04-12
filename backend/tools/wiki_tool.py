import requests
from typing import Dict, Optional
import re
import time

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "EcoBot/1.0 (https://github.com/namikazi25/Ecobot; contact@ecobot.org)"
}

def search_wikipedia(query: str, sentences: int = 3) -> Dict:
    """Search Wikipedia using a query string and return the best matching result with a content summary.

    Implements exponential backoff and error handling for robustness.
    Args:
        query (str): The user search query.
        sentences (int): Number of summary sentences to request.
    Returns:
        dict: Best Wikipedia result content or an error dict if no suitable results found.
    """
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
    """Get detailed page summary and metadata for a given Wikipedia page id with a given summary length.
    Args:
        pageid (int): Wikipedia internal page ID.
        sentences (int): Number of summary sentences to extract.
    Returns:
        dict: Extract, URLs and metadata for the page, or error details.
    """
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
            "extract": page.get("extract", "No summary available."),
            "url": page.get("fullurl", ""),
            "title": page.get("title", "Unknown Page"),
            "last_revision": page.get("revisions", [{}])[0].get("timestamp", None)
        }
    except Exception as e:
        return {"error": f"Failed to get page details: {str(e)}"}

def fetch_full_page(query: str) -> Dict:
    """Return the full plain-text Wikipedia article for a given query.
    Args:
        query (str): Article title or user query string.
    Returns:
        dict: Article full extract, metadata, or user-facing error message if not found.
    """
    params = {
        "action": "query",
        "prop": "extracts|info",
        "titles": query,
        "format": "json",
        "explaintext": True,
        "exsectionformat": "plain",
        "exlimit": 1,
        "inprop": "url"
    }
    try:
        response = requests.get(WIKIPEDIA_API, params=params, headers=HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
        pages = data.get('query', {}).get('pages', {})
        page = next(iter(pages.values())) if pages else None
        if not page or "missing" in page:
            return {"error": "No Wikipedia article found for your query."}
        return {
            "extract": page.get("extract", "No article text found."),
            "fullurl": page.get("fullurl", ""),
            "title": page.get("title", "Unknown Page")
        }
    except Exception as e:
        return {"error": f"Failed fetching full Wikipedia article: {str(e)}"}

def clean_html(html: str) -> str:
    """Basic HTML cleaning while preserving structure"""
    return re.sub(r'<[^>]+>', '', html)

def clean_text(text: str) -> str:
    """Clean text for GPT consumption"""
    return re.sub(r'\s+', ' ', text).strip()