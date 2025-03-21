# backend/tools/valyu_tool.py
from valyu import Valyu
import os

VALYU_API_KEY = os.getenv("VALYU_API_KEY")
print(VALYU_API_KEY)
class ValyuClient:
    def __init__(self):
        self.client = Valyu(api_key="PgKSGjVQ7s17Wpzmh1thV3AniBYaDtae25J5vPcA")
    
    def search_context(self, query: str, max_price: float = 5.0) -> dict:
        try:
            response = self.client.context(
                query=query,
                search_type="all",
                max_num_results=3,
                similarity_threshold=0.5,
                max_price=1
            )
            return self._format_response(response)
        except Exception as e:
            return {"error": str(e)}

    def _format_response(self, raw_response: dict) -> dict:
        return {
            "content": "\n\n".join([f"{r.title}: {r.content}" for r in raw_response.results]),
            "sources": [r.url for r in raw_response.results],
            "cost": raw_response.total_deduction_dollars
        }