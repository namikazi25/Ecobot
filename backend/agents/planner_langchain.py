from typing import Dict, Any, List, Optional
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from core.llm_factory import get_llm

class PlanSchema(BaseModel):
    """Schema for the planning output"""
    tool: str = Field(description="The tool to use (gpt, image, pdf, wiki, or wiki_full)")
    data: str = Field(description="The data or query to process")
    file_type: Optional[str] = Field(None, description="File type for image/pdf processing")

class PlanningChain:
    """LangChain-based planner that determines the best tool for a given query."""
    
    def __init__(self):
        self.llm = get_llm()
        self.output_parser = PydanticOutputParser(pydantic_object=PlanSchema)
        
        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    """You are a planning agent responsible for selecting the appropriate tool for user queries.
                    Available tools:
                    - gpt: For general text-based queries and conversations
                    - image: For image analysis and species identification
                    - pdf: For processing and analyzing PDF documents
                    - wiki: For retrieving Wikipedia summaries
                    - wiki_full: For fetching complete Wikipedia articles
                    
                    {format_instructions}
                    """
                ),
                HumanMessagePromptTemplate.from_template(
                    """Query: {query}
                    Context: {context}
                    History: {history}
                    
                    Select the most appropriate tool and format the response according to the schema.
                    """
                )
            ]
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "No recent interactions"
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-5:]])
    
    def plan(self, query: str, context: Optional[Dict[str, Any]] = None, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Creates a plan using LangChain components."""
        try:
            # Prepare inputs
            context = context or {}
            history = history or []
            
            # Get the formatting instructions
            format_instructions = self.output_parser.get_format_instructions()
            
            # Execute the chain
            result = self.chain.invoke({
                "query": query,
                "context": str(context),
                "history": self._format_history(history),
                "format_instructions": format_instructions
            })
            
            # Parse the output
            plan = self.output_parser.parse(result["text"])
            return plan.dict()
            
        except Exception as e:
            return {"error": f"❌ Planning error: {str(e)}"}