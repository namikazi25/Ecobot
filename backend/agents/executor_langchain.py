from typing import Optional, List, Dict, Any
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredImageLoader
from ...core.llm_factory import get_llm
from ..tools.wiki_tool import search_wikipedia, fetch_full_page

class ExecutorChain:
    """LangChain-based executor that processes requests using appropriate tools."""
    
    def __init__(self):
        self.llm = get_llm()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        
        # Initialize tool-specific prompts
        self.image_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    """You are an expert in species identification and ecological analysis.
                    Analyze the provided image and answer the user's query with scientific accuracy."""
                ),
                HumanMessagePromptTemplate.from_template(
                    "Image content: {image_text}\nQuery: {query}"
                )
            ]
        )
        
        self.pdf_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    """You are a scientific document analyzer.
                    Review the provided PDF content and answer the user's query."""
                ),
                HumanMessagePromptTemplate.from_template(
                    "Document content: {pdf_text}\nQuery: {query}"
                )
            ]
        )
    
    def execute(self, plan: Dict[str, Any], history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Executes the validated plan using LangChain components."""
        history = history or []
        response = {"response": "", "sources": []}
        
        try:
            tool = plan.get("tool")
            data = plan.get("data")
            file_type = plan.get("file_type")
            
            if tool == "gpt":
                chain = LLMChain(llm=self.llm, prompt=ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        """You are EcoBot, an AI-powered ecological assistant.
                        Provide scientific and informative responses about biodiversity and ecosystems."""
                    ),
                    HumanMessagePromptTemplate.from_template("{query}")
                ]))
                result = chain.invoke({"query": data})
                response["response"] = result["text"]
            
            elif tool == "image":
                if not data.get("file_bytes") or not data.get("user_query"):
                    return {"response": "❌ Missing image data or query", "sources": []}
                    
                # Process image using LangChain's document loader
                loader = UnstructuredImageLoader(file_bytes=data["file_bytes"])
                image_doc = loader.load()[0]
                
                # Create and run the image analysis chain
                image_chain = LLMChain(llm=self.llm, prompt=self.image_prompt)
                result = image_chain.invoke({
                    "image_text": image_doc.page_content,
                    "query": data["user_query"]
                })
                response["response"] = result["text"]
            
            elif tool == "pdf":
                if not data.get("extracted_text"):
                    return {"response": "❌ Missing PDF content", "sources": []}
                
                # Split PDF content into manageable chunks
                docs = self.text_splitter.create_documents([data["extracted_text"]])
                
                # Create and run the PDF analysis chain
                pdf_chain = LLMChain(llm=self.llm, prompt=self.pdf_prompt)
                result = pdf_chain.invoke({
                    "pdf_text": "\n".join([doc.page_content for doc in docs]),
                    "query": data.get("user_query", "Summarize this document.")
                })
                response["response"] = result["text"]
            
            elif tool == "wiki":
                result = search_wikipedia(data)
                if "error" in result:
                    return self._fallback_response(data)
                response["response"] = self._format_wiki_summary(result)
                response["sources"] = [result["url"]]
            
            elif tool == "wiki_full":
                result = fetch_full_page(data)
                if "error" in result:
                    return self._fallback_response(data)
                response["response"] = self._format_full_wiki(result)
                response["sources"] = [result["url"]]
            
            else:
                response["response"] = "❌ Unknown tool selected"
            
        except Exception as e:
            response["response"] = f"⚠️ Execution Error: {str(e)}"
        
        # Update chat history
        history.append({"role": "assistant", "content": response["response"]})
        return {**response, "history": history}
    
    def _format_wiki_summary(self, result: Dict[str, str]) -> str:
        return f"""🌿 **{result['title']}**  
{result['summary']}  
📅 Last Updated: {result['last_updated'][:10]}  
🔗 [Read More]({result['url']})"""
    
    def _format_full_wiki(self, result: Dict[str, str]) -> str:
        return f"""📖 **Full Article**: {result['url']}  
{result['content'][:2500]}...  
**Sections**: {', '.join(result['sections'][:5])}"""
    
    def _fallback_response(self, query: str) -> Dict[str, Any]:
        chain = LLMChain(llm=self.llm, prompt=ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Provide a response based on your knowledge since Wikipedia lookup failed."
            ),
            HumanMessagePromptTemplate.from_template("{query}")
        ]))
        result = chain.invoke({"query": query})
        return {
            "response": f"❌ Wikipedia lookup failed. Alternative response:\n{result['text']}",
            "sources": [],
            "history": []
        }