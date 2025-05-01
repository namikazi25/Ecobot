import pytest
from backend.agents.planner_langchain import PlanningChain
from backend.agents.evaluator_langchain import EvaluatingChain
from backend.agents.executor_langchain import ExecutorChain

@pytest.fixture
def planner():
    return PlanningChain()

@pytest.fixture
def evaluator():
    return EvaluatingChain()

@pytest.fixture
def executor():
    return ExecutorChain()

def test_langchain_pipeline(planner, evaluator, executor):
    """Test the LangChain-based agent pipeline for a text query."""
    query = "Tell me about the Red Fox."
    
    # Test planning phase
    plan = planner.plan(query)
    assert "tool" in plan, "Plan should contain a tool selection"
    assert "data" in plan, "Plan should contain query data"
    
    # Test evaluation phase
    evaluation = evaluator.evaluate(plan)
    assert "error" not in evaluation, "Evaluation should not contain errors"
    
    # Test execution phase
    response = executor.execute(evaluation)
    assert "response" in response, "Execution should return a response"
    assert len(response["response"]) > 0, "Response should not be empty"
    assert "sources" in response, "Response should include sources"
    assert "history" in response, "Response should include chat history"

def test_image_pipeline(planner, evaluator, executor):
    """Test the LangChain-based agent pipeline for image processing."""
    query = "What species is in this image?"
    context = {"file_type": "image/jpeg", "file_bytes": b"dummy_image_data"}
    
    plan = planner.plan(query, context=context)
    assert plan["tool"] == "image", "Should select image tool for image queries"
    
    evaluation = evaluator.evaluate(plan)
    assert "error" not in evaluation, "Evaluation should accept image processing"
    
    # Mock image data for execution
    evaluation["data"] = {"file_bytes": b"dummy_image_data", "user_query": query}
    response = executor.execute(evaluation)
    assert "response" in response, "Should process image query"

def test_wiki_pipeline(planner, evaluator, executor):
    """Test the LangChain-based agent pipeline for Wikipedia queries."""
    query = "Tell me about biodiversity in rainforests"
    
    plan = planner.plan(query)
    assert plan["tool"] in ["wiki", "wiki_full"], "Should select wiki tool for knowledge queries"
    
    evaluation = evaluator.evaluate(plan)
    assert "error" not in evaluation, "Evaluation should accept wiki queries"
    
    response = executor.execute(evaluation)
    assert "sources" in response, "Wiki response should include sources"
    assert any("wikipedia.org" in source for source in response["sources"]), "Should include Wikipedia URL"