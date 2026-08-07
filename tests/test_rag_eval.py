import pytest

# Mock test suite evaluating RAG retrieval and answer precision for IMC Prosperity wiki rules

@pytest.fixture
def sample_ground_truth_queries():
    return [
        {
            "query": "What is the position limit for AMETHYSTS?",
            "expected_keyword": "20",
            "category": "wiki",
        },
        {
            "query": "What is the maximum position limit for STARFRUIT?",
            "expected_keyword": "20",
            "category": "wiki",
        },
        {
            "query": "How many trading rounds are in IMC Prosperity 2?",
            "expected_keyword": "5",
            "category": "wiki",
        },
    ]

def test_ground_truth_query_structure(sample_ground_truth_queries):
    assert len(sample_ground_truth_queries) == 3
    for item in sample_ground_truth_queries:
        assert "query" in item
        assert "expected_keyword" in item
        assert "category" in item

def test_mock_rag_precision_scoring(sample_ground_truth_queries):
    # Simulated top-1 retrieval evaluation
    retrieved_mock_docs = [
        "AMETHYSTS position limit is set to 20 contracts per trader.",
        "STARFRUIT position limit is capped at 20 contracts.",
        "IMC Prosperity consists of 5 main trading rounds over 10 days.",
    ]
    
    hits = 0
    for idx, item in enumerate(sample_ground_truth_queries):
        if item["expected_keyword"] in retrieved_mock_docs[idx]:
            hits += 1
            
    precision = hits / len(sample_ground_truth_queries)
    assert precision == 1.0, f"Expected 100% precision on mock benchmark, got {precision}"
