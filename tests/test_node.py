def test_confidence_gate():
    from src.nodes.confidence_check_node import ConfidenceCheckNode
    gate = ConfidenceCheckNode(0.8)
    assert gate(0.5) is True
    assert gate(0.9) is False
