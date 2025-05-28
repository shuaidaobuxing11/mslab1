import unittest
import networkx as nx

# 被测试函数
def query_bridge_words(graph, word1, word2):
    if not hasattr(graph, "neighbors"):
        raise TypeError("graph must be a valid graph with 'neighbors' method")
    if not isinstance(word1, str) or not isinstance(word2, str):
        raise TypeError("word1 and word2 must be strings")
    
    bridge_words = []
    if word1 not in graph or word2 not in graph:
        return f"No {word1} or {word2} in the graph!"
    
    for neighbor in graph.neighbors(word1):
        if graph.has_edge(neighbor, word2):
            bridge_words.append(neighbor)
    
    if not bridge_words:
        return f"No bridge words from {word1} to {word2}!"
    
    return bridge_words

# 测试类
class TestQueryBridgeWords(unittest.TestCase):

    def test_valid_with_bridge(self):
        # 用例 1：存在 bridge word（E1）
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c")])
        result = query_bridge_words(graph, "a", "c")
        self.assertEqual(result, ["b"])

    def test_valid_no_bridge(self):
        # 用例 2：不存在 bridge word（E2）
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("c", "d")])
        result = query_bridge_words(graph, "a", "d")
        self.assertEqual(result, "No bridge words from a to d!")

    def test_word_not_in_graph(self):
        # 用例 3：word1 不在图中（I3）
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b")])
        result = query_bridge_words(graph, "x", "b")
        self.assertEqual(result, "No x or b in the graph!")

    def test_invalid_graph_and_type(self):
    # 非图对象
        graph = [("a", "b")]  # list，没有 neighbors 方法
        with self.assertRaises(TypeError):
            query_bridge_words(graph, "a", "b")
    
    # graph 合法，但 word1 不是字符串
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b")])
        with self.assertRaises(TypeError):
            query_bridge_words(graph, 123, "b")
        
        # graph 合法，但 word2 不是字符串
        with self.assertRaises(TypeError):
            query_bridge_words(graph, "a", None)

