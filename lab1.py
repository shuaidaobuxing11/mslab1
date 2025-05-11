import re
import random
import networkx as nx
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from collections import defaultdict
import matplotlib as mpl

import msvcrt  # 用于检测键盘输入
import math




# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 用于读取文本并构建有向图的函数
def read_file_and_build_graph(filename):
    # 创建一个有向图
    graph = nx.DiGraph()

    # 读取文本文件
    with open(filename, 'r', encoding='utf-8') as file:  # 添加编码支持
        text = file.read()

    # 将所有的非字母字符替换为空格，并将文本转为小写
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()

    # 按空格分割文本成单词
    words = text.split()

    # 遍历文本中的所有相邻单词，构建图
    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]
        
        # 如果这两个单词在图中不存在，先加入它们
        if not graph.has_node(word1):
            graph.add_node(word1)
        if not graph.has_node(word2):
            graph.add_node(word2)
        
        # 如果图中已经有从word1到word2的边，增加权重，否则建立新边
        if graph.has_edge(word1, word2):
            graph[word1][word2]['weight'] += 1
        else:
            graph.add_edge(word1, word2, weight=1)

    print('生成完毕')
    return graph

# 展示有向图并保存为文件，并高亮显示最短路径
def show_directed_graph(graph, shortest_paths=None, filename="directed_graph.png"):
    # 如果图太大，只显示权重较大的边
    if graph.number_of_edges() > 100:
        edge_weights = nx.get_edge_attributes(graph, 'weight')
        sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)
        top_edges = [edge for edge, _ in sorted_edges[:100]]
        subgraph = graph.edge_subgraph(top_edges)
    else:
        subgraph = graph
    
    pos = nx.spring_layout(subgraph, k=1, iterations=50, seed=42)
    plt.figure(figsize=(20, 16))
    plt.gca().set_facecolor('#f0f0f0')
    
    # 绘制基本的有向图
    nx.draw(subgraph, pos,
            with_labels=True,
            node_size=2000,
            node_color='#A0CBE2',
            font_size=8,
            font_weight='bold',
            font_color='black',
            arrows=True,
            arrowsize=15,
            edge_color='gray',
            width=1.0,
            alpha=0.7)
    
    # 添加边权重标签
    edge_labels = nx.get_edge_attributes(subgraph, 'weight')
    nx.draw_networkx_edge_labels(subgraph, pos,
                                edge_labels=edge_labels,
                                font_size=6,
                                font_color='red')
    
    if shortest_paths:
        # 定义更多的颜色
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', 
                 '#D4A5A5', '#9B59B6', '#3498DB', '#E67E22', '#2ECC71',
                 '#1ABC9C', '#F1C40F', '#E74C3C', '#34495E', '#16A085',
                 '#D35400', '#8E44AD', '#27AE60', '#2980B9', '#C0392B']
        
        # 为每条路径使用不同的颜色
        for i, path in enumerate(shortest_paths):
            color = colors[i % len(colors)]
            edges = list(zip(path[:-1], path[1:]))
            
            # 绘制路径的边
            nx.draw_networkx_edges(subgraph, pos,
                                 edgelist=edges,
                                 width=2,
                                 edge_color=color,
                                 alpha=0.8,
                                 arrows=True,
                                 arrowsize=20)
            
            # 绘制路径的节点
            nx.draw_networkx_nodes(subgraph, pos,
                                 nodelist=path,
                                 node_size=2000,
                                 node_color=color,
                                 alpha=0.8)
            
            # 为路径上的节点添加标签
            nx.draw_networkx_labels(subgraph, pos,
                                  {node: node for node in path},
                                  font_size=10,
                                  font_weight='bold',
                                  font_color='black')
        
        # 添加图例
        legend_elements = [plt.Line2D([0], [0], color=colors[i % len(colors)], 
                                    label=f'路径 {i+1}', linewidth=2)
                          for i in range(len(shortest_paths))]
        plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("文本分析有向图\n(红色数字表示相邻出现次数，不同颜色表示不同路径)", 
              fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"图形已保存为 {filename}")
    plt.show()

# 查询桥接词
def query_bridge_words(graph, word1, word2):
    bridge_words = []
    if word1 not in graph or word2 not in graph:
        return f"No {word1} or {word2} in the graph!"
    
    for neighbor in graph.neighbors(word1):
        if graph.has_edge(neighbor, word2):
            bridge_words.append(neighbor)
    
    if not bridge_words:
        return f"No bridge words from {word1} to {word2}!"
    
    return bridge_words

# 生成新文本，插入桥接词
def generate_new_text(graph, input_text):
    words = input_text.split()
    new_text = []
    
    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]
        
        bridge_words = query_bridge_words(graph, word1, word2)
        if "No bridge words" in bridge_words:
            new_text.append(word1)
        else:
            # 随机选择一个桥接词插入
            bridge_word = random.choice(bridge_words.split(":")[1].strip().split(","))
            new_text.append(word1)
            new_text.append(bridge_word)
    
    new_text.append(words[-1])
    return " ".join(new_text)

# 自定义Dijkstra算法实现，支持多条最短路径
def dijkstra(graph, start, end=None):
    # 初始化距离字典和父节点字典
    distances = {node: float('infinity') for node in graph.nodes()}
    distances[start] = 0
    previous = {node: [] for node in graph.nodes()}  # 改为列表以存储多个父节点
    unvisited = set(graph.nodes())
    
    while unvisited:
        # 找到未访问节点中距离最小的节点
        current = min(unvisited, key=lambda node: distances[node])
        
        # 如果当前节点到起点的距离是无穷大，说明无法到达
        if distances[current] == float('infinity'):
            break
            
        # 如果找到了目标节点，可以提前结束
        if end and current == end:
            break
            
        unvisited.remove(current)
        
        # 更新邻居节点的距离
        for neighbor in graph.neighbors(current):
            if neighbor in unvisited:
                # 获取边的权重，如果没有权重则默认为1
                weight = graph[current][neighbor].get('weight', 1)
                new_distance = distances[current] + weight
                
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = [current]  # 重置为新的最短路径
                elif new_distance == distances[neighbor]:
                    previous[neighbor].append(current)  # 添加另一条最短路径
    
    # 如果指定了终点，返回从起点到终点的所有最短路径
    if end:
        if distances[end] == float('infinity'):
            return None
            
        def build_paths(node, path, paths):
            if node == start:
                paths.append(path)
                return
            for prev in previous[node]:
                build_paths(prev, [prev] + path, paths)
        
        paths = []
        build_paths(end, [end], paths)
        return paths
    
    # 如果没有指定终点，返回所有路径
    all_paths = {}
    for node in graph.nodes():
        if node != start and distances[node] != float('infinity'):
            paths = []
            def build_all_paths(current, path):
                if current == start:
                    paths.append(path)
                    return
                for prev in previous[current]:
                    build_all_paths(prev, [prev] + path)
            
            build_all_paths(node, [node])
            all_paths[node] = paths
    
    return all_paths

# 计算最短路径
def calc_shortest_path(graph, word1, word2=None):
    if word1 not in graph:
        return f"单词 '{word1}' 不在图中"
    
    if word2 is None:
        # 计算到所有其他节点的最短路径
        all_paths = dijkstra(graph, word1)
        if all_paths is None:
            return f"无法从 '{word1}' 到达任何其他节点"
        return all_paths
    else:
        if word2 not in graph:
            return f"单词 '{word2}' 不在图中"
        
        # 计算两个单词之间的所有最短路径
        paths = dijkstra(graph, word1, word2)
        if paths is None:
            return f"从 {word1} 到 {word2} 没有路径"
        return paths

# 计算TF-IDF值
def calculate_tfidf(graph):
    # 获取所有节点（单词）
    nodes = list(graph.nodes())
    n = len(nodes)
    
    # 计算每个单词的TF（词频）
    tf = {}
    for node in nodes:
        # 计算单词的入度和出度之和作为词频
        tf[node] = graph.in_degree(node) + graph.out_degree(node)
    
    # 计算每个单词的IDF（逆文档频率）
    idf = {}
    for node in nodes:
        # 计算包含该单词的文档数（即该单词的邻居节点数）
        docs_with_word = len(list(graph.neighbors(node))) + len(list(graph.predecessors(node)))
        if docs_with_word > 0:
            idf[node] = math.log(n / docs_with_word)
        else:
            idf[node] = 0
    
    # 计算TF-IDF值
    tfidf = {}
    for node in nodes:
        tfidf[node] = tf[node] * idf[node]
    
    # 归一化TF-IDF值
    total_tfidf = sum(tfidf.values())
    if total_tfidf > 0:
        for node in nodes:
            tfidf[node] = tfidf[node] / total_tfidf
    
    return tfidf

# 计算PageRank值
def calc_page_rank(graph, word, max_iter=100, damping=0.85, tol=1.0e-6):
    # 检查单词是否在图中
    if word not in graph:
        return f"单词 '{word}' 不在图中"
    
    # 获取所有节点
    nodes = list(graph.nodes())
    n = len(nodes)
    
    # 计算TF-IDF值作为初始PageRank值
    tfidf = calculate_tfidf(graph)
    
    # 初始化PageRank值
    pagerank = {node: tfidf[node] for node in nodes}
    
    # 计算每个节点的出度
    out_degrees = {node: graph.out_degree(node) for node in nodes}
    
    # 找出所有出度为0的节点
    dangling_nodes = [node for node, degree in out_degrees.items() if degree == 0]
    
    # 迭代计算PageRank
    for _ in range(max_iter):
        new_pagerank = {}
        
        # 计算来自其他节点的贡献
        for node in nodes:
            incoming_contribution = 0.0
            for neighbor in graph.predecessors(node):
                if out_degrees[neighbor] > 0:
                    incoming_contribution += pagerank[neighbor] / out_degrees[neighbor]
            
            # 计算来自出度为0节点的贡献
            if dangling_nodes:
                dangling_contribution = sum(pagerank[node] for node in dangling_nodes) / (n - len(dangling_nodes))
                incoming_contribution += dangling_contribution
            
            # 应用PageRank公式，加入TF-IDF权重
            new_pagerank[node] = (1 - damping) * tfidf[node] + damping * incoming_contribution
        
        # 检查收敛性
        diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in nodes)
        if diff < tol:
            break
            
        pagerank = new_pagerank
    
    # 获取前5个最高PageRank值的节点
    top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n图中PageRank值最高的5个节点:")
    for node, rank in top_nodes:
        print(f"- {node}: {rank:.6f}")
    
    return pagerank[word]

# 随机游走
def random_walk(graph):
    # 随机选择一个起点
    start_node = random.choice(list(graph.nodes()))
    current_node = start_node
    visited_nodes = [current_node]
    visited_edges = set()
    
    while True:
        # 获取当前节点的所有出边
        neighbors = list(graph.neighbors(current_node))
        
        # 如果没有出边，停止游走
        if not neighbors:
            break
        
        # 随机选择一个邻居
        next_node = random.choice(neighbors)
        edge = (current_node, next_node)
        
        # 如果遇到重复边，停止游走
        if edge in visited_edges:
            break
        
        # 记录访问的边和节点
        visited_edges.add(edge)
        visited_nodes.append(next_node)
        current_node = next_node
    
    # 将游走结果写入文件
    output_file = "random_walk_result.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(' '.join(visited_nodes))
    
    return visited_nodes

# 选择文件路径函数（使用文件对话框）
def choose_file():
    root = Tk()
    root.withdraw()  # 不显示根窗口
    file_path = filedialog.askopenfilename(title="Select a Text File", filetypes=[("Text Files", "*.txt")])
    return file_path

# 主程序入口
def main():
    # 让用户选择文件
    filename = choose_file()
    if filename:
        print(f"Selected file: {filename}")
        # 读取文件并生成图
        graph = read_file_and_build_graph(filename)
        
        # 展示生成的有向图并保存为图像
        show_directed_graph(graph, None, "generated_graph.png")

        while True:
            print("\nChoose an option:")
            print("1. Query bridge words")
            print("2. Generate new text with bridge words")
            print("3. Calculate shortest path between two words")
            print("4. Calculate PageRank for a word")
            print("5. Perform random walk")
            print("6. Exit")
            
            choice = input("Enter your choice (1-6): ")
            
            if choice == "1":
                word1 = input("Enter the first word: ")
                word2 = input("Enter the second word: ")
                result = query_bridge_words(graph, word1, word2)
                if isinstance(result, list):
                    print(f"The bridge words from {word1} to {word2} are: {', '.join(result)}")
                else:
                    print(result)
            
            elif choice == "2":
                input_text = input("Enter the text: ")
                # 将输入文本转换为小写并分割成单词
                words = input_text.lower().split()
                new_text = []
                
                for i in range(len(words) - 1):
                    word1 = words[i]
                    word2 = words[i + 1]
                    
                    # 查询桥接词
                    bridge_words = query_bridge_words(graph, word1, word2)
                    
                    # 添加第一个单词
                    new_text.append(word1)
                    
                    # 如果有桥接词，随机选择一个插入
                    if isinstance(bridge_words, list) and bridge_words:
                        bridge_word = random.choice(bridge_words)
                        new_text.append(bridge_word)
                
                # 添加最后一个单词
                new_text.append(words[-1])
                
                print("\n原始文本:", input_text)
                print("生成的新文本:", " ".join(new_text))
            
            elif choice == "3":
                word1 = input("Enter the first word: ")
                word2 = input("Enter the second word (press Enter to calculate paths to all words): ")
                
                if word2.strip() == "":
                    # 计算到所有其他节点的最短路径
                    all_paths = calc_shortest_path(graph, word1)
                    if isinstance(all_paths, dict):
                        print(f"\n从 '{word1}' 到其他单词的最短路径:")
                        for target, paths in all_paths.items():
                            print(f"\n到 '{target}' 的最短路径:")
                            for path in paths:
                                print(f"路径: {' -> '.join(path)}")
                                # 为每条路径生成一个图形
                                show_directed_graph(graph, [path], f"shortest_path_{word1}_to_{target}.png")
                    else:
                        print(all_paths)
                else:
                    # 计算两个单词之间的所有最短路径
                    paths = calc_shortest_path(graph, word1, word2)
                    if isinstance(paths, str):
                        print(paths)
                    else:
                        print(f"从 {word1} 到 {word2} 的所有最短路径:")
                        for path in paths:
                            print(f"路径: {' -> '.join(path)}")
                        # 展示图形并高亮显示最短路径
                        show_directed_graph(graph, paths, "highlighted_graph.png")
            
            elif choice == "4":
                word = input("Enter the word: ")
                print(f"PageRank for {word}: {calc_page_rank(graph, word)}")
            
            elif choice == "5":
                print("Random walk:", random_walk(graph))
            
            elif choice == "6":
                print("Exiting program.")
                break
            
            else:
                print("Invalid choice. Please try again.")
    else:
        print("No file selected. Exiting program.")

if __name__ == "__main__":
    main()
