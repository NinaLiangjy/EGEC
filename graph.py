# coding=utf-8
from __future__ import unicode_literals
import re
from pyecharts.render import make_snapshot
import networkx as nx
import numpy as np
import pandas as pd
import argparse
import logging
import os
from os.path import join
import collections
from matplotlib import pyplot as plt
import json
import time
from pyecharts.options.series_options import LabelOpts
from tqdm import tqdm
import copy
from pyecharts import options as opts
from pyecharts.charts import Graph, Tab
from pyecharts.commons.utils import JsCode
import math
import random
import addressparser as addr
import spacy
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score,classification_report
# from snapshot_pyppeteer import snapshot


def color_mapping(value, value_range=(0, 1)):
    # color_range = ["#50a3ba", "#eac763", "#d94e5d"]
    # if color_range is None:
    color_range = ((80, 163, 186), (234, 199, 99), (217, 78, 93))
    (low, up) = value_range
    value = min(value, up)
    value = max(value, low)

    if value > low + 0.5 * (up - low):
        value -= 0.5 * (up - low)
        c_low, c_up = color_range[1], color_range[2]
    else:
        c_low, c_up = color_range[0], color_range[1]

    
    offset = (value - low) / (up - low)
    color = []
    for i in range(3):
        color.append(int((offset * (c_up[i] - c_low[i]) + c_low[i])))
    return 'rgb' + tuple(color).__str__()


def neg_log_f(x):
    beta = 1e-8
    base = 1.1
    return -math.log2(1 + beta - x)


def scale_norm(G: nx.DiGraph, sta_node, hop=3):  # 将节点出边权重归一化到和为1并增大方差
    if sta_node not in G.nodes:
        return
    neighbors = G[sta_node]
    log_w_sum = 0

    for end_node in neighbors.keys():
        weight = neighbors[end_node]['weight']
        neg_log_w = neg_log_f(weight)
        log_w_sum += neg_log_w
        G.edges[sta_node, end_node]['weight'] = neg_log_w

    for end_node in neighbors.keys():
        neg_log_w = neighbors[end_node]['weight']
        norm_w = neg_log_w / log_w_sum
        G.edges[sta_node, end_node]['weight'] = norm_w

    if hop > 1:
        scale_norm(G, sta_node, hop - 1)


def process_time(time_str):
    date, time = time_str.split(' ')
    y, m, d = map(int, date.split('-'))
    h, minus, s = map(int, time.split(':'))

    if 7 <= h <= 18:
        daytime = '白天'
    else:
        daytime = '晚上'

    if m <= 3:
        season = '春季'
    elif m <= 6:
        season = '夏季'
    elif m <= 9:
        season = '秋季'
    else:
        season = '冬季'
    return season, daytime


json_path = r'./res_merge/meta_data.json'


def clear_json():
    if os.path.exists(json_path):
        os.remove(json_path)


def read_from_json(key):
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        json_dict = json.load(f)
        try:
            return json_dict[key]
        except KeyError:
            return None


def write_to_json(key, value):
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            json_dict = json.load(f)
    else:
        json_dict = {}

    json_dict[key] = value
    json_str = json.dumps(json_dict, ensure_ascii=False, indent=4)
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json_str)


class Reasoning:
    def __init__(self, args):
        self.args = args
        self.wdir = args.wdir
        os.makedirs(self.wdir, exist_ok=True)
        self.logger = logging.getLogger('Reasoning')
        self.logger.addHandler(logging.FileHandler(join(self.wdir, 'out.txt'), mode='a'))
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)
        self.logger.info(time.ctime())
        self.dataset_name = args.dataset.split('/')[-1].replace('.json', '')
        self.data = json.load(open(self.args.dataset, "r", encoding="utf-8"))
        self.data = sorted(self.data, key=lambda x: x['受理时间'], reverse=False)
        # random.shuffle(self.data)
        # json.dump(self.data, open(join(self.wdir, 'data.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        self.n_data = len(self.data)
        self.train, self.test = self.data[:int(self.n_data * 0.9)], self.data[int(self.n_data * 0.9):]
        # self.test = self.train

        self._make_graphs()

        self.nodes_set = {}     # 方便按某种类型遍历顶点
        for node in self.graph.nodes:
            node_type = self.graph.nodes[node]['type']
            if node_type not in self.nodes_set:
                self.nodes_set[node_type] = set()
            self.nodes_set[node_type].add(node)

        print(f'n_nodes: {len(self.graph.nodes)}, n_edges: {len(self.graph.edges)}')    

        for type in self.nodes_set:
            print(f'{type}: {len(self.nodes_set[type])}')

    def _make_graphs(self):
        graph_path = join(self.wdir, 'graph.gexf')
        undirect_graph_path = join(self.wdir, "undirect_graph.gexf")
        if not self.args.remake and os.path.exists(graph_path):
            self.logger.info('从已有数据读取')
            self.graph = nx.read_gexf(graph_path)
            self.undirect_graph = nx.read_gexf(undirect_graph_path)

        else:
            self.logger.info('从原始数据构造')
            # data = json.load(open(self.args.dataset, 'r', encoding='utf-8'))
            data = self.train
            print(f'构图样本数: {len(data)}')

            time1 = time.time()
            n_items = len(data)
            node_2_items = {}
            node_weight = {}
            graph = nx.Graph()
            combined_nodes = {}
            for idx in tqdm(range(n_items), desc='预处理: '):
                row = data[idx]
                for key in row.keys():
                    if row[key] is None:
                        row[key] = f'未标定_{key}'
                label_columns = ['业务类型', '案件大类', '案件小类',]
                labels = [row[col] for col in label_columns if row[col] is not None]

                # department = row["merge_label" if self.args.merge_label else "三级主责"]
                department = row["merge_label"]
                item_id = idx
                time_str = row['受理时间']
                street = row["街镇"]
                NC = row['居委']
                informations = row['抽取标签']

                for label in labels:
                    if label not in graph.nodes:
                        graph.add_node(label, type='label')
                        node_2_items[label] = []
                        node_weight[label] = 0

                    node_2_items[label].append(item_id)
                    node_weight[label] += 1

                if department not in graph.nodes:
                    graph.add_node(department, type='department')
                    node_2_items[department] = []
                    node_weight[department] = 0

                node_2_items[department].append(item_id)
                node_weight[department] += 1

                if street not in graph.nodes:
                    graph.add_node(street, type='street')
                    node_2_items[street] = []
                    node_weight[street] = 0
                node_2_items[street].append(item_id)
                node_weight[street] += 1

                if NC not in graph.nodes:
                    graph.add_node(NC, type="NC")
                    node_2_items[NC] = []
                    node_weight[NC] = 0
                node_2_items[NC].append(item_id)
                node_weight[NC] += 1

                for information in informations:
                    tag = information.replace("\"","").strip() # strip 包含了去空格

                    # 规则 1 合并 ABCD == CDAB 类型
                    if len(tag) == 4 and tag[2:] + tag[:2] in graph.nodes:
                        tag = tag[2:] + tag[:2]
                    # 规则 2 合并
                    if re.match('疫情.*控', tag):
                        tag = '疫情管控'
                    if '消防通道' in tag:
                        tag = '消防通道'

                    # 规则 3 同义词库

                    similar_tags = [
                            ['疫情管控', '疫情防控', '疫情防治', '疫情防控', '疫情防疫'],
                            ['核酸检测', '核酸采样', '新冠病毒检测', ],
                            ['新冠疫情', '新冠病毒'],
                            ['核酸报告', '核酸要求', '核酸证明'],
                            ['动迁补偿','动迁款发放'],
                            ['密接隔离','次密接隔离']

                        ]
                    for similar_tag in similar_tags:
                        if tag in similar_tag:
                            tag = similar_tag[0]
                            break

                    if tag not in graph.nodes:
                        # 无向图
                        graph.add_node(tag, type='information')
                        node_2_items[tag] = []
                        node_weight[tag] = 0
                    node_2_items[tag].append(item_id)
                    node_weight[tag] += 1

            node_2_items = {k: set(v) for k, v in node_2_items.items()}

            time2 = time.time()
            # 无向图构建
            nodes_list = list(graph.nodes)
            for i, node_i in tqdm(enumerate(nodes_list), desc='无向图构建', total=len(nodes_list)):
                graph.nodes[node_i]['weight'] = node_weight[node_i]
                for node_j in nodes_list[i + 1:]:
                    items_i = node_2_items[node_i]
                    items_j = node_2_items[node_j]
                    co_occurance = len(items_i & items_j)
                    if co_occurance != 0:
                        graph.add_weighted_edges_from([(node_i, node_j, co_occurance)])
                        graph.add_weighted_edges_from([(node_j, node_i, co_occurance)])

            self.undirect_graph = copy.deepcopy(graph)      # 无向图
            nx.write_gexf(self.undirect_graph, undirect_graph_path)

            time3 = time.time()
            # 有向图构建
            digraph = nx.DiGraph()
            digraph.add_nodes_from(graph.nodes(data=True))
            for start_node in tqdm(digraph.nodes, total=len(digraph.nodes), desc='有向图构建'):
                all_weight_sums = 0
                weight_sums = {}
                for end_node in graph.neighbors(start_node):
                    node_type = graph.nodes[end_node]['type']
                    weight = graph.edges[start_node, end_node]['weight']
                    if node_type not in weight_sums:
                        weight_sums[node_type] = 0
                    weight_sums[node_type] += weight
                    all_weight_sums += weight

                for end_node in graph.neighbors(start_node):
                    node_type = graph.nodes[end_node]['type']
                    weight = graph.edges[start_node, end_node]['weight']
                    co_prob = weight / weight_sums[node_type]
                    # co_prob = weight / all_weight_sums
                    digraph.add_weighted_edges_from([(start_node, end_node, co_prob)])

            # self.digraph = digraph

            self.graph = digraph

            time4 = time.time()
            nx.write_gexf(self.graph, graph_path)

            print(time1, time2, time3, time4)
            print(time4-time1)


    def _make_hierarchical(self):

        hier_struct = read_from_json('hier')
        if not self.args.remake and hier_struct is not None:
            return hier_struct

        hier_struct = {}
        for root_node in tqdm(self.graph.nodes, desc='构造分层结构'):
            node_type = self.graph.nodes[root_node]['type']
            if node_type == 'label':
                continue
            if node_type not in hier_struct:
                hier_struct[node_type] = {}

            res = {'2': [], '3': []}
            neighbors_list = [(node, self.graph[root_node][node]['weight'])
                              for node in nx.neighbors(self.graph, root_node) if
                              self.graph.nodes[node]['type'] == 'label']

            neighbors_list = sorted(neighbors_list, key=lambda node: node[1], reverse=True)

            # 选第二层节点
            n_layer2 = int(len(neighbors_list) * self.args.thresh)
            for idx, (node, weight) in enumerate(neighbors_list):
                if idx < n_layer2:
                    res['2'].append(node)

            # 选择第三层节点
            for layer3_node in self.graph.nodes:
                if self.graph.nodes[layer3_node]['type'] != 'label' or layer3_node in res['2']:
                    continue
                for layer2_node in res['2']:
                    if self.graph.has_edge(layer2_node, layer3_node):  # 只选择与第二层有边的点
                        res['3'].append(layer3_node)
                        break

            hier_struct[node_type][root_node] = res

        return hier_struct

    def draw_hierarchical(self, root_node):
        width, height = 1200, 520
        n_layer2, n_layer3 = 10, 40  # 第二第三层最多显示的节点数
        node_show_size = [35, 20, 10]  # 各层节点大小比例
        show_node_label = [True, True, True]  # 各层节点标签是否显示
        link_width = 2  # 线条宽度
        link_color_range = ((240, 240, 240), (0, 0, 0))
        base_alpha = 0.2  # 0 - 1间，防止线条完全透明 线条绘制的不透明度为 min(1, base_alpha + weight)

        root_type = self.graph.nodes[root_node]['type']
        layer2_nodes = self.hier[root_type][root_node]['2'][:n_layer2]
        layer3_nodes = self.hier[root_type][root_node]['3'][:n_layer3]
        n_layer2, n_layer3 = len(layer2_nodes), len(layer3_nodes)

        if root_type == 'department':
            root_type = '部门'
        elif root_type == 'season':
            root_type = '季节'
        else:
            root_type = '位置'
        categories = [
            opts.GraphCategory(root_type, symbol_size=node_show_size[0],
                               label_opts=opts.LabelOpts(show_node_label[0])),
            opts.GraphCategory('结果', symbol_size=node_show_size[1], label_opts=opts.LabelOpts(show_node_label[1])),
            opts.GraphCategory('原因', symbol_size=node_show_size[2],
                               label_opts=opts.LabelOpts(show_node_label[2], position='bottom'))
        ]

        nodes, links = [], []
        G = nx.DiGraph()
        G.add_node(root_node, x=width / 2, y=(height // 3) * 0.5, category=0)

        if n_layer2 > 0:
            curr_x = (width / n_layer2) / 2
            curr_y = (height // 3) * 1.5
            for layer2_node in layer2_nodes:
                G.add_node(layer2_node, x=curr_x, y=curr_y, category=1)
                G.add_edge(root_node, layer2_node, weight=self.graph.edges[root_node, layer2_node]['weight'])
                curr_x += width / n_layer2

        if n_layer3 > 0:
            tmp = {}
            for layer3_node in layer3_nodes:
                layer3_neighbors = self.graph.neighbors(layer3_node)
                for layer2_node in layer3_neighbors:
                    if layer2_node not in layer2_nodes:
                        continue
                    weight = self.graph.edges[layer2_node, layer3_node]['weight']
                    if layer3_node not in tmp:
                        tmp[layer3_node] = []
                    tmp[layer3_node].append((layer2_node, weight))

            if len(tmp) > 0:
                n_layer3 = len(tmp)  # 去除掉第三层中与第二层无直连边的节点，真正可视化的第三层节点数
                curr_x = (width / n_layer3) / 2
                curr_y = (height // 3) * 2.5
                for layer3_node in tmp:
                    G.add_node(layer3_node, x=curr_x, y=curr_y, category=2)
                    # 从第三层取测试样本
                    curr_x += width / n_layer3
                    for (layer2_node, weight) in tmp[layer3_node]:
                        G.add_edge(layer2_node, layer3_node, weight=weight)

        for node in G:
            scale_norm(G, node)

        add_edges, remove_edges = [], []
        add_nodes, remove_nodes = [], []

        for node in G.nodes:
            x, y, cate = G.nodes[node]['x'], G.nodes[node]['y'], G.nodes[node]['category']
            if cate == 2:
                node_line = ''.join([c + '\n' for c in node])
                add_nodes.append((node_line, x, y, cate))
                for u, v in G.edges:
                    weight = G.edges[u, v]['weight']
                    if u == node:
                        add_edges.append((node_line, v, weight))
                        remove_edges.append((u, v))
                    elif v == node:
                        add_edges.append((u, node_line, weight))
                        remove_edges.append((u, v))
                remove_nodes.append(node)

        for (node, x, y, cate) in add_nodes:
            G.add_node(node, x=x, y=y, category=cate)
        for (u, v, w) in add_edges:
            G.add_edge(u, v, weight=w)
        for (u, v) in remove_edges:
            G.remove_edge(u, v)
        for node in remove_nodes:
            G.remove_node(node)

        nodes, links = [], []
        for node in G.nodes:
            x, y, cate = G.nodes[node]['x'], G.nodes[node]['y'], G.nodes[node]['category']
            nodes.append(
                opts.GraphNode(
                    node, x=x, y=y, category=cate
                )
            )

        for u, v in G.edges:
            weight = G.edges[u, v]['weight']
            links.append(
                opts.GraphLink(
                    u, v, value=weight,
                    label_opts=opts.LabelOpts(False),
                    linestyle_opts=opts.LineStyleOpts(
                        # color=color_mapping(weight, color_range=link_color_range),
                        width=link_width, curve=0.0, opacity=min(1, weight + base_alpha)
                    )
                )
            )

        color_bar = opts.VisualMapOpts(is_show=True, max_=1, min_=0,
                                       series_index=links,
                                       range_color=(color_mapping(0), color_mapping(1)))
        # w, h = '100%', 'calc(100vh)'
        w, h = '{}px'.format(width), '{}px'.format(height)
        c = (
            Graph(init_opts=opts.InitOpts(
                width=w, height=h, page_title='', chart_id=''.format(root_node),
                animation_opts=opts.AnimationOpts(animation=False)))
                .add("", nodes, links, categories, repulsion=500,
                     edge_label=opts.LabelOpts(is_show=True, position='middle', formatter="{c}"),
                     edge_symbol=['', ''],  
                    #  is_selected=True,
                     is_focusnode=True, is_roam=True,
                     layout='none')
                .set_global_opts(title_opts=opts.TitleOpts(title=""),
                                 legend_opts=opts.LegendOpts(is_show=True),
                                 visualmap_opts=color_bar)
                .add_js_funcs(
                '''

                '''
            )
        )

        return c

    def find_hotspot(self):

        nodes_graph = dict()
        for i in tqdm(self.nodes_set['information'], total=len(self.nodes_set['information']), desc='计算标签节点度数'):
            nodes_graph[i] = self.graph.nodes[i]['weight']

        nodes_reverse = sorted(nodes_graph.items(), key = lambda x:x[1], reverse = True)
        n = 15 # 字典中前 n 个度最大的nodes
        nodes_max = nodes_reverse[:n]
        print("度最大的节点:", nodes_max)
        # 存入excel文件
        col_name = []
        col_freq = []
        for hotspot in nodes_max:
            col_name.append(hotspot[0])
            col_freq.append(int(hotspot[1]))
        df = pd.DataFrame({'热点':col_name,'度数':col_freq})
        df.to_excel(f"./hotspot/{self.dataset_name}.xlsx",index=False)

        nodes_desc = f''.join([f'({node}: {degree})\n' for node, degree in nodes_max])
        print_list = ""
        for nodes in nodes_max:
            print_list = print_list + str(nodes) + '\n'

        hotspots = []
        for i in nodes_max:
            hotspots.append({'node': i[0], 'degreed': i[1], 'neighbors': []})
            edges_list = [] # 取权重最大的k条边
            k = 5
            for j in self.nodes_set["information"]:
                if self.undirect_graph.has_edge(i[0], j):
                    tuple_edges = (self.undirect_graph.edges[i[0],j]['weight'], i[0], j)
                    edges_list.append(tuple_edges)
            edges_list.sort(key=lambda x: x[0],reverse=True)
            for idx, m in enumerate(edges_list):
                if idx >= k:
                    break
                if m:
                    hotspots[-1]["neighbors"].append(
                        (m[0], m[2])
                    )
                    print_list = print_list + str(m) + '\n'
        with open(f'hotspot/json/hotspot{self.dataset_name}.json', 'w', encoding='utf-8') as f:
            json.dump(hotspots, f, ensure_ascii=False, indent=4)
        with open(f"hotspot/txt/hotspot{self.dataset_name}.txt", "w", encoding="utf-8") as f:
            f.write(print_list)    

        ### 可视化
        line_width = 2
        eps = {
            "起始点": 1500,
            "一跳结果": 60,
            "多跳结果": 10,
            "其它": 1,
        }  # 斥力因子，越大节点越远离
        gravity = 0.8  # 引力因子，越大节点越靠近
        edge_scale = 2

        categories = [
            opts.GraphCategory("热点事件", symbol_size=15 * 2, label_opts=opts.LabelOpts(True)),
            opts.GraphCategory("关联事件", symbol_size=7 * 2, label_opts=opts.LabelOpts(True)),
        ]
        links, nodes = [], []
        appended_node = []
        report_desc_input = ''
        for hotpot in hotspots:
            master_node = hotpot["node"]
            master_degreed = hotpot["degreed"]

            nodes.append(opts.GraphNode(master_node, category=0, value=master_degreed))
            appended_node.append(master_node)

        for hotpot in hotspots:
            master_node = hotpot["node"]
            master_degreed = hotpot["degreed"]
            report_desc_input += f"- {master_node} ({master_degreed})\n"
            for neighbor in hotpot["neighbors"]:
                neighbor_weight, neighbor_node = neighbor
                report_desc_input += f"  - {neighbor_node} ({neighbor_weight})\n"
                if neighbor_node not in appended_node :
                    nodes.append(opts.GraphNode(neighbor_node, category=1))
                    appended_node.append(neighbor_node)
                assert master_node in appended_node
                links.append(
                    opts.GraphLink(
                        master_node,
                        neighbor_node,
                        neighbor_weight,
                        label_opts=opts.LabelOpts(False),
                        linestyle_opts=opts.LineStyleOpts(
                            curve=0,
                            width=line_width,
                        ),
                    )
                )
        with open(f"hotspot/markdown/{self.dataset_name}.md", "w", encoding="utf-8") as f:
            f.write(report_desc_input)
        w, h = 2180, 1000
        ratio = 0.85
        w, h = int(w * ratio), int(1080 * ratio)

        c = (
            Graph(
                init_opts=opts.InitOpts(
                    width="{}px".format(w),
                    height="{}px".format(h),
                    page_title="{}".format("global graph"),
                    animation_opts=opts.AnimationOpts(animation=False),
                )
            )
            .add(
                "",
                nodes,
                links,
                categories,
                # edge_symbol=['', 'arrow'],
                edge_symbol_size=6,
                repulsion=8000,
                gravity=gravity,
                symbol_size = 20,
                edge_length=edge_scale,
                is_draggable=False,
                is_focusnode=True,
            )
            .set_global_opts(
                legend_opts=opts.LegendOpts(is_show=True),
                # visualmap_opts=color_bar
                title_opts=opts.TitleOpts(
                    title="",
                    subtitle=nodes_desc,
                    pos_left="left",
                    pos_top="top",
                    # pos_bottom='10'
                ),
            )
        )
        c.render(f"hotspot/html/{self.dataset_name}.html")
        # make_snapshot(snapshot, c.render(), f'hotspot/png/{self.dataset_name}.png')

    def draw_global_graph(self):

        alpha_base = 0.4
        line_width = 2
        eps = {'起始点': 1500, '一跳结果': 60, '多跳结果': 10, '其它': 1}  # 斥力因子，越大节点越远离
        gravity = 0.8  # 引力因子，越大节点越靠近
        edge_scale = 2

        categories = [
            opts.GraphCategory('标签', symbol_size=7, label_opts=opts.LabelOpts(True)),
            opts.GraphCategory('部门', symbol_size=15, label_opts=opts.LabelOpts(True)),
            opts.GraphCategory('街镇', symbol_size=15, label_opts=opts.LabelOpts(True)),
            opts.GraphCategory('居委', symbol_size=15, label_opts=opts.LabelOpts(True)),
        ]

        links, nodes = [], []
        for node in self.graph.nodes:
            node_type = self.graph.nodes[node]['type']
            cate = {'label': 0, 'department': 1, 'street': 2, 'NC': 3,'information':4}[node_type]
            nodes.append(opts.GraphNode(node, category=cate))

        # max_list=[]
        for i, j in self.graph.edges:
            if i == "投诉类" or i == "求助类" or j == "求助类" or j == "投诉类":
                continue
            weight = self.graph.edges[i, j]['weight']
            if weight > 10:
                link_color = color_mapping(weight)
                links.append(opts.GraphLink(
                    i, j, weight, label_opts=opts.LabelOpts(False),
                    linestyle_opts=opts.LineStyleOpts(
                        curve=0,
                        color=link_color,
                        width=line_width,

                    )
                ))
        #     tpu=(self.graph.edges[i, j]['weight'],i,j)
        #     max_list.append(tpu)
        # max_list.sort(key=lambda x:x[0],reverse=True)
        # for i in range(20):
        #     print(max_list[i])

        color_bar = opts.VisualMapOpts(
            # is_show=True, max_=1, min_=0,
            is_show=True, max_=500, min_=0,
            series_index=links,
            # range_color=(color_mapping(0), color_mapping(1))
        )

        w, h = 2180, 1000
        ratio = 0.85
        w, h = int(w * ratio), int(1080 * ratio)

        c = (
            Graph(init_opts=opts.InitOpts(
                width='{}px'.format(w), height='{}px'.format(h), page_title='{}'.format('global graph'),
                animation_opts=opts.AnimationOpts(animation=False)))
                .add(
                "",
                nodes,
                links,
                categories,
                edge_symbol=['', 'arrow'],
                edge_symbol_size=6,
                repulsion=8000,
                gravity=gravity,
                edge_length=edge_scale,
                is_draggable=False,
                # is_selected=True,
                is_focusnode=True,

            )
                .set_global_opts(
                legend_opts=opts.LegendOpts(is_show=True),
                visualmap_opts=color_bar
            )
        )

        c.render('./res_case/global_graph.html')

    def _res_to_str(self, infer_res):
        infer_str = ''
        if infer_res is not None:
            for cnt, label in enumerate(infer_res):
                weight = round(infer_res[label]['weight'], 4)
                department = infer_res[label]['department'][0] if 'department' in infer_res[label] else ''
                department_prob = round(infer_res[label]['department'][1], 4) if 'department' in infer_res[label] else ''
                season = infer_res[label]['season'][0] if 'season' in infer_res[label] else ''
                season_prob = round(infer_res[label]['season'][1], 4) if 'season' in infer_res[label] else ''
                geo = infer_res[label]['geo'][0] if 'geo' in infer_res[label] else ''
                geo_prob = round(infer_res[label]['geo'][1], 4) if 'geo' in infer_res[label] else ''
                infer_str += '{} {}  {} {}   {} {}   {} {}\n'.format(
                    label, weight, department, department_prob,
                    season, season_prob, geo, geo_prob, 4
                )
                if cnt >= self.args.k:
                    break
            return infer_str
        return "None"

    def classfication(self):

        test_data = self.test
        print(f"测试集大小: {len(test_data)}")
        true_count = 0
        os.system("rm /your_path/infer_graph_*.html")

        label_dic = {
            "64facbd4b760829705f68103": 0,
            "65f7d926b760829705f752ee": 1,
            "区红十字会": 2,
            "区总工会": 3,
            "区医保": 4,
            "区民宗办": 5,
            "区金融办": 6,
            "区信访办": 7,
            "区民防办": 8,
            "区科协": 9,
            "黄浦烟草二公司": 10,
            "豫园集团": 13,
            "四大集团": 15,
            "后勤集团": 16,
            "淮海集团": 18,
            "外滩集团": 19,
            "新世界集团": 21,
            "豫园商城": 23,
            "区商务委": 24,
            "区卫生健康委": 25,
            "区发改委": 26,
            "区政法委": 27,
            "区建设管理委": 28,
            "区科委": 29,
            "区国资委": 30,
            "区党史研究室": 31,
            "区退役军人局": 32,
            "区公安分局": 33,
            "区市场局": 34,
            "区生态环境局": 35,
            "区绿化市容局": 36,
            "区应急局": 37,
            "区民政局": 38,
            "区财政局": 39,
            "区档案局": 40,
            "区司法局": 41,
            "区城管执法局": 42,
            "区文化旅游局": 43,
            "区规划资源局": 44,
            "区人社局": 45,
            "区体育局": 46,
            "区教育局": 47,
            "区审计局": 48,
            "区统计局": 49,
            "区老干部局": 50,
            "区住房保障局": 51,
            "区行政服务中心": 52,
            "区人武": 53,
            "老凤祥": 54,
            "区侨联": 55,
            "区工商联": 56,
            "区妇联": 57,
            "区残联": 58,
            "街道":59,
            "消防救援支队": 69
        }
        y_pred = []#预测结果
        y_val = []#分类结果
        for idx in tqdm(range(len(test_data)), desc='测试集推理: '):
            query_nodes = []
            row = test_data[idx]

            # ground_true = row["merge_label" if self.args.merge_label else "三级主责"]
            ground_true = row["merge_label"]
            for col in ['业务类型', '案件大类', '案件小类',]:
                query_nodes.append(row[col])
                pass

            street = row["街镇"]
            NC = row['居委']
            query_nodes.append(street)
            query_nodes.append(NC)
            for tag in row['抽取标签']:
                tag = tag.replace('\"', '').strip()
                if len(tag) == 4 and tag[2:] + tag[:2] in self.graph.nodes:
                    tag = tag[2:] + tag[:2]
                query_nodes.append(tag)
            query_nodes = list(set(query_nodes))

            departs = {}

            for label in query_nodes:
                if label not in self.graph.nodes():
                    continue
                for depart in self.graph[label]:
                    if self.graph.nodes[depart]['type'] != 'department':
                        continue

                    weight = self.graph[label][depart]["weight"]
                    if depart not in departs:
                        departs[depart] = []

                    departs[depart].append((weight))

            infer_res = []

            for depart in departs:
                n_items = len(departs[depart])
                infer_res.append((depart, sum(departs[depart]) / n_items))
            infer_res = sorted(infer_res, key=lambda x: x[1], reverse=True)
            
            # #计算召回率
            # y_pred.append(int(label_dic[infer_res[0][0]]))
            # y_val.append(int(label_dic[ground_true]))



            # self.draw_infer_graph_hier(row, filename=f'infer_graph_{idx}', infer_res=infer_res)

            is_merge_street = True
            def is_correct(ground_true, infer_res):
                for i in range(4):
                    if infer_res[i][0] == ground_true:
                        return True
            try :
                if is_correct(ground_true, infer_res):
                    true_count += 1
            except Exception:
                continue

            if idx % 100 == 0:
                tqdm.write(f'{idx} / {len(test_data)} 准确率: {true_count / (idx + 1)}')

        print(f"总体准确率: {true_count / len(test_data)}")

        # # 计算宏平均召回率
        # macro_recall = recall_score(y_val, y_pred, average='macro')

        # # 计算微平均召回率
        # micro_recall = recall_score(y_val, y_pred, average='micro')

        # print(f"Macro-average Recall: {macro_recall}")
        # print(f"Micro-average Recall: {micro_recall}")

        # # 也可以通过classification_report获取更详细的分类指标报告，其中包含了各类别的召回率等信息
        # report = classification_report(y_val, y_pred)
        # print(report)

    def draw_infer_graph_hier(self, row, filename='infer_gragh', infer_res=[]):
        width, height = 1920, 1080
        width, height = int(width * 0.8), int(height * 0.6)
        topK = 10
        depart_nodes = [depart for depart, score in infer_res[:topK]]
        depart_scores = [score for depart, score in infer_res[:topK]]
        query_nodes = []
        for col in ['业务类型', '案件大类', '案件小类', '街镇', '居委']:
            query_nodes.append(row[col])

        informations = []
        for tag in row["抽取标签"]:
            tag = tag.replace('"', "").strip()
            if len(tag) == 4 and tag[2:] + tag[:2] in self.graph.nodes:
                tag = tag[2:] + tag[:2]
            query_nodes.append(tag)
            informations.append(tag)

        alpha_base = 0.4
        line_width = 2
        gravity = 0.8  # 引力因子，越大节点越靠近
        edge_scale = 2
        text_desc = ''
        for i, c in enumerate(row["问题描述"]):
            if i % 130 == 0:
                text_desc += '\n'
            text_desc += c
        text_desc += f'\n抽取标签:{str(informations)}'
        # text_desc += f'\n真实标签:{row["merge_label" if self.args.merge_label else "三级主责"]}'
        text_desc += f'\n真实标签:{row["merge_label"]}'
        categories = [
            opts.GraphCategory("案件分类", symbol_size=15, label_opts=opts.LabelOpts(True)),
            opts.GraphCategory("部门", symbol_size=15, label_opts=opts.LabelOpts(True)),
            opts.GraphCategory("街镇", symbol_size=15, label_opts=opts.LabelOpts(True)),
            opts.GraphCategory("居委", symbol_size=15, label_opts=opts.LabelOpts(True)),
            opts.GraphCategory("事件标签", symbol_size=15, label_opts=opts.LabelOpts(True)),
        ]
        cates = {'label': 0, 'department': 1, 'street': 2, 'NC': 3, 'information': 4}
        links, nodes = [], []
        upper_y, lower_y = height // 3 * 0.5, height // 3 * 2.3

        for i, qry_node in enumerate(query_nodes):
            x = int(width * (i + 1) / (len(query_nodes) + 1))
            if qry_node not in self.graph.nodes:
                if qry_node in informations:
                    node_type = 'information'
                else:
                    continue
            else:
                node_type = self.graph.nodes[qry_node]['type']
            nodes.append(opts.GraphNode(qry_node, x=x, y=upper_y, category=cates[node_type],
                                        label_opts=opts.LabelOpts(True, position='top')))

        for i, depart_node in enumerate(depart_nodes):
            x = int(width * (i + 1) / (len(depart_nodes) + 1))
            score = depart_scores[i]
            score = round(score, 4)
            nodes.append(opts.GraphNode(
                depart_node, x=x, y=lower_y, 
                category=cates['department'], 
                value=score,
                label_opts=opts.LabelOpts(
                    True, 
                    position='bottom',
                    formatter="{b}\n({c})",  # {b} 表示标签，{c} 表示 value
                    
                )                 
            ))

        for qry_node in query_nodes:

            edges = []
            weight_sum = 0

            def exp(x):
                return x
                # return 2.71828 ** x

            for depart_node in depart_nodes:
                if not (qry_node in self.graph.nodes and depart_node in self.graph.nodes):
                    continue
                if qry_node in self.graph and self.graph.has_edge(qry_node, depart_node):
                    weight = self.graph.edges[qry_node, depart_node]['weight']
                    edges.append((depart_node, weight))
                    weight_sum += exp(weight)

            for depart_node, weight in edges:
                # weight = exp(weight) / weight_sum

                links.append(opts.GraphLink(
                        qry_node, depart_node, weight, label_opts=opts.LabelOpts(False),
                        linestyle_opts=opts.LineStyleOpts(
                            curve=0,
                            color=color_mapping(weight),
                            width=line_width,
                        )
                    ))

        color_bar = opts.VisualMapOpts(
            is_show=True, max_=1, min_=0,
            series_index=links,
            range_color=["#50a3ba", "#eac763", "#d94e5d"],
            orient='vertical',
            pos_top='10%',
        )

        c = (
            Graph(
                init_opts=opts.InitOpts(
                    width="{}px".format(width),
                    height="{}px".format(height),
                    page_title="{}".format("infer graph"),
                    animation_opts=opts.AnimationOpts(animation=False),
                )
            )
            .add(
                "",
                nodes,
                links,
                categories,
                edge_symbol=["", "arrow"],
                edge_symbol_size=6,
                repulsion=500,
                gravity=gravity,
                edge_length=edge_scale,
                is_draggable=False,
                is_focusnode=True,
                is_roam=False,
                layout="none",
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="", 
                    subtitle=text_desc, 
                    pos_left='left', 
                    pos_top='bottom',
                    # pos_bottom='10'
                ),
                legend_opts=opts.LegendOpts(is_show=True),
                visualmap_opts=color_bar,
            )
        )

        c.render(f'your_path{filename}.html')


        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        default="your_file.json",
        help="数据集路径",
    )
    parser.add_argument('--wdir', default='your_path', help='中间结果保存目录')
    parser.add_argument('--remake', default=True, action='store_true', help='重新从数据集构图')
    parser.add_argument('--test_label', default=None, help='指定测试样例')
    parser.add_argument('--thresh', default=0.2, type=float, help='0 - 1,分层第二层占部门邻居总数的比例')
    parser.add_argument(
        "--merge_label", default=True, type=bool, help="是否合并部门标签"
    )
    parser.add_argument('--accn', default=2, type= int, help="topk准确率")
    args = parser.parse_args()

    if args.remake:
        if os.path.exists(args.wdir):
            import shutil
            shutil.rmtree(args.wdir)

    r = Reasoning(args)
    r.draw_global_graph()
    # r.find_hotspot()
    # root = "区建设管理委"
    # r.draw_hierarchical(root)
    r.classfication()
