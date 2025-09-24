import pandas as pd
import matplotlib.pyplot as plt
import re
from typing import List, Tuple, Dict
import networkx as nx
from math import sqrt
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, MultiPoint
from shapely.ops import nearest_points
from collections import OrderedDict
import traceback

# -------------------------- 1. 全局配置参数（移除评分相关参数） --------------------------
EXCEL_PATH = "cleaned_excels/2.xlsx"
UNIT_CONVERT = 1000  # 单位转换系数（如原始数据为毫米，转换为米）
FIG_SIZE = (15, 12)  # 图像尺寸
LINE_WIDTH = 1.2  # 线条宽度
POINT_SIZE = 1.5  # 点大小
PLANT_ALPHA = 0.6  # 植物透明度
PATH_COLOR = "#FF0000"  # 路径颜色（红色）
PATH_STYLE = "-"  # 路径样式（实线）
DEBUG_GRAPH = True  # 调试模式（显示障碍物区域）

# 入口和出口坐标（根据实际Excel数据范围调整）
START_POINT = (-11.5, 32.6)
END_POINT = (29.3, 65.8)

# 元素样式配置（景观元素的颜色与标签）
ELEMENT_STYLE = {
    "半开放建筑": ("#696969", "#D3D3D3", "半开放建筑"),
    "实体建筑": ("#000000", "#000000", "实体建筑"),
    "道路": ("#FFD700", "#FFFACD", "道路"),
    "假山": ("#DC143C", "#FFB6C1", "假山"),
    "水体": ("#1E90FF", "#87CEFA", "水体")
}

# 障碍物与景点定义（仅用于路径避障和路网优化）
ATTRACTION_ELEMENTS = ["假山", "水体", "半开放建筑", "实体建筑"]  # 需靠近的景观元素
HARD_OBSTACLE_ELEMENTS = ["实体建筑"]  # 不可穿越的硬障碍物
SOFT_OBSTACLE_ELEMENTS = ["假山", "水体"]  # 可靠近的软障碍物
BUFFER_DISTANCE = 0.2  # 障碍物缓冲距离（减小以增强路网连通性）
CONNECTION_THRESHOLD = 10.0  # 路网节点连接阈值（增大以减少孤立分量）


# -------------------------- 2. 辅助函数（移除评分相关函数） --------------------------
def flatten_geometries(geom):
    """将MultiPolygon拆分为单个Polygon列表，处理障碍物几何形状"""
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    elif isinstance(geom, Polygon):
        return [geom]
    return []


def read_excel_sheet(excel_path: str, sheet_name: str) -> pd.DataFrame:
    """读取Excel指定工作表，统一列名用于后续坐标提取"""
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl", header=0)
        if len(df.columns) >= 2:
            df.columns = ["segment_coords", "no_segment_coords"]  # 线段坐标列、散点坐标列
        else:
            df = df.iloc[:, 0].to_frame(name="segment_coords")
            df["no_segment_coords"] = pd.NA  # 无散点数据时填充空值
        return df
    except Exception as e:
        print(f"⚠️  读取【{sheet_name}】工作表失败: {e}")
        return pd.DataFrame()


def extract_segment_coords(coord_series: pd.Series) -> List[List[Tuple[float, float]]]:
    """从Excel列中提取线段坐标（格式：{0;序号}分隔线段，点为"编号{X,Y,Z}"）"""
    segments, current_segment = [], []
    # 正则表达式：匹配分段标记和点坐标
    pattern_segment = re.compile(r"\{0;(\d+)\}")
    pattern_point = re.compile(r"(\d+)\.?\s*\{(.*?)\s*,\s*(.*?)\s*,\s*(.*?)\}")

    for cell in coord_series.dropna():
        cell_str = str(cell).strip()
        # 遇到分段标记，保存当前线段并重置
        if pattern_segment.match(cell_str):
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            continue
        # 提取点坐标（取X、Y，忽略Z轴）
        point_match = pattern_point.match(cell_str)
        if point_match:
            try:
                x = float(point_match.group(2)) / UNIT_CONVERT
                y = float(point_match.group(3)) / UNIT_CONVERT
                current_segment.append((x, y))
            except (ValueError, IndexError) as e:
                print(f"⚠️  解析线段坐标失败（内容：{cell_str}）: {e}")
                continue
    # 保存最后一段未完成的线段
    if current_segment:
        segments.append(current_segment)
    return segments


def extract_no_segment_coords(coord_series: pd.Series) -> List[Tuple[float, float]]:
    """从Excel列中提取散点坐标（格式：{X,Y,Z}）"""
    points = []
    pattern = re.compile(r"\{(.*?)\s*,\s*(.*?)\s*,\s*(.*?)\}")
    for cell in coord_series.dropna():
        cell_str = str(cell).strip()
        match = pattern.search(cell_str)
        if match:
            try:
                x = float(match.group(1)) / UNIT_CONVERT
                y = float(match.group(2)) / UNIT_CONVERT
                points.append((x, y))
            except (ValueError, IndexError) as e:
                print(f"⚠️  解析散点坐标失败（内容：{cell_str}）: {e}")
                continue
    return points


def extract_water_coords(coord_series: pd.Series) -> List[List[Tuple[float, float]]]:
    """适配水体坐标格式（与普通线段格式一致，复用提取函数）"""
    return extract_segment_coords(coord_series)


def extract_plant_data(df: pd.DataFrame) -> List[Tuple[float, float, float]]:
    """提取植物数据（坐标X,Y + 半径R），用于绘制圆形植物图形"""
    plants = []
    # 正则表达式：匹配植物坐标和半径
    pattern_coord = re.compile(r"\{(.*?)\s*,\s*(.*?)\s*,\s*(.*?)\}")
    pattern_radius = re.compile(r"(\d+\.?\d*)")

    for _, row in df.dropna().iterrows():
        # 提取坐标（第一列）
        coord_str = str(row.iloc[0]).strip()
        coord_match = pattern_coord.search(coord_str)
        if not coord_match:
            print(f"⚠️  植物坐标格式错误（内容：{coord_str}），跳过")
            continue
        try:
            x = float(coord_match.group(1)) / UNIT_CONVERT
            y = float(coord_match.group(2)) / UNIT_CONVERT
        except (ValueError, IndexError) as e:
            print(f"⚠️  植物坐标转换失败（内容：{coord_str}）: {e}，跳过")
            continue

        # 提取半径（第二列）
        radius_str = str(row.iloc[1]).strip()
        radius_match = pattern_radius.search(radius_str)
        if not radius_match:
            print(f"⚠️  植物半径格式错误（内容：{radius_str}），使用默认值0.5米")
            radius = 0.5 / UNIT_CONVERT
        else:
            try:
                radius = float(radius_match.group(1)) / UNIT_CONVERT
            except ValueError as e:
                print(f"⚠️  植物半径转换失败（内容：{radius_str}）: {e}，使用默认值0.5米")
                radius = 0.5 / UNIT_CONVERT

        plants.append((x, y, radius))
    return plants


def weighted_edge_cost(u, v, data, G):
    """带景观偏好的边成本计算（靠近景点的边成本降低，引导路径经过景点）"""
    base_weight = data['weight']  # 基础成本：两点之间的直线距离
    # 获取节点关联的景观属性（无属性则视为普通道路点）
    key_attr = G.nodes[u].get('attr_name') or G.nodes[v].get('attr_name')

    # 不同景观的成本权重（优先选择靠近自然景点的路径）
    if key_attr:
        if "假山" in key_attr:
            return base_weight * 0.6  # 靠近假山，成本降低40%
        if "水体" in key_attr:
            return base_weight * 0.7  # 靠近水体，成本降低30%
        if "建筑" in key_attr:
            return base_weight * 0.9  # 靠近建筑，成本降低10%
    return base_weight  # 普通道路点，保持基础成本


def is_line_intersect_obstacles(point1: Tuple[float, float], point2: Tuple[float, float],
                                obstacles: List[Polygon]) -> bool:
    """检测线段是否与障碍物相交（处理MultiPolygon，确保避障有效性）"""
    if not obstacles:
        return False
    line = LineString([point1, point2])

    # 检查线段与所有障碍物（含拆分后的子多边形）的交集
    for obstacle in obstacles:
        for poly in flatten_geometries(obstacle):
            if line.intersects(poly):
                return True  # 相交=不可通行
    return False


# -------------------------- 3. 核心路径规划函数（无评分相关逻辑） --------------------------
def plan_path_covering_attractions_with_obstacles(
        road_segments: List[List[Tuple[float, float]]],
        attractions_data: Dict[str, List[List[Tuple[float, float]]]],
        obstacles_data: Dict[str, List[List[Tuple[float, float]]]],
        ax=None
) -> List[Tuple[float, float]]:
    """
    带避障和景观偏好的TSP路径规划
    输入：道路线段、景点数据、障碍物数据、绘图轴
    输出：规划后的路径点列表（空列表表示规划失败）
    """
    global START_POINT, END_POINT

    # 1. 校验输入：无道路数据则直接返回失败
    if not road_segments:
        print("❌ 无有效道路数据，无法规划路径")
        return []

    # 2. 生成障碍物几何形状（处理硬/软障碍物，支持MultiPolygon）
    ## 2.1 硬障碍物（实体建筑，不可穿越）
    hard_obstacles = []
    for elem in HARD_OBSTACLE_ELEMENTS:
        for seg in obstacles_data.get(elem, []):
            if len(seg) >= 3:  # 至少3个点构成闭合多边形
                try:
                    poly = Polygon(seg)
                    # 修复无效多边形（如自相交）
                    if not poly.is_valid:
                        fixed_poly = poly.buffer(0)
                        print(f"🔧 修复实体建筑多边形（原始点数量：{len(seg)}，修复后类型：{type(fixed_poly).__name__}）")
                        poly = fixed_poly
                    # 添加缓冲距离（避免路径紧贴障碍物）
                    buffered_poly = poly.buffer(BUFFER_DISTANCE)
                    hard_obstacles.extend(flatten_geometries(buffered_poly))
                except Exception as e:
                    print(f"⚠️  生成实体建筑障碍物失败: {e}")
    print(f"📊 有效硬障碍物数量（拆分后）：{len(hard_obstacles)}")

    ## 2.2 软障碍物（假山/水体，可靠近）
    soft_obstacles = []
    for elem in SOFT_OBSTACLE_ELEMENTS:
        for seg in obstacles_data.get(elem, []):
            if len(seg) >= 3:
                try:
                    poly = Polygon(seg)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    # 软障碍物缓冲距离减半（允许更靠近）
                    buffered_poly = poly.buffer(BUFFER_DISTANCE * 0.5)
                    soft_obstacles.extend(flatten_geometries(buffered_poly))
                except Exception as e:
                    print(f"⚠️  生成{elem}障碍物失败: {e}")
    print(f"📊 有效软障碍物数量（拆分后）：{len(soft_obstacles)}")

    # 3. 构建路网图（节点=道路点，边=可通行路段）
    G = nx.Graph()
    road_points_set = set()  # 存储所有道路节点（去重）

    ## 3.1 添加原始道路线段（过滤穿越硬障碍物的路段）
    for segment in road_segments:
        for i in range(len(segment) - 1):
            p1, p2 = segment[i], segment[i + 1]
            if not is_line_intersect_obstacles(p1, p2, hard_obstacles):
                # 计算路段距离（作为边权重）
                dist = sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                # 为节点添加默认属性（避免后续KeyError）
                if p1 not in G.nodes:
                    G.add_node(p1, attr_name="道路点")
                    road_points_set.add(p1)
                if p2 not in G.nodes:
                    G.add_node(p2, attr_name="道路点")
                    road_points_set.add(p2)
                # 添加可通行路段
                G.add_edge(p1, p2, weight=dist)
    # 校验路网有效性
    if not G.nodes:
        print("❌ 所有道路均穿越硬障碍物，无法构建路网")
        return []
    print(f"📊 路网基础状态：{len(G.nodes)}个节点，{len(G.edges)}条边")

    ## 3.2 增强路网连通性（减少孤立分量，优先确保入口连通）
    road_points_list = list(road_points_set)
    # 入口周边节点（20米范围内）优先连接
    entry_buffer = 20.0
    entry_area_nodes = [
        rp for rp in road_points_list
        if sqrt((rp[0] - START_POINT[0]) ** 2 + (rp[1] - START_POINT[1]) ** 2) <= entry_buffer
    ]
    other_nodes = [rp for rp in road_points_list if rp not in entry_area_nodes]

    # 连接入口区域节点（阈值×1.5，降低孤立概率）
    for i, p1 in enumerate(entry_area_nodes):
        for j, p2 in enumerate(entry_area_nodes[i + 1:], i + 1):
            if G.has_edge(p1, p2):
                continue  # 跳过已存在的边
            dist = sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            if dist <= CONNECTION_THRESHOLD * 1.5 and not is_line_intersect_obstacles(p1, p2, hard_obstacles):
                G.add_edge(p1, p2, weight=dist)

    # 连接其他区域节点（默认阈值）
    for i, p1 in enumerate(road_points_list):
        for j, p2 in enumerate(road_points_list[i + 1:], i + 1):
            if G.has_edge(p1, p2):
                continue
            dist = sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            if dist <= CONNECTION_THRESHOLD and not is_line_intersect_obstacles(p1, p2, hard_obstacles):
                G.add_edge(p1, p2, weight=dist)
    print(f"📊 路网增强后：{len(G.nodes)}个节点，{len(G.edges)}条边")
    road_points_multipoint = MultiPoint(road_points_list) if road_points_list else None

    # 4. 连接关键节点（入口、出口、景点锚点）
    key_points_to_visit = []  # 用于TSP规划的关键点列表

    def try_connect_point(point: Tuple[float, float], name: str, max_attempts: int = 80) -> bool:
        """尝试将关键点连接到路网，返回连接结果（True=成功）"""
        if not road_points_list:
            print(f"❌ 无法连接【{name}】：无路网节点可用")
            return False
        # 已在路网中则无需重复连接
        if point in G.nodes:
            print(f"✅ 【{name}】已在路网中")
            return True

        # 按距离排序道路节点，优先连接近处且不穿障碍物的节点
        sorted_road_points = sorted(
            road_points_list,
            key=lambda rp: sqrt((rp[0] - point[0]) ** 2 + (rp[1] - point[1]) ** 2)
        )

        # 优先尝试不穿硬障碍物的连接
        for road_pt in sorted_road_points[:max_attempts]:
            if not is_line_intersect_obstacles(point, road_pt, hard_obstacles):
                dist = sqrt((point[0] - road_pt[0]) ** 2 + (point[1] - road_pt[1]) ** 2)
                G.add_node(point, attr_name=name)
                G.add_edge(point, road_pt, weight=dist)
                road_points_set.add(point)
                print(f"✅ 【{name}】连接到路网（距离：{dist:.2f}m，不穿硬障碍物）")
                return True

        # 尝试更远节点（仍不穿硬障碍物）
        for road_pt in sorted_road_points[max_attempts:max_attempts + 50]:
            if not is_line_intersect_obstacles(point, road_pt, hard_obstacles):
                dist = sqrt((point[0] - road_pt[0]) ** 2 + (point[1] - road_pt[1]) ** 2)
                G.add_node(point, attr_name=name)
                G.add_edge(point, road_pt, weight=dist)
                road_points_set.add(point)
                print(f"✅ 【{name}】连接到远端路网（距离：{dist:.2f}m）")
                return True

        # 强制连接（即使靠近障碍物，避免关键点丢失）
        closest_road_pt = sorted_road_points[0]
        dist = sqrt((point[0] - closest_road_pt[0]) ** 2 + (point[1] - closest_road_pt[1]) ** 2)
        G.add_node(point, attr_name=name)
        G.add_edge(point, closest_road_pt, weight=dist)
        road_points_set.add(point)
        print(f"⚠️  【{name}】强制连接（可能靠近障碍物，距离：{dist:.2f}m）")
        return True

    ## 4.1 连接入口和出口（优先确保起点终点连通）
    if try_connect_point(START_POINT, "入口", max_attempts=100):
        key_points_to_visit.append(START_POINT)
    if try_connect_point(END_POINT, "出口", max_attempts=100):
        if END_POINT not in key_points_to_visit:
            key_points_to_visit.append(END_POINT)

    ## 4.2 连接景点（添加景点锚点到关键点列表）
    for attr_name, segments in attractions_data.items():
        all_attr_points = [p for seg in segments for p in seg]
        if not all_attr_points or len(all_attr_points) < 3:
            print(f"⚠️  景点【{attr_name}】有效点不足3个，跳过连接")
            continue
        try:
            # 构建景点多边形，找路网中最近的点作为连接锚点
            attr_poly = Polygon(all_attr_points)
            if not attr_poly.is_valid:
                attr_poly = attr_poly.buffer(0)
            if not road_points_multipoint:
                continue
            closest_road_pt, closest_attr_pt = nearest_points(road_points_multipoint, attr_poly)
            attr_anchor = (closest_attr_pt.x, closest_attr_pt.y)  # 景点锚点（靠近路网侧）
            # 连接景点锚点到路网
            if try_connect_point(attr_anchor, f"{attr_name}_锚点"):
                key_points_to_visit.append(attr_anchor)
        except Exception as e:
            print(f"⚠️  连接景点【{attr_name}】失败: {e}")

    ## 4.3 关键点去重与有效性校验
    key_points_to_visit = list(OrderedDict.fromkeys(key_points_to_visit))  # 去重
    key_point_labels = [G.nodes[p].get('attr_name', '未知点') for p in key_points_to_visit]
    print(f"📊 最终关键点列表（{len(key_points_to_visit)}个）：{key_point_labels}")
    if len(key_points_to_visit) < 2:
        print("❌ 关键点不足2个，无法执行TSP规划")
        return []

    # 5. 连通分量优化（选择包含最多关键点的分量，确保路径连通）
    if not nx.is_connected(G):
        print("🔍 路网不连通，分析连通分量...")
        connected_components = list(nx.connected_components(G))
        # 统计每个分量的关键点数量
        component_info = []
        for cc in connected_components:
            cc_kps = [p for p in key_points_to_visit if p in cc]
            component_info.append((cc, len(cc), len(cc_kps)))
            print(f"   分量{len(component_info) - 1}：{len(cc)}个节点，{len(cc_kps)}个关键点")

        # 选择关键点最多的分量（优先），无则选择节点最多的分量
        component_info.sort(key=lambda x: (-x[2], -x[1]))
        best_cc, best_cc_node_count, best_cc_kp_count = component_info[0]
        # 截取路网到最优分量
        G = G.subgraph(best_cc).copy()
        key_points_to_visit = [p for p in key_points_to_visit if p in best_cc]
        print(f"📊 选择最优分量：{best_cc_node_count}个节点，{len(key_points_to_visit)}个关键点")

        # 再次校验关键点数量（避免分量截取后不足）
        if len(key_points_to_visit) < 2:
            print("❌ 最优分量中关键点不足2个，补充分量内路网节点")
            cc_road_nodes = [n for n in best_cc if G.nodes[n]['attr_name'] == "道路点"]
            supplement_count = 2 - len(key_points_to_visit)
            if len(cc_road_nodes) >= supplement_count:
                for i, node in enumerate(cc_road_nodes[:supplement_count]):
                    G.nodes[node]['attr_name'] = f"分量补充点_{i + 1}"
                    key_points_to_visit.append(node)
                print(f"✅ 补充{supplement_count}个节点，关键点总数：{len(key_points_to_visit)}")
            else:
                print("❌ 分量内无足够节点补充，无法规划路径")
                return []

    # 6. TSP路径规划（带景观偏好，不形成回路）
    try:
        print(f"\n🚀 开始TSP路径规划（有效关键点：{len(key_points_to_visit)}个）")
        # 确保入口在关键点列表首位（从入口出发）
        if START_POINT in key_points_to_visit and key_points_to_visit[0] != START_POINT:
            key_points_to_visit.remove(START_POINT)
            key_points_to_visit.insert(0, START_POINT)

        # 执行TSP（使用NetworkX近似算法，cycle=False=不回到起点）
        tsp_nodes = nx.approximation.traveling_salesman_problem(
            G,
            nodes=key_points_to_visit,
            weight=lambda u, v, data: weighted_edge_cost(u, v, data, G),  # 带景观偏好的成本
            cycle=False
        )

        # 调整路径：确保终点为出口（若出口在TSP结果中）
        if END_POINT in tsp_nodes and tsp_nodes[-1] != END_POINT:
            tsp_nodes.remove(END_POINT)
            tsp_nodes.append(END_POINT)
        print(f"📊 TSP关键点顺序：{[G.nodes[p]['attr_name'] for p in tsp_nodes]}")

        # 生成完整路径（连接TSP关键点之间的最短路径）
        full_path = []
        for i in range(len(tsp_nodes) - 1):
            source = tsp_nodes[i]
            target = tsp_nodes[i + 1]
            # 跳过无效节点（理论上不会出现）
            if source not in G.nodes or target not in G.nodes:
                print(f"⚠️  节点{source if source not in G.nodes else target}不在路网中，跳过")
                continue
            # 计算两点间最短路径（避障）
            try:
                sub_path = nx.shortest_path(G, source=source, target=target, weight='weight')
                full_path.extend(sub_path[:-1])  # 避免重复添加终点（下一段的起点）
            except nx.NetworkXNoPath:
                print(f"⚠️  {G.nodes[source]['attr_name']} → {G.nodes[target]['attr_name']} 无路径，直接连接")
                full_path.append(source)

        # 添加最后一个关键点的终点
        if tsp_nodes and full_path:
            full_path.append(tsp_nodes[-1])

        # 路径有效性校验与信息输出
        if full_path and len(full_path) >= 2:
            # 计算路径总长度
            total_length = 0.0
            for i in range(len(full_path) - 1):
                p1 = full_path[i]
                p2 = full_path[i + 1]
                total_length += sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            print(f"✅ 路径规划完成：{len(full_path)}个点，总长度{total_length:.2f}m")

            # 调试绘图：标记关键点（紫色星号）
            if DEBUG_GRAPH and ax:
                for point in key_points_to_visit:
                    if point in G.nodes:
                        label = G.nodes[point]['attr_name']
                        ax.scatter(
                            point[0], point[1],
                            color='purple', s=120, marker='*', zorder=12, edgecolor='white', linewidth=1
                        )
                        ax.annotate(
                            label, (point[0], point[1]),
                            xytext=(6, 6), textcoords='offset points',
                            fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='purple', alpha=0.7, edgecolor='none')
                        )
            return full_path
        else:
            print("❌ 生成的路径无效（点数量不足2个）")
            return []

    except Exception as e:
        print(f"❌ TSP路径规划失败: {e}")
        # TSP失败时的备选方案：入口→出口的最短路径（不经过景点）
        print("🔧 尝试备选方案：入口→出口直接最短路径")
        if START_POINT in G.nodes and END_POINT in G.nodes:
            try:
                backup_path = nx.shortest_path(G, source=START_POINT, target=END_POINT, weight='weight')
                print(
                    f"✅ 备选路径生成：{len(backup_path)}个点，总长度{sqrt(sum((backup_path[i + 1][0] - backup_path[i][0]) ** 2 + (backup_path[i + 1][1] - backup_path[i][1]) ** 2 for i in range(len(backup_path) - 1))):.2f}m")
                return backup_path
            except nx.NetworkXNoPath:
                print("❌ 备选方案失败：入口与出口无直接路径")
        traceback.print_exc()
        return []


# -------------------------- 4. 绘图主函数（移除评分相关绘图） --------------------------
def plot_comprehensive_garden():
    """主函数：整合数据读取、路径规划、景观与路径可视化"""
    # 初始化绘图配置（支持中文）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # 1. 读取Excel数据并提取景观元素（道路、建筑、假山、水体、植物）
    road_segments = []  # 道路线段数据
    attractions_data = {elem: [] for elem in ATTRACTION_ELEMENTS}  # 景点数据（用于路径偏好）
    obstacles_data = {elem: [] for elem in HARD_OBSTACLE_ELEMENTS + SOFT_OBSTACLE_ELEMENTS}  # 障碍物数据

    ## 1.1 读取并绘制道路数据
    road_df = read_excel_sheet(EXCEL_PATH, "道路")
    if not road_df.empty:
        road_segments = extract_segment_coords(road_df["segment_coords"])
        # 绘制道路线条
        seg_color, _, label = ELEMENT_STYLE["道路"]
        for i, seg in enumerate(road_segments):
            if len(seg) >= 2:
                ax.plot(
                    [p[0] for p in seg], [p[1] for p in seg],
                    color=seg_color, linewidth=LINE_WIDTH + 0.2,
                    label=label if i == 0 else "", alpha=0.9
                )
    print(f"📊 读取道路数据：{len(road_segments)}条线段")

    ## 1.2 读取并绘制建筑、假山、水体（景点+障碍物）
    for elem in ["半开放建筑", "实体建筑", "假山", "水体"]:
        elem_df = read_excel_sheet(EXCEL_PATH, elem)
        if elem_df.empty:
            print(f"⚠️  未读取到【{elem}】数据，跳过绘制")
            continue
        # 提取线段坐标（景观元素为闭合多边形）
        elem_segments = extract_segment_coords(elem_df["segment_coords"])
        if elem == "水体":
            elem_segments = extract_water_coords(elem_df["segment_coords"])  # 适配水体格式
        # 分类存储数据（景点/障碍物）
        if elem in ATTRACTION_ELEMENTS:
            attractions_data[elem] = elem_segments
        if elem in obstacles_data:
            obstacles_data[elem] = elem_segments

        # 绘制景观元素（线条+填充）
        seg_color, fill_color, label = ELEMENT_STYLE[elem]
        for i, seg in enumerate(elem_segments):
            if len(seg) >= 2:
                # 绘制线条
                ax.plot(
                    [p[0] for p in seg], [p[1] for p in seg],
                    color=seg_color, linewidth=LINE_WIDTH,
                    label=label if i == 0 else "", alpha=0.8
                )
                # 绘制填充（闭合多边形）
                if len(seg) >= 3:
                    ax.fill(
                        [p[0] for p in seg], [p[1] for p in seg],
                        color=fill_color, alpha=0.3 if elem != "实体建筑" else 0.8
                    )
        print(f"📊 绘制【{elem}】：{len(elem_segments)}条线段")

    ## 1.3 读取并绘制植物数据（圆形图形）
    plant_df = read_excel_sheet(EXCEL_PATH, "植物")
    if not plant_df.empty and len(plant_df.columns) >= 2:
        plants = extract_plant_data(plant_df)
        plant_color = "#228B22"  # 植物颜色（绿色）
        for (x, y, radius) in plants:
            ax.add_patch(
                plt.Circle(
                    (x, y), radius,
                    facecolor=plant_color, edgecolor="#006400",  # 深绿边框
                    alpha=PLANT_ALPHA, linewidth=0.5
                )
            )
        # 添加植物图例（避免与其他元素重复）
        from matplotlib.lines import Line2D
        plant_legend = Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=plant_color, markersize=10,
            label="植物"
        )
        ax.add_artist(ax.legend(handles=[plant_legend], loc='upper right', fontsize=9))
        print(f"📊 绘制植物：{len(plants)}株")

    # 2. 执行路径规划
    print("\n=== 开始避障路径规划 ===")
    path = plan_path_covering_attractions_with_obstacles(
        road_segments=road_segments,
        attractions_data=attractions_data,
        obstacles_data=obstacles_data,
        ax=ax
    )

    # 3. 绘制规划路径（红色实线，置于顶层）
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        # 绘制路径线条
        ax.plot(
            path_x, path_y,
            color=PATH_COLOR, linewidth=LINE_WIDTH + 0.3,
            linestyle=PATH_STYLE, label="规划路径",
            zorder=10  # 路径置于顶层，避免被景观元素遮挡
        )
        # 标记入口（绿色三角形）和出口（蓝色倒三角形）
        ax.scatter(
            path[0][0], path[0][1],
            color="green", s=250, marker="^",
            label="入口", zorder=11, edgecolor="black", linewidth=1.5
        )
        ax.scatter(
            path[-1][0], path[-1][1],
            color="blue", s=250, marker="v",
            label="出口", zorder=11, edgecolor="black", linewidth=1.5
        )
        print("✅ 路径绘制完成")
    else:
        print("❌ 无有效路径可绘制")

    # 4. 图像样式优化（坐标轴、图例、标题）
    ax.set_xlabel("X坐标（米）", fontsize=12)
    ax.set_ylabel("Y坐标（米）", fontsize=12)
    ax.set_title("园林景观与避障路径规划", fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.axis("equal")  # 等比例坐标，避免图形拉伸

    # 整合图例（去重，置于图外右侧）
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))  # 去重重复图例
    ax.legend(
        by_label.values(), by_label.keys(),
        loc="upper left", bbox_to_anchor=(1.02, 1),
        fontsize=10, frameon=True, fancybox=True, shadow=True
    )

    # 调整布局（预留图例空间）
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    # 显示图像
    plt.show()


# -------------------------- 5. 主程序执行 --------------------------
if __name__ == "__main__":
    # 检查核心依赖库是否安装（避免因缺失库崩溃）
    try:
        import pandas
        import matplotlib
        import networkx
        import shapely
        from openpyxl import load_workbook
    except ImportError as e:
        missing_lib = str(e).split("No module named ")[-1].strip("'")
        print(f"❌ 缺少依赖库：{missing_lib}，请先安装（命令：pip install {missing_lib}）")
    else:
        # 执行景观绘制与路径规划
        plot_comprehensive_garden()
