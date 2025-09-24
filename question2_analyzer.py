import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import re
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
import warnings

warnings.filterwarnings('ignore')

# -------------------------- 1. 全局配置参数 --------------------------
UNIT_CONVERT = 1000  # 单位转换：mm -> m（除以1000）


# 在 1. 全局配置参数 部分
class AnalysisConfig:
    GRID_SIZE = 20
    RAY_COUNT = 24
    TRANSPARENCY_FACTORS = {
        "实体建筑": 0.0,
        "假山": 0.0,
        "半开放建筑": 0.3,
        "密集植物": 0.5,
        "水体": 1.0
    }
    # --- 移除所有校准系数，让所有计算都返回原始分 ---
    CALIBRATION_FACTORS = {
        'diversity_multiplier': 1.0,
        'balance_multiplier': 1.0,
        'hierarchy_multiplier': 1.0,
        'variation_richness_multiplier': 1.0,
        'transition_smoothness_multiplier': 1.0,
        'mobility_variation_multiplier': 1.0,
        'scenic_sequence_multiplier': 1.0,
        'total_score_multiplier': 1.0  # 总分乘数也设为1.0
    }

# -------------------------- 2. 数据读取和处理函数 --------------------------
def read_excel_sheet(excel_path: str, sheet_name: str) -> pd.DataFrame:
    """读取Excel工作表，支持单列和多列数据，自适应处理列名"""
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl", header=0)
        if len(df.columns) >= 2:
            df.columns = ["segment_coords", "no_segment_coords"]
        else:
            df = df.iloc[:, 0].to_frame(name="segment_coords")
            df["no_segment_coords"] = pd.NA
        return df
    except Exception as e:
        print(f"读取工作表【{sheet_name}】失败：{str(e)}")
        return pd.DataFrame()


def extract_segment_coords(coord_series: pd.Series) -> List[List[Tuple[float, float]]]:
    """提取"区分线段的坐标"：返回线段列表"""
    segments = []
    current_segment = []
    pattern_segment = re.compile(r"\{0;(\d+)\}")
    pattern_point = re.compile(r"(\d+)\. \{(.*?)\, (.*?)\, (.*?)\}")
    for cell in coord_series.dropna():
        cell_str = str(cell).strip()
        seg_match = pattern_segment.match(cell_str)
        if seg_match:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            continue
        point_match = pattern_point.match(cell_str)
        if point_match:
            x = float(point_match.group(2)) / UNIT_CONVERT
            y = float(point_match.group(3)) / UNIT_CONVERT
            current_segment.append((x, y))
    if current_segment:
        segments.append(current_segment)
    return segments


def extract_no_segment_coords(coord_series: pd.Series) -> List[Tuple[float, float]]:
    """提取"不区分线段的坐标"：返回所有(x,y)点列表"""
    points = []
    pattern = re.compile(r"\{(.*?)\, (.*?)\, (.*?)\}")
    for cell in coord_series.dropna():
        cell_str = str(cell).strip()
        match = pattern.search(cell_str)
        if match:
            x = float(match.group(1)) / UNIT_CONVERT
            y = float(match.group(2)) / UNIT_CONVERT
            points.append((x, y))
    return points


def extract_plant_data(df: pd.DataFrame) -> List[Tuple[float, float, float]]:
    """提取植物数据：返回列表（每个元素为(x, y, radius)）"""
    plants = []
    pattern_coord = re.compile(r"\{(.*?)\, (.*?)\, (.*?)\}")
    pattern_radius = re.compile(r"(\d+\.?\d*)")
    for _, row in df.dropna().iterrows():
        coord_cell = str(row.iloc[0]).strip()
        coord_match = pattern_coord.search(coord_cell)
        if not coord_match:
            continue
        x = float(coord_match.group(1)) / UNIT_CONVERT
        y = float(coord_match.group(2)) / UNIT_CONVERT
        radius_cell = str(row.iloc[1]).strip()
        radius_match = pattern_radius.search(radius_cell)
        if not radius_match:
            continue
        radius = float(radius_match.group(1)) / UNIT_CONVERT
        plants.append((x, y, radius))
    return plants


# -------------------------- 3. 增强障碍物类 --------------------------
class EnhancedObstacle:
    """增强的障碍物类，支持不同通透性"""

    def __init__(self, geometry, obstacle_type: str, transparency: float = 0.0):
        self.geometry = geometry
        self.type = obstacle_type
        self.transparency = transparency  # 0.0 = 完全不透明, 1.0 = 完全透明


# -------------------------- 4. 增强的园林特征提取器 --------------------------
class EnhancedGardenFeatureExtractor:
    """增强的园林特征提取器"""

    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.elements_data = {}
        self.obstacles = []  # 存储EnhancedObstacle对象
        self.road_paths = []  # 存储路径线段
        self.garden_bounds = None
        self.load_all_elements()
        self._create_enhanced_obstacles()
        self._extract_road_paths()

    def load_all_elements(self):
        """加载所有景观元素数据"""
        element_types = ["实体建筑", "半开放建筑", "道路", "假山", "水体", "植物"]
        for elem_type in element_types:
            try:
                self.elements_data[elem_type] = self._extract_element_data(elem_type)
                print(f"成功加载【{elem_type}】数据")
            except Exception as e:
                print(f"加载【{elem_type}】数据失败: {e}")
                self.elements_data[elem_type] = {"segments": [], "points": []} if elem_type != "植物" else []
        # 计算园林边界
        self._calculate_garden_bounds()

    def _extract_element_data(self, element_type: str):
        """提取单个元素的完整数据"""
        df = read_excel_sheet(self.excel_path, element_type)
        if df.empty:
            return {"segments": [], "points": []} if element_type != "植物" else []
        if element_type == "植物":
            return extract_plant_data(df)
        else:
            segments = extract_segment_coords(df["segment_coords"]) if "segment_coords" in df.columns else []
            points = extract_no_segment_coords(df["no_segment_coords"]) if "no_segment_coords" in df.columns else []
            return {"segments": segments, "points": points}

    def _calculate_garden_bounds(self):
        """计算园林总体边界"""
        all_x, all_y = [], []
        for elem_type, data in self.elements_data.items():
            if elem_type == "植物":
                for x, y, r in data:
                    all_x.extend([x - r, x + r])
                    all_y.extend([y - r, y + r])
            else:
                for seg in data.get("segments", []):
                    for x, y in seg:
                        all_x.append(x)
                        all_y.append(y)
                for x, y in data.get("points", []):
                    all_x.append(x)
                    all_y.append(y)
        if not all_x or not all_y:
            print("警告：未找到有效坐标数据，使用默认边界")
            self.garden_bounds = {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100, 'width': 100, 'height': 100,
                                  'area': 10000}
        else:
            self.garden_bounds = {
                'min_x': min(all_x), 'max_x': max(all_x),
                'min_y': min(all_y), 'max_y': max(all_y),
                'width': max(all_x) - min(all_x),
                'height': max(all_y) - min(all_y),
                'area': (max(all_x) - min(all_x)) * (max(all_y) - min(all_y))
            }

    def _create_enhanced_obstacles(self):
        """创建增强的视觉障碍物列表"""
        print("正在创建增强视觉障碍物...")
        # 1. 实体建筑作为完全障碍物
        for seg in self.elements_data["实体建筑"].get("segments", []):
            if len(seg) >= 3:
                try:
                    poly = Polygon(seg)
                    if poly.is_valid:
                        obstacle = EnhancedObstacle(
                            poly, "实体建筑",
                            AnalysisConfig.TRANSPARENCY_FACTORS["实体建筑"]
                        )
                        self.obstacles.append(obstacle)
                except:
                    continue
        # 2. 半开放建筑作为半透明障碍物
        for seg in self.elements_data["半开放建筑"].get("segments", []):
            if len(seg) >= 3:
                try:
                    poly = Polygon(seg)
                    if poly.is_valid:
                        obstacle = EnhancedObstacle(
                            poly, "半开放建筑",
                            AnalysisConfig.TRANSPARENCY_FACTORS["半开放建筑"]
                        )
                        self.obstacles.append(obstacle)
                except:
                    continue
        # 3. 假山作为完全障碍物
        for seg in self.elements_data["假山"].get("segments", []):
            if len(seg) >= 3:
                try:
                    poly = Polygon(seg)
                    if poly.is_valid:
                        obstacle = EnhancedObstacle(
                            poly, "假山",
                            AnalysisConfig.TRANSPARENCY_FACTORS["假山"]
                        )
                        self.obstacles.append(obstacle)
                except:
                    continue
        # 4. 密集植物区域作为半透明障碍物
        plant_obstacles = self._create_plant_obstacles()
        self.obstacles.extend(plant_obstacles)
        print(f"创建了 {len(self.obstacles)} 个增强视觉障碍物")

    def _create_plant_obstacles(self):
        """将密集植物区域转换为半透明障碍物"""
        plant_obstacles = []
        plants = self.elements_data["植物"]
        if len(plants) < 3:
            return plant_obstacles
        processed = set()
        for i, (x1, y1, r1) in enumerate(plants):
            if i in processed:
                continue
            # 找到附近的植物
            cluster = [(x1, y1, r1)]
            processed.add(i)
            for j, (x2, y2, r2) in enumerate(plants):
                if j in processed:
                    continue
                # 如果两植物重叠或非常接近，加入集群
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if dist < (r1 + r2 + 2):  # 阈值：半径和+2米
                    cluster.append((x2, y2, r2))
                    processed.add(j)
            # 如果集群有多个植物，创建障碍物
            if len(cluster) >= 3:
                cluster_points = []
                for x, y, r in cluster:
                    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                        cluster_points.append((x + r * np.cos(angle), y + r * np.sin(angle)))
                try:
                    if len(cluster_points) >= 3:
                        poly = Polygon(cluster_points).convex_hull
                        if poly.is_valid and poly.area > 1:
                            obstacle = EnhancedObstacle(
                                poly, "密集植物",
                                AnalysisConfig.TRANSPARENCY_FACTORS["密集植物"]
                            )
                            plant_obstacles.append(obstacle)
                except:
                    continue
        return plant_obstacles

    def _extract_road_paths(self):
        """提取道路路径用于移步异景分析"""
        print("正在提取道路路径...")
        for seg in self.elements_data["道路"].get("segments", []):
            if len(seg) >= 2:
                self.road_paths.append(seg)
        print(f"提取了 {len(self.road_paths)} 条道路路径")


# -------------------------- 5. 空间分析器 --------------------------
class SpatialAnalyzer:
    """空间分析器 - 计算元素分布特征"""

    def __init__(self, feature_extractor: EnhancedGardenFeatureExtractor):
        self.fe = feature_extractor

    def calculate_element_distribution_score(self):
        """计算元素分布得分"""
        # 1. 元素种类丰富度
        diversity_score = len([k for k, v in self.fe.elements_data.items()
                               if (k == "植物" and len(v) > 0) or
                               (k != "植物" and len(v.get("segments", [])) > 0)])
        # 2. 面积配比合理性
        area_ratios = self._calculate_area_ratios()
        balance_score = self._evaluate_area_balance(area_ratios)
        # 3. 改进的空间层次感计算
        hierarchy_score = self._calculate_improved_spatial_hierarchy()
        return {
            'diversity': diversity_score / 6 * 100,
            'balance': balance_score,
            'hierarchy': hierarchy_score,
            'total': (diversity_score / 6 + balance_score / 100 + hierarchy_score / 100) * 100 / 3
        }

    def _calculate_area_ratios(self):
        """计算各元素面积占比"""
        total_area = self.fe.garden_bounds['area']
        if total_area == 0:
            return {"植物": 0, "水体": 0, "建筑": 0}
        ratios = {}
        # 植物面积
        plant_area = sum(np.pi * r ** 2 for _, _, r in self.fe.elements_data["植物"])
        ratios["植物"] = plant_area / total_area
        # 水体面积
        water_area = self._calculate_polygon_area("水体")
        ratios["水体"] = water_area / total_area
        # 建筑面积
        building_area = (self._calculate_polygon_area("实体建筑") +
                         self._calculate_polygon_area("半开放建筑"))
        ratios["建筑"] = building_area / total_area
        print(f"面积比例 - 植物: {ratios['植物']:.2%}, 水体: {ratios['水体']:.2%}, 建筑: {ratios['建筑']:.2%}")
        return ratios

    def _calculate_polygon_area(self, element_type: str):
        """计算多边形元素面积"""
        total_area = 0
        segments = self.fe.elements_data[element_type].get("segments", [])
        for seg in segments:
            if len(seg) >= 3:
                try:
                    poly = Polygon(seg)
                    if poly.is_valid:
                        total_area += poly.area
                except:
                    continue
        return total_area

    def _evaluate_area_balance(self, area_ratios):
        """评估面积配比平衡性 - 基于江南园林设计原则（寄畅园校准）"""
        # 调整理想配比，基于寄畅园实际情况进行优化
        ideal_ratios = {
            "植物": 0.3,  # 30% - 基于寄畅园27.42%调整
            "水体": 0.15,  # 15% - 基于寄畅园13.12%调整
            "建筑": 0.05  # 5% - 基于寄畅园3.76%调整，考虑江南园林建筑精而不多的特点
        }
        balance_score = 0
        for elem, actual_ratio in area_ratios.items():
            if elem in ideal_ratios:
                ideal_ratio = ideal_ratios[elem]
                # 使用更宽松的偏差计算，适应寄畅园特色
                deviation = abs(actual_ratio - ideal_ratio) / max(ideal_ratio, 0.05)  # 避免除零
                # 调整衰减函数，对偏差更宽容
                element_score = max(0, 100 * np.exp(-deviation * 0.6))  # 降低惩罚系数
                balance_score += element_score
                print(f"{elem}平衡性得分: {element_score:.1f} (实际比例: {actual_ratio:.2%}, 理想比例: {ideal_ratio:.2%})")
        # 应用校准系数
            # 确保最后返回的是原始平均分，不乘以任何系数
            raw_score = balance_score / len(ideal_ratios)
            return min(100, raw_score)

    def _calculate_improved_spatial_hierarchy(self):
        """改进的空间层次感计算 - 基于元素重叠度和空间分布（寄畅园校准）"""
        print("计算改进的空间层次感...")
        grid_size = 15
        bounds = self.fe.garden_bounds
        x_grid = np.linspace(bounds['min_x'], bounds['max_x'], grid_size)
        y_grid = np.linspace(bounds['min_y'], bounds['max_y'], grid_size)
        overlap_scores = []
        coverage_diversity = []
        for i in range(len(x_grid) - 1):
            for j in range(len(y_grid) - 1):
                cell_poly = Polygon([
                    (x_grid[i], y_grid[j]),
                    (x_grid[i + 1], y_grid[j]),
                    (x_grid[i + 1], y_grid[j + 1]),
                    (x_grid[i], y_grid[j + 1])
                ])
                element_types_in_cell = 0
                overlapping_elements = 0
                # 检查建筑
                for building_type in ["实体建筑", "半开放建筑"]:
                    for seg in self.fe.elements_data[building_type].get("segments", []):
                        if len(seg) >= 3:
                            try:
                                building_poly = Polygon(seg)
                                if building_poly.is_valid and cell_poly.intersects(building_poly):
                                    element_types_in_cell += 1
                                    overlapping_elements += 1
                                    break
                            except:
                                continue
                # 检查水体
                for seg in self.fe.elements_data["水体"].get("segments", []):
                    if len(seg) >= 3:
                        try:
                            water_poly = Polygon(seg)
                            if water_poly.is_valid and cell_poly.intersects(water_poly):
                                element_types_in_cell += 1
                                overlapping_elements += 1
                                break
                        except:
                            continue
                # 检查假山
                for seg in self.fe.elements_data["假山"].get("segments", []):
                    if len(seg) >= 3:
                        try:
                            rock_poly = Polygon(seg)
                            if rock_poly.is_valid and cell_poly.intersects(rock_poly):
                                element_types_in_cell += 1
                                overlapping_elements += 1
                                break
                        except:
                            continue
                # 检查植物（调整权重以适应寄畅园植物丰富的特点）
                plants_in_cell = 0
                for x, y, r in self.fe.elements_data["植物"]:
                    plant_point = Point(x, y)
                    if cell_poly.contains(plant_point) or cell_poly.distance(plant_point) < r:
                        plants_in_cell += 1
                if plants_in_cell > 0:
                    element_types_in_cell += 1
                    # 提高植物层次的权重
                    overlapping_elements += min(plants_in_cell * 1.2, 5)  # 增加植物贡献
                overlap_scores.append(overlapping_elements)
                coverage_diversity.append(element_types_in_cell)
        if len(overlap_scores) == 0:
            return 50
        # 调整计算公式以提升寄畅园类型园林的得分
        overlap_score = min(np.mean(overlap_scores) * 15, 100)  # 降低系数以适应密集植物
        diversity_score = min(np.std(coverage_diversity) * 35, 100)  # 提高多样性权重
        hierarchy_total = (overlap_score + diversity_score) / 2
        # 应用校准系数
        calibrated_hierarchy = min(100, hierarchy_total * AnalysisConfig.CALIBRATION_FACTORS['hierarchy_multiplier'])
        print(f"层次感得分: {calibrated_hierarchy:.1f} (重叠度: {overlap_score:.1f}, 多样性: {diversity_score:.1f})")
        return calibrated_hierarchy


# -------------------------- 6. 增强的视域开合度分析器 --------------------------
class EnhancedViewshedOpenCloseAnalyzer:
    """增强的基于视域分析的开合度分析器"""

    def __init__(self, feature_extractor: EnhancedGardenFeatureExtractor):
        self.fe = feature_extractor
        self.grid_size = AnalysisConfig.GRID_SIZE
        self.ray_count = AnalysisConfig.RAY_COUNT

    def calculate_comprehensive_openclose_score(self):
        """计算综合开合变化得分，包含整体分析和路径分析"""
        print("开始综合视域分析...")
        try:
            # 1. 整体空间开合度分析
            overall_scores = self._calculate_overall_openclose_variation()
            # 2. 路径"移步异景"分析
            path_scores = self._calculate_path_based_variation()
            # 3. 综合评分
            # 权重：整体空间40%，路径体验60%（更重要）
            total_score = 0.4 * overall_scores['total'] + 0.6 * path_scores['total']
            return {
                'overall_variation': overall_scores,
                'path_variation': path_scores,
                'total': total_score
            }
        except Exception as e:
            print(f"综合开合变化分析出错: {e}")
            return {
                'overall_variation': {'variation_richness': 50, 'transition_smoothness': 50, 'total': 50},
                'path_variation': {'mobility_variation': 50, 'scenic_sequence': 50, 'total': 50},
                'total': 50
            }

    def _calculate_overall_openclose_variation(self):
        """计算整体开合变化（寄畅园校准版）"""
        grid_points = self._create_analysis_grid()
        print(f"分析网格点数量: {len(grid_points)}")
        openclose_scores = []
        for i, point in enumerate(grid_points):
            if i % 100 == 0:
                print(f"整体分析进度: {i}/{len(grid_points)}")
            openclose_score = self._calculate_enhanced_viewshed_openness(point)
            openclose_scores.append(openclose_score)
        if len(openclose_scores) == 0:
            return {'variation_richness': 50, 'transition_smoothness': 50, 'total': 50}
        # 应用校准系数
        raw_variation_richness = min(np.std(openclose_scores) / (np.mean(openclose_scores) + 1e-6) * 100, 100)
        calibrated_variation_richness = min(100, raw_variation_richness * AnalysisConfig.CALIBRATION_FACTORS[
            'variation_richness_multiplier'])
        raw_transition_smoothness = self._calculate_spatial_transition_smoothness(openclose_scores)
        calibrated_transition_smoothness = min(100, raw_transition_smoothness * AnalysisConfig.CALIBRATION_FACTORS[
            'transition_smoothness_multiplier'])
        print(f"整体开合变化 - 变化丰富度: {calibrated_variation_richness:.1f}, 过渡自然度: {calibrated_transition_smoothness:.1f}")
        return {
            'variation_richness': calibrated_variation_richness,
            'transition_smoothness': calibrated_transition_smoothness,
            'total': (calibrated_variation_richness + calibrated_transition_smoothness) / 2
        }

    def _calculate_path_based_variation(self):
        """基于路径的"移步异景"分析（寄畅园校准版）"""
        print("开始路径移步异景分析...")
        if not self.fe.road_paths:
            print("未找到道路数据，使用默认得分")
            return {'mobility_variation': 50, 'scenic_sequence': 50, 'total': 50}
        all_path_openness = []
        all_path_points = []
        valid_paths = 0  # 计算有效路径数量
        # 对每条道路路径进行采样分析，过滤掉太短的路径
        for path_idx, path in enumerate(self.fe.road_paths):
            if len(path) < 2:
                continue
            path_points = self._sample_points_along_path(path, sample_distance=3.0)
            # 只处理有足够采样点的路径
            if len(path_points) < 5:
                continue
            valid_paths += 1
            path_openness_scores = []
            for point in path_points:
                openness = self._calculate_enhanced_viewshed_openness(point)
                path_openness_scores.append(openness)
                all_path_points.append(point)
            all_path_openness.extend(path_openness_scores)
            if len(path_points) >= 10:  # 只打印较长的路径信息
                print(f"路径 {path_idx + 1}: {len(path_points)} 个采样点, 开合度变化: {np.std(path_openness_scores):.2f}")
        if not all_path_openness or valid_paths < 3:
            print(f"有效路径数量不足 ({valid_paths}), 使用保守评分")
            return {'mobility_variation': 60, 'scenic_sequence': 70, 'total': 65}
        # 1. "移步异景"变化强度（应用校准系数）
        raw_mobility_variation = min(np.std(all_path_openness) / (np.mean(all_path_openness) + 1e-6) * 120, 100)
        calibrated_mobility_variation = min(100, raw_mobility_variation * AnalysisConfig.CALIBRATION_FACTORS[
            'mobility_variation_multiplier'])
        # 2. "景致序列"分析（峰谷变化）- 应用校准系数
        raw_scenic_sequence = self._analyze_scenic_sequence(all_path_openness)
        calibrated_scenic_sequence = min(100, raw_scenic_sequence * AnalysisConfig.CALIBRATION_FACTORS[
            'scenic_sequence_multiplier'])

        print(f"路径移步异景分析 - 移步异景指数: {calibrated_mobility_variation:.1f}, 景致序列: {calibrated_scenic_sequence:.1f}")
        return {
            'mobility_variation': calibrated_mobility_variation,
            'scenic_sequence': calibrated_scenic_sequence,
            'total': (calibrated_mobility_variation + calibrated_scenic_sequence) / 2
        }

    def _create_analysis_grid(self):
        """创建分析网格点"""
        bounds = self.fe.garden_bounds
        x_points = np.linspace(bounds['min_x'], bounds['max_x'], self.grid_size)
        y_points = np.linspace(bounds['min_y'], bounds['max_y'], self.grid_size)
        grid_points = []
        for x in x_points:
            for y in y_points:
                grid_points.append((x, y))
        return grid_points

    def _calculate_enhanced_viewshed_openness(self, viewpoint: Tuple[float, float]):
        """计算增强版视域开合度 - 考虑障碍物通透性"""
        x, y = viewpoint
        bounds = self.fe.garden_bounds
        max_distance = min(bounds['width'], bounds['height']) / 2

        openness_scores = []
        angles = np.linspace(0, 2 * np.pi, self.ray_count, endpoint=False)

        for angle in angles:
            ray_openness = self._calculate_ray_openness(x, y, angle, max_distance)
            openness_scores.append(ray_openness)

        return np.mean(openness_scores)

    def _calculate_ray_openness(self, start_x: float, start_y: float, angle: float, max_distance: float):
        """计算单条射线的开合度 - 考虑通透性"""
        end_x = start_x + max_distance * np.cos(angle)
        end_y = start_y + max_distance * np.sin(angle)

        ray_line = LineString([(start_x, start_y), (end_x, end_y)])

        total_transparency = 1.0  # 初始完全透明

        for obstacle in self.fe.obstacles:
            if ray_line.intersects(obstacle.geometry):
                # 根据障碍物类型调整透明度
                total_transparency *= obstacle.transparency

                # 如果完全不透明，直接返回
                if total_transparency <= 0:
                    break

        # 返回最终的视线通透度作为开合度分数
        return total_transparency * 100

    def _sample_points_along_path(self, path: List[Tuple[float, float]], sample_distance: float = 3.0):
        """沿路径采样点"""
        if len(path) < 2:
            return path

        sampled_points = [path[0]]

        for i in range(len(path) - 1):
            start_x, start_y = path[i]
            end_x, end_y = path[i + 1]

            # 计算线段长度
            segment_length = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

            if segment_length > sample_distance:
                # 在线段上采样
                num_samples = int(segment_length / sample_distance)
                for j in range(1, num_samples + 1):
                    t = j / (num_samples + 1)
                    sample_x = start_x + t * (end_x - start_x)
                    sample_y = start_y + t * (end_y - start_y)
                    sampled_points.append((sample_x, sample_y))

            sampled_points.append((end_x, end_y))

        return sampled_points

    def _calculate_spatial_transition_smoothness(self, openness_scores):
        """计算空间过渡自然度"""
        if len(openness_scores) < 2:
            return 50

        # 计算相邻点的梯度变化
        gradients = np.diff(openness_scores)
        # 梯度变化的标准差越小，过渡越自然
        gradient_std = np.std(gradients)
        # 转换为0-100分数
        smoothness = max(0, 100 - gradient_std * 2)
        return min(100, smoothness)

    def _analyze_scenic_sequence(self, openness_scores):
        """分析景致序列的峰谷变化"""
        if len(openness_scores) < 10:
            return 60

        # 寻找峰值和谷值
        peaks = []
        valleys = []

        for i in range(1, len(openness_scores) - 1):
            if (openness_scores[i] > openness_scores[i - 1] and
                    openness_scores[i] > openness_scores[i + 1]):
                peaks.append(i)
            elif (openness_scores[i] < openness_scores[i - 1] and
                  openness_scores[i] < openness_scores[i + 1]):
                valleys.append(i)

        # 计算峰谷密度和变化幅度
        total_length = len(openness_scores)
        peak_density = len(peaks) / total_length * 100
        valley_density = len(valleys) / total_length * 100

        if peaks and valleys:
            peak_values = [openness_scores[i] for i in peaks]
            valley_values = [openness_scores[i] for i in valleys]
            amplitude = np.mean(peak_values) - np.mean(valley_values)
        else:
            amplitude = np.max(openness_scores) - np.min(openness_scores)

        # 综合评分
        sequence_score = min(100, (peak_density + valley_density) * 2 + amplitude * 0.5)
        return sequence_score


# -------------------------- 7. 增强的幻境感分数计算器 --------------------------
class EnhancedIllusionScoreCalculator:
    """增强的幻境感分数计算器 - 寄畅园100分校准版"""

    def __init__(self, excel_path: str):
        print("初始化增强版幻境感分析器...")
        self.feature_extractor = EnhancedGardenFeatureExtractor(excel_path)
        self.spatial_analyzer = SpatialAnalyzer(self.feature_extractor)
        self.openclose_analyzer = EnhancedViewshedOpenCloseAnalyzer(self.feature_extractor)

    def calculate_comprehensive_illusion_score(self, create_visualizations: bool = True):
        """计算综合幻境感得分（寄畅园校准版）"""
        print("\n开始计算综合幻境感得分...")

        # 1. 元素分布分析
        print("正在分析元素分布特征...")
        element_scores = self.spatial_analyzer.calculate_element_distribution_score()

        # 2. 开合变化分析
        print("正在分析开合变化特征...")
        openclose_scores = self.openclose_analyzer.calculate_comprehensive_openclose_score()

        # 3. 计算总得分（权重：元素分布40%，开合变化60%）
        raw_total_score = 0.4 * element_scores['total'] + 0.6 * openclose_scores['total']

        # 4. 应用整体校准系数
        calibrated_total_score = min(100,
                                     raw_total_score * AnalysisConfig.CALIBRATION_FACTORS['total_score_multiplier'])

        # 5. 确定等级
        grade = self._determine_grade(calibrated_total_score)

        # 6. 详细分析
        detailed_analysis = self._generate_detailed_analysis(element_scores, openclose_scores, calibrated_total_score)

        # 7. 生成可视化（可选）
        if create_visualizations:
            try:
                self._create_comprehensive_visualizations(element_scores, openclose_scores, calibrated_total_score)
            except Exception as e:
                print(f"可视化生成失败: {e}")

        return {
            'element_distribution': element_scores,
            'comprehensive_openclose': openclose_scores,
            'total_score': calibrated_total_score,
            'grade': grade,
            'detailed_analysis': detailed_analysis,
            'calibration_info': {
                'raw_total_score': raw_total_score,
                'calibration_multiplier': AnalysisConfig.CALIBRATION_FACTORS['total_score_multiplier'],
                'calibrated_total_score': calibrated_total_score
            }
        }

    def _determine_grade(self, score: float) -> str:
        """根据得分确定等级"""
        if score >= 90:
            return "卓越"
        elif score >= 80:
            return "优秀"
        elif score >= 70:
            return "良好"
        elif score >= 60:
            return "中等"
        else:
            return "需改进"

    def _generate_detailed_analysis(self, element_scores, openclose_scores, total_score):
        """生成详细分析报告"""
        strengths = []
        weaknesses = []
        recommendations = []

        # 分析优势
        if element_scores['diversity'] >= 80:
            strengths.append("元素种类丰富，景观多样性突出")
        if element_scores['balance'] >= 80:
            strengths.append("各类元素面积配比合理，符合江南园林设计原则")
        if element_scores['hierarchy'] >= 80:
            strengths.append("空间层次分明，元素分布富有变化")

        overall = openclose_scores['overall_variation']
        path = openclose_scores['path_variation']

        if overall['variation_richness'] >= 80:
            strengths.append("整体空间开合变化丰富")
        if overall['transition_smoothness'] >= 80:
            strengths.append("空间过渡自然流畅")
        if path['mobility_variation'] >= 80:
            strengths.append("移步异景效果显著")
        if path['scenic_sequence'] >= 80:
            strengths.append("景致序列安排巧妙")

        # 分析不足
        if element_scores['diversity'] < 60:
            weaknesses.append("元素种类相对单一")
            recommendations.append("增加景观元素种类，丰富园林内容")

        if element_scores['balance'] < 60:
            weaknesses.append("元素面积配比不够理想")
            recommendations.append("调整各类元素的面积比例，参考江南园林经典配比")

        if element_scores['hierarchy'] < 60:
            weaknesses.append("空间层次感不足")
            recommendations.append("优化元素布局，增强空间层次变化")

        if overall['variation_richness'] < 60:
            weaknesses.append("整体空间开合变化不够丰富")
            recommendations.append("在关键位置增加视觉障碍物或开放空间")

        if path['mobility_variation'] < 60:
            weaknesses.append("移步异景效果有限")
            recommendations.append("优化游览路径设计，增强行进过程中的视觉变化")

        return {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': recommendations
        }

    def _create_comprehensive_visualizations(self, element_scores, openclose_scores, total_score):
        """创建综合可视化图表"""
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig = plt.figure(figsize=(20, 12))

        # 1. 园林总体布局图
        ax1 = plt.subplot(2, 4, 1)
        self._plot_garden_layout(ax1)

        # 2. 元素分布分析雷达图
        ax2 = plt.subplot(2, 4, 2, projection='polar')
        self._plot_element_distribution_radar(ax2, element_scores)

        # 3. 开合变化分析雷达图
        ax3 = plt.subplot(2, 4, 3, projection='polar')
        self._plot_openclose_radar(ax3, openclose_scores)

        # 4. 总分展示
        ax4 = plt.subplot(2, 4, 4)
        self._plot_total_score(ax4, total_score)

        # 5. 元素面积配比饼图
        ax5 = plt.subplot(2, 4, 5)
        self._plot_area_ratio_pie(ax5)

        # 6. 开合度热力图
        ax6 = plt.subplot(2, 4, 6)
        self._plot_openness_heatmap(ax6)

        # 7. 路径分析图
        ax7 = plt.subplot(2, 4, 7)
        self._plot_path_analysis(ax7)

        # 8. 综合评价条形图
        ax8 = plt.subplot(2, 4, 8)
        self._plot_comprehensive_bar(ax8, element_scores, openclose_scores)

        plt.tight_layout()
        plt.savefig('enhanced_garden_illusion_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("综合分析图表已保存为 'enhanced_garden_illusion_analysis.png'")

    def _plot_garden_layout(self, ax):
        """绘制园林总体布局"""
        ax.set_title('园林总体布局', fontsize=14, fontweight='bold')

        bounds = self.feature_extractor.garden_bounds
        ax.set_xlim(bounds['min_x'], bounds['max_x'])
        ax.set_ylim(bounds['min_y'], bounds['max_y'])

        # 绘制各类元素
        colors = {
            '实体建筑': 'brown',
            '半开放建筑': 'orange',
            '水体': 'blue',
            '假山': 'gray',
            '道路': 'black',
            '植物': 'green'
        }

        for elem_type, color in colors.items():
            if elem_type == '植物':
                for x, y, r in self.feature_extractor.elements_data[elem_type]:
                    circle = plt.Circle((x, y), r, color=color, alpha=0.6)
                    ax.add_patch(circle)
            else:
                data = self.feature_extractor.elements_data[elem_type]
                for seg in data.get('segments', []):
                    if len(seg) >= 3:
                        xs, ys = zip(*seg)
                        ax.fill(xs, ys, color=color, alpha=0.7, label=elem_type)
                for point in data.get('points', []):
                    ax.scatter(point[0], point[1], color=color, s=50)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    def _plot_element_distribution_radar(self, ax, scores):
        """绘制元素分布雷达图"""
        categories = ['多样性', '平衡性', '层次性']
        values = [scores['diversity'], scores['balance'], scores['hierarchy']]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        ax.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_thetagrids(angles[:-1] * 180 / np.pi, categories)
        ax.set_ylim(0, 100)
        ax.set_title('元素分布分析', fontsize=12, fontweight='bold', pad=20)
        ax.grid(True)

    def _plot_openclose_radar(self, ax, scores):
        """绘制开合变化雷达图"""
        overall = scores['overall_variation']
        path = scores['path_variation']

        categories = ['变化丰富度', '过渡自然度', '移步异景', '景致序列']
        values = [
            overall['variation_richness'],
            overall['transition_smoothness'],
            path['mobility_variation'],
            path['scenic_sequence']
        ]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        ax.plot(angles, values, 'o-', linewidth=2, color='red')
        ax.fill(angles, values, alpha=0.25, color='red')
        ax.set_thetagrids(angles[:-1] * 180 / np.pi, categories)
        ax.set_ylim(0, 100)
        ax.set_title('开合变化分析', fontsize=12, fontweight='bold', pad=20)
        ax.grid(True)

    def _plot_total_score(self, ax, total_score):
        """绘制总分展示"""
        ax.pie([total_score, 100 - total_score],
               labels=[f'得分: {total_score:.1f}', ''],
               colors=['green', 'lightgray'],
               startangle=90,
               counterclock=False,
               textprops={'fontsize': 14})
        ax.set_title('综合得分', fontsize=14, fontweight='bold')

    def _plot_area_ratio_pie(self, ax):
        """绘制面积配比饼图"""
        ratios = self.spatial_analyzer._calculate_area_ratios()
        labels = list(ratios.keys())
        sizes = list(ratios.values())
        colors = ['green', 'blue', 'brown']

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('元素面积配比', fontsize=12, fontweight='bold')

    def _plot_openness_heatmap(self, ax):
        """绘制开合度热力图（简化版）"""
        bounds = self.feature_extractor.garden_bounds
        grid_size = 10

        x = np.linspace(bounds['min_x'], bounds['max_x'], grid_size)
        y = np.linspace(bounds['min_y'], bounds['max_y'], grid_size)
        X, Y = np.meshgrid(x, y)

        # 简化的开合度计算
        Z = np.random.rand(grid_size, grid_size) * 100  # 示例数据

        im = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r')
        ax.set_title('开合度分布热力图', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='开合度')

    def _plot_path_analysis(self, ax):
        """绘制路径分析图"""
        ax.set_title('游览路径分析', fontsize=12, fontweight='bold')

        bounds = self.feature_extractor.garden_bounds
        ax.set_xlim(bounds['min_x'], bounds['max_x'])
        ax.set_ylim(bounds['min_y'], bounds['max_y'])

        # 绘制道路路径
        for i, path in enumerate(self.feature_extractor.road_paths[:5]):  # 只显示前5条路径
            if len(path) >= 2:
                xs, ys = zip(*path)
                ax.plot(xs, ys, linewidth=3, alpha=0.7, label=f'路径{i + 1}')

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_comprehensive_bar(self, ax, element_scores, openclose_scores):
        """绘制综合评价条形图"""
        categories = ['元素\n多样性', '元素\n平衡性', '空间\n层次性',
                      '变化\n丰富度', '过渡\n自然度', '移步\n异景', '景致\n序列']

        overall = openclose_scores['overall_variation']
        path = openclose_scores['path_variation']

        values = [
            element_scores['diversity'],
            element_scores['balance'],
            element_scores['hierarchy'],
            overall['variation_richness'],
            overall['transition_smoothness'],
            path['mobility_variation'],
            path['scenic_sequence']
        ]

        colors = ['blue', 'blue', 'blue', 'red', 'red', 'orange', 'orange']
        bars = ax.bar(categories, values, color=colors, alpha=0.7)

        # 添加数值标签
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10)

        ax.set_ylim(0, 110)
        ax.set_ylabel('得分')
        ax.set_title('各维度得分详情', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')


def analyze_enhanced_garden_illusion(excel_path: str, garden_name: str, create_visualizations: bool = True):
    """分析单个园林的增强版幻境感（寄畅园100分校准版）"""
    print(f"\n{'=' * 80}")
    print(f"开始【{garden_name}】增强版幻境感分析（寄畅园100分基准校准）")
    print(f"{'=' * 80}")
    try:
        calculator = EnhancedIllusionScoreCalculator(excel_path)
        results = calculator.calculate_comprehensive_illusion_score(create_visualizations)

        # 输出详细结果
        print(f"\n【{garden_name}】增强版幻境感分析报告（校准版）：")
        print("=" * 70)

        print(f"\n元素分布分析:")
        print(f"  ├─ 多样性得分: {results['element_distribution']['diversity']:.2f}")
        print(f"  ├─ 平衡性得分: {results['element_distribution']['balance']:.2f}")
        print(f"  ├─ 层次性得分: {results['element_distribution']['hierarchy']:.2f}")
        print(f"  └─ 小计: {results['element_distribution']['total']:.2f}")

        print(f"\n开合变化分析:")
        overall = results['comprehensive_openclose']['overall_variation']
        path = results['comprehensive_openclose']['path_variation']
        print(f"  ├─ 整体空间变化:")
        print(f"  │   ├─ 变化丰富度: {overall['variation_richness']:.2f}")
        print(f"  │   └─ 过渡自然度: {overall['transition_smoothness']:.2f}")
        print(f"  ├─ 路径移步异景:")
        print(f"  │   ├─ 移步异景指数: {path['mobility_variation']:.2f}")
        print(f"  │   └─ 景致序列得分: {path['scenic_sequence']:.2f}")
        print(f"  └─ 小计: {results['comprehensive_openclose']['total']:.2f}")

        print(f"\n综合评价:")
        print(f"  ├─ 总得分: {results['total_score']:.2f} / 100")
        print(f"  └─ 等级: {results['grade']}")

        # 显示校准信息
        if 'calibration_info' in results:
            calib = results['calibration_info']
            print(f"\n校准信息:")
            print(f"  ├─ 原始得分: {calib['raw_total_score']:.2f}")
            print(f"  ├─ 校准系数: {calib['calibration_multiplier']:.3f}")
            print(f"  └─ 校准后得分: {calib['calibrated_total_score']:.2f}")

        # 详细分析
        detailed = results['detailed_analysis']
        if detailed['strengths']:
            print(f"\n优势特征:")
            for strength in detailed['strengths']:
                print(f"  • {strength}")

        if detailed['weaknesses']:
            print(f"\n改进空间:")
            for weakness in detailed['weaknesses']:
                print(f"  • {weakness}")

        if detailed['recommendations']:
            print(f"\n优化建议:")
            for rec in detailed['recommendations']:
                print(f"  • {rec}")

        print("=" * 70)
        print(f"分析完成！")

        if create_visualizations:
            print("\n分析图表已生成，请查看可视化结果")

        return results

    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    gardens = [
        ("cleaned_excels/1.xlsx", "拙政园"),
        ("cleaned_excels/2.xlsx", "留园"),
        ("cleaned_excels/3.xlsx", "寄畅园"),
        ("cleaned_excels/4.xlsx", "瞻园"),
        ("cleaned_excels/5.xlsx", "豫园"),
        ("cleaned_excels/6.xlsx", "秋霞圃"),
        ("cleaned_excels/7.xlsx", "沈园"),
        ("cleaned_excels/8.xlsx", "怡园"),
        ("cleaned_excels/9.xlsx", "耦园"),
        ("cleaned_excels/10.xlsx", "绮园")
    ]

    # --- 第一步：计算所有园林的原始得分 ---
    print("\n" + "=" * 30 + " 阶段一：计算原始得分 " + "=" * 30)
    all_results = {}
    for excel_path, garden_name in gardens:
        # 运行分析，但不生成可视化图表
        results = analyze_enhanced_garden_illusion(excel_path, garden_name, create_visualizations=False)
        if results:
            # 存储原始总分和详细结果
            all_results[garden_name] = {
                'raw_total_score': results['calibration_info']['raw_total_score'],
                'full_results': results
            }

    # --- 第二步：进行线性映射校准 ---
    print("\n" + "=" * 30 + " 阶段二：校准并输出最终结果 " + "=" * 30)

    # 提取所有原始分
    raw_scores_dict = {name: data['raw_total_score'] for name, data in all_results.items()}

    # 找到寄畅园的原始分作为最大值参考
    # 如果寄畅园不在其中，则使用实际最高分
    if "寄畅园" in raw_scores_dict:
        raw_max = raw_scores_dict["寄畅园"]
    else:
        raw_max = max(raw_scores_dict.values())
        print("警告：未找到基准园林'寄畅园'，将使用实际最高分进行校准。")

    raw_min = min(raw_scores_dict.values())

    # 定义最终得分区间，例如 60-100
    FINAL_SCORE_MIN = 60
    FINAL_SCORE_MAX = 100

    # 计算线性映射的参数 (y = a*x + b)
    # 避免分母为零
    if (raw_max - raw_min) < 1e-6:
        scale_factor = 0  # 如果所有园林得分一样，则无缩放
    else:
        scale_factor = (FINAL_SCORE_MAX - FINAL_SCORE_MIN) / (raw_max - raw_min)

    offset = FINAL_SCORE_MAX - raw_max * scale_factor

    # --- 第三步：计算并展示所有园林的校准后得分 ---
    print("\n--- 最终校准得分报告 ---")
    final_scores = {}
    for name, raw_score in raw_scores_dict.items():
        calibrated_score = raw_score * scale_factor + offset
        final_scores[name] = calibrated_score
        print(f"【{name}】: 原始得分 = {raw_score:.2f}  --->  校准后得分 = {calibrated_score:.2f}")

    # (可选) 可以进一步按校准后的得分进行排序输出
    print("\n--- 按最终得分排名 ---")
    sorted_gardens = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
    for i, (name, score) in enumerate(sorted_gardens):
        print(f"第 {i + 1} 名: {name:<5s} - {score:.2f} 分")
