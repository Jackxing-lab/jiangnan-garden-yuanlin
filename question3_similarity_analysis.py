import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import OrderedDict

# -------------------------- 0. 导入你前两题的核心分析器 --------------------------
# 假设你已经将问题二的代码封装成一个名为 "question2_analyzer.py" 的文件
# 从中导入核心的幻境感计算器类
# 如果你的文件名不同，请修改这里的 'question2_analyzer'
try:
    from question2_analyzer import EnhancedIllusionScoreCalculator
except ImportError:
    print("❌ 错误：无法导入 'question2_analyzer.py'。")
    print("请确保将第二题的代码保存为 'question2_analyzer.py' 并且与此脚本在同一目录下。")


    # 如果无法导入，定义一个桩类以避免程序崩溃，但这无法实际运行
    class EnhancedIllusionScoreCalculator:
        def __init__(self, path):
            print(f"警告：正在使用桩实现 EnhancedIllusionScoreCalculator for {path}")

        def calculate_comprehensive_illusion_score(self, create_visualizations=False):
            # 返回一个符合结构的假数据，以便代码结构检查
            return {
                'element_distribution': {
                    'raw_values': {'plant_ratio': 0, 'water_ratio': 0, 'building_ratio': 0, 'overlap_score': 0,
                                   'diversity_score': 0}},
                'comprehensive_openclose': {
                    'raw_values': {'overall_openness_mean': 0, 'overall_openness_std': 0, 'path_openness_mean': 0,
                                   'path_openness_std': 0, 'peak_valley_density': 0, 'peak_valley_amplitude': 0}},
                # 假设问题一的代码可以计算这两个值
                'physical_features': {'total_road_length': 0, 'waterfront_complexity': 0}
            }


# -------------------------- 1. 特征提取器 --------------------------

def extract_aesthetic_features(excel_path: str, garden_name: str) -> dict:
    """
    为单个园林提取用于相似度分析的美学特征向量。
    该函数调用问题二的分析器，并从中提取原始、细粒度的特征值。

    Args:
        excel_path (str): 园林数据Excel文件的路径。
        garden_name (str): 园林名称。

    Returns:
        dict: 包含园林名称和其特征值的字典。
    """
    print(f"\n===== 正在为【{garden_name}】提取美学特征... =====")
    try:
        # 初始化问题二的分析器
        # 注意：为了获取原始特征，可能需要微调你的第二题代码，
        # 让 'calculate_...' 方法返回一个包含原始值的字典。
        # 这里假设你的代码经过了这样的修改。
        calculator = EnhancedIllusionScoreCalculator(excel_path)

        # 为了演示，我们先模拟调用并获取一个包含所有所需原始值的 `results` 字典
        # 在你的实际应用中, 你需要修改第二题代码，让它能返回这样的结构
        results = calculator.get_raw_feature_values()  # 假设有这样一个新方法

        # ----------------- 从结果中提取特征值 -----------------
        # 静态构成维度
        static_features = results['element_distribution']
        plant_ratio = static_features.get('plant_ratio', 0)
        water_ratio = static_features.get('water_ratio', 0)
        building_ratio = static_features.get('building_ratio', 0)
        overlap_score = static_features.get('overlap_score', 0)
        diversity_score = static_features.get('diversity_score', 0)

        # 动态体验维度
        dynamic_features = results['comprehensive_openclose']
        overall_openness_mean = dynamic_features.get('overall_openness_mean', 0)
        overall_openness_std = dynamic_features.get('overall_openness_std', 0)
        path_openness_mean = dynamic_features.get('path_openness_mean', 0)
        path_openness_std = dynamic_features.get('path_openness_std', 0)
        peak_valley_density = dynamic_features.get('peak_valley_density', 0)
        peak_valley_amplitude = dynamic_features.get('peak_valley_amplitude', 0)

        # 基础物理维度 (假设可以从问题一或二的代码中获得)
        physical_features = results['physical_features']
        total_road_length = physical_features.get('total_road_length', 0)
        waterfront_complexity = physical_features.get('waterfront_complexity', 0)

        # ----------------- 组合成特征字典 -----------------
        feature_dict = OrderedDict([
            ('garden_name', garden_name),
            # 静态构成
            ('plant_ratio', plant_ratio),
            ('water_ratio', water_ratio),
            ('building_ratio', building_ratio),
            ('spatial_overlap', overlap_score),
            ('spatial_diversity', diversity_score),
            # 动态体验
            ('global_openness_mean', overall_openness_mean),
            ('global_openness_std', overall_openness_std),
            ('path_openness_mean', path_openness_mean),
            ('path_openness_std', path_openness_std),
            ('sequence_density', peak_valley_density),
            ('sequence_amplitude', peak_valley_amplitude),
            # 物理基础
            ('road_length', total_road_length),
            ('waterfront_complexity', waterfront_complexity)
        ])

        print(f"【{garden_name}】特征提取成功。")
        return feature_dict

    except Exception as e:
        print(f"❌ 为【{garden_name}】提取特征时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


# -------------------------- 2. 主分析流程 --------------------------

class SimilarityAnalyzer:
    def __init__(self, garden_data_list: list):
        """
        初始化相似度分析器。

        Args:
            garden_data_list (list): 包含园林路径和名称的元组列表。
        """
        self.garden_data_list = garden_data_list
        self.feature_df = None
        self.normalized_matrix = None
        self.scaler = MinMaxScaler()
        self.garden_names = [g[1] for g in garden_data_list]
        self.feature_labels = []

    def run_full_analysis(self):
        """
        执行完整的相似度分析流程。
        """
        print("\n\n" + "=" * 30 + " 阶段一：批量提取特征向量 " + "=" * 30)
        self._extract_all_features()
        if self.feature_df is None or self.feature_df.empty:
            print("❌ 特征提取失败，无法继续分析。")
            return

        print("\n\n" + "=" * 30 + " 阶段二：数据标准化 " + "=" * 30)
        self._normalize_features()

        print("\n\n" + "=" * 30 + " 阶段三：计算相似度并可视化 " + "=" * 30)
        self._analyze_and_plot_similarity()

        print("\n\n" + "=" * 30 + " 阶段四：层次聚类分析 " + "=" * 30)
        self._perform_hierarchical_clustering()

    def _extract_all_features(self):
        """
        遍历所有园林，提取它们的特征并整合成一个DataFrame。
        """
        feature_list = []
        # 注意: 这里需要一个修改版的`EnhancedIllusionScoreCalculator`
        # 为了让代码能运行，我将创建模拟数据。
        # 在你的实际使用中，请替换为真实的特征提取调用。
        print("提示：正在使用模拟数据进行演示。请修改您的 Q2 代码以返回真实的原始特征。")
        for i, (path, name) in enumerate(self.garden_data_list):
            # 真实调用应该是:
            # features = extract_aesthetic_features(path, name)
            # if features:
            #     feature_list.append(features)

            # --- 使用模拟数据进行演示 ---
            np.random.seed(i * 42)  # 确保每次运行模拟数据一致
            simulated_features = OrderedDict([
                ('garden_name', name),
                ('plant_ratio', np.random.uniform(0.15, 0.4)),
                ('water_ratio', np.random.uniform(0.05, 0.25)),
                ('building_ratio', np.random.uniform(0.03, 0.1)),
                ('spatial_overlap', np.random.uniform(2.0, 5.0)),
                ('spatial_diversity', np.random.uniform(1.5, 4.0)),
                ('global_openness_mean', np.random.uniform(40, 70)),
                ('global_openness_std', np.random.uniform(10, 30)),
                ('path_openness_mean', np.random.uniform(30, 60)),
                ('path_openness_std', np.random.uniform(15, 35)),
                ('sequence_density', np.random.uniform(10, 25)),
                ('sequence_amplitude', np.random.uniform(20, 50)),
                ('road_length', np.random.uniform(500, 2000)),
                ('waterfront_complexity', np.random.uniform(1.5, 4.5))
            ])
            feature_list.append(simulated_features)
            # --- 模拟数据结束 ---

        if not feature_list:
            return

        self.feature_df = pd.DataFrame(feature_list)
        self.feature_df = self.feature_df.set_index('garden_name')
        self.feature_labels = self.feature_df.columns.tolist()
        print("\n所有园林特征提取完成，原始特征矩阵如下：")
        print(self.feature_df)

    def _normalize_features(self):
        """
        使用Min-Max Scaling对特征矩阵进行归一化。
        """
        # 拟合并转换数据
        self.normalized_matrix = self.scaler.fit_transform(self.feature_df)

        # 将归一化后的数据转为DataFrame以便查看
        normalized_df = pd.DataFrame(self.normalized_matrix,
                                     index=self.garden_names,
                                     columns=self.feature_labels)
        print("\n特征矩阵归一化 (Min-Max Scaling) 完成，结果如下：")
        print(normalized_df)

    def _analyze_and_plot_similarity(self):
        """
        计算余弦相似度矩阵并使用热力图可视化。
        """
        if self.normalized_matrix is None:
            print("❌ 归一化矩阵为空，无法计算相似度。")
            return

        # 计算余弦相似度
        similarity_matrix = cosine_similarity(self.normalized_matrix)
        similarity_df = pd.DataFrame(similarity_matrix,
                                     index=self.garden_names,
                                     columns=self.garden_names)

        print("\n园林间余弦相似度矩阵计算完成：")
        print(similarity_df.round(3))

        # --- 修改部分开始 ---
        # 使用 seaborn 的函数来设置样式，兼容性更好
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        # --- 修改部分结束 ---

        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_df, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
        plt.title('江南古典园林美学特征相似度热力图', fontsize=18, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def _perform_hierarchical_clustering(self):
        """
        执行层次聚类并使用树状图可视化。
        """
        if self.normalized_matrix is None:
            print("❌ 归一化矩阵为空，无法进行聚类。")
            return

        # 使用 'ward' 方法进行聚类，它倾向于发现大小相近的簇
        linked = linkage(self.normalized_matrix, method='ward')

        plt.figure(figsize=(14, 8))
        dendrogram(linked,
                   orientation='top',
                   labels=self.garden_names,
                   distance_sort='descending',
                   show_leaf_counts=True)
        plt.title('江南古典园林层次聚类树状图', fontsize=18, pad=20)
        plt.ylabel('聚类距离 (Ward)', fontsize=12)
        plt.xlabel('园林名称', fontsize=12)
        plt.xticks(fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        print("\n层次聚类分析完成。请观察树状图，距离越近的园林被合并得越早，代表它们越相似。")
        print("可以根据树状图的结构，将园林划分为不同的流派（例如，从某个高度横切）。")


# -------------------------- 3. 主程序入口 --------------------------

if __name__ == "__main__":
    # 定义10个园林的数据路径和名称
    gardens_to_analyze = [
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

    # 实例化并运行分析器
    analyzer = SimilarityAnalyzer(gardens_to_analyze)
    analyzer.run_full_analysis()

    # =========================================================================
    # PART 2: 泛化性验证 (回答 3.2)
    # =========================================================================
    print("\n\n" + "=" * 30 + " Part 2: 泛化性验证 " + "=" * 30)
    new_garden_path = "cleaned_excels/个园.xlsx"  # 假设这是新园林的路径
    new_garden_name = "个园"

    if analyzer.scaler is not None and analyzer.feature_df is not None:
        print(f"正在对新园林【{new_garden_name}】进行分析...")

        # 1. 提取新园林的特征向量 (同样，这里使用模拟数据)
        np.random.seed(100)  # 新的随机种子
        new_garden_features_dict = OrderedDict([
            ('garden_name', new_garden_name),
            ('plant_ratio', np.random.uniform(0.15, 0.4)),
            ('water_ratio', np.random.uniform(0.05, 0.25)),
            ('building_ratio', np.random.uniform(0.03, 0.1)),
            ('spatial_overlap', np.random.uniform(2.0, 5.0)),
            ('spatial_diversity', np.random.uniform(1.5, 4.0)),
            ('global_openness_mean', np.random.uniform(40, 70)),
            ('global_openness_std', np.random.uniform(10, 30)),
            ('path_openness_mean', np.random.uniform(30, 60)),
            ('path_openness_std', np.random.uniform(15, 35)),
            ('sequence_density', np.random.uniform(10, 25)),
            ('sequence_amplitude', np.random.uniform(20, 50)),
            ('road_length', np.random.uniform(500, 2000)),
            ('waterfront_complexity', np.random.uniform(1.5, 4.5))
        ])

        # 2. 将新园林特征转为numpy数组
        new_garden_vector = np.array([list(new_garden_features_dict.values())[1:]])

        # 3. 使用已有的scaler进行transform (关键步骤!)
        normalized_new_vector = analyzer.scaler.transform(new_garden_vector)

        # 4. 计算与所有原始园林的相似度
        similarities = cosine_similarity(normalized_new_vector, analyzer.normalized_matrix)

        # 5. 结果分析
        similarity_scores = pd.Series(similarities[0], index=analyzer.garden_names)
        print(f"\n【{new_garden_name}】与其它园林的相似度得分：")
        print(similarity_scores.sort_values(ascending=False).round(3))

        most_similar_garden = similarity_scores.idxmax()
        highest_score = similarity_scores.max()

        print(f"\n分析结论：新园林【{new_garden_name}】与【{most_similar_garden}】最为相似 (相似度: {highest_score:.3f})。")
        print("接下来，您可以结合文献资料，论证此结果的合理性。")
    else:
        print("❌ 无法进行泛化性验证，因为初始分析未成功完成。")
