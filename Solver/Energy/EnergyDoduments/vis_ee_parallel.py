import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class EdgeEdgeDistance:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
    
    def point_edge_distance_coeff(self, p, e0, e1):
        """计算点到边的最接近点参数"""
        r = e1 - e0
        d = np.dot(r, r)  # squared_norm
        
        if d <= self.epsilon:
            return np.array([1.0, 0.0])  # 如果边退化为点，选择第一个端点
        
        t = np.dot(r, p - e0) / d
        t_clipped = np.clip(t, 0.0, 1.0)
        return np.array([1.0 - t_clipped, t_clipped])
    
    def point_edge_distance_squared(self, p, e0, e1):
        """计算点到边的平方距离"""
        coeff = self.point_edge_distance_coeff(p, e0, e1)
        closest_point = coeff[0] * e0 + coeff[1] * e1
        return np.sum((p - closest_point) ** 2), coeff
    
    def edge_edge_distance_coeff(self, ea0, ea1, eb0, eb1):
        """计算边-边距离的系数（无约束优化）"""
        r0 = ea1 - ea0
        r1 = eb1 - eb0
        
        # 构建线性系统 A * x = b
        A = np.column_stack([r0, -r1])  # 3x2矩阵
        A_T = A.T  # 2x3转置
        b = eb0 - ea0
        
        # 解正规方程 (A^T A) x = A^T b
        ATA = A_T @ A  # 2x2矩阵
        ATb = A_T @ b  # 2维向量
        
        try:
            # 检查矩阵是否可逆
            if np.linalg.det(ATA) > self.epsilon:
                x = np.linalg.solve(ATA, ATb)
                bary = np.array([1.0 - x[0], x[0], 1.0 - x[1], x[1]])
                return bary
        except np.linalg.LinAlgError:
            pass
        
        # 如果无法求解，返回中点
        return np.array([0.5, 0.5, 0.5, 0.5])
    
    def edge_edge_distance_coeff_unclassified(self, ea0, ea1, eb0, eb1):
        """处理边-边距离的完整分类方法"""
        # 首先尝试无约束解
        c = self.edge_edge_distance_coeff(ea0, ea1, eb0, eb1)
        
        # 检查是否在有效范围内
        if np.all(c >= 0.0) and np.all(c <= 1.0):
            return c
        
        # 如果无约束解无效，检查所有边界情况
        # 情况1: 边A的端点0到边B
        c1 = self.point_edge_distance_coeff(ea0, eb0, eb1)
        if c1[1] < 0.0:
            c1 = np.array([0.0, 1.0])  # 使用eb0
        elif c1[1] > 1.0:
            c1 = np.array([1.0, 0.0])  # 使用eb1
        
        # 情况2: 边A的端点1到边B
        c2 = self.point_edge_distance_coeff(ea1, eb0, eb1)
        if c2[1] < 0.0:
            c2 = np.array([0.0, 1.0])
        elif c2[1] > 1.0:
            c2 = np.array([1.0, 0.0])
        
        # 情况3: 边B的端点0到边A
        c3 = self.point_edge_distance_coeff(eb0, ea0, ea1)
        if c3[1] < 0.0:
            c3 = np.array([0.0, 1.0])
        elif c3[1] > 1.0:
            c3 = np.array([1.0, 0.0])
        
        # 情况4: 边B的端点1到边A
        c4 = self.point_edge_distance_coeff(eb1, ea0, ea1)
        if c4[1] < 0.0:
            c4 = np.array([0.0, 1.0])
        elif c4[1] > 1.0:
            c4 = np.array([1.0, 0.0])
        
        # 定义四种边界情况
        types = [
            np.array([1.0, 0.0, c1[0], c1[1]]),  # 使用ea0
            np.array([0.0, 1.0, c2[0], c2[1]]),  # 使用ea1
            np.array([c3[0], c3[1], 1.0, 0.0]),  # 使用eb0
            np.array([c4[0], c4[1], 0.0, 1.0])   # 使用eb1
        ]
        
        # 找到最小距离的情况
        min_distance = float('inf')
        result = types[0]
        
        for i, coeff in enumerate(types):
            x0 = coeff[0] * ea0 + coeff[1] * ea1
            x1 = coeff[2] * eb0 + coeff[3] * eb1
            distance = np.sum((x1 - x0) ** 2)
            
            if distance < min_distance:
                min_distance = distance
                result = coeff
        
        return result
    
    def edge_edge_distance_squared_unclassified(self, ea0, ea1, eb0, eb1):
        """计算边-边平方距离（完整分类版本）"""
        c = self.edge_edge_distance_coeff_unclassified(ea0, ea1, eb0, eb1)
        x0 = c[0] * ea0 + c[1] * ea1
        x1 = c[2] * eb0 + c[3] * eb1
        return np.sum((x1 - x0) ** 2), c
    
    def analyze_edge_edge_relationship(self, ea0, ea1, eb0, eb1):
        """分析边-边关系，特别处理几乎平行的情况"""
        r0 = ea1 - ea0
        r1 = eb1 - eb0
        
        # 计算方向向量
        dir0 = r0 / (np.linalg.norm(r0) + self.epsilon)
        dir1 = r1 / (np.linalg.norm(r1) + self.epsilon)
        
        # 检查平行度
        dot_product = np.abs(np.dot(dir0, dir1))
        is_parallel = dot_product > 0.999  # 几乎平行
        
        # 计算距离
        distance_squared, coeff = self.edge_edge_distance_squared_unclassified(ea0, ea1, eb0, eb1)
        distance = np.sqrt(distance_squared)
        
        # 确定最近点类型
        type_desc = "internal"
        if np.allclose(coeff[:2], [1, 0]) or np.allclose(coeff[:2], [0, 1]):
            type_desc = "edge_A_endpoint"
        elif np.allclose(coeff[2:], [1, 0]) or np.allclose(coeff[2:], [0, 1]):
            type_desc = "edge_B_endpoint"
        
        return {
            'distance': distance,
            'distance_squared': distance_squared,
            'coefficients': coeff,
            'is_parallel': is_parallel,
            'parallel_dot': dot_product,
            'type': type_desc,
            'closest_point_A': coeff[0] * ea0 + coeff[1] * ea1,
            'closest_point_B': coeff[2] * eb0 + coeff[3] * eb1
        }

# 使用示例和分析
def comprehensive_analysis():
    # 输入数据
    t0_ea0 = np.array([-0.00052356004, 0.5066976, 0.00046172945])
    t0_ea1 = np.array([0.1994806, 0.50669956, 0.20045757])
    t0_eb0 = np.array([1.4930345e-05, 0.5027302, -6.327331e-05])
    t0_eb1 = np.array([0.50006753, 0.5027298, 0.4998841])

    t1_ea0 = np.array([-0.0005480349, 0.5066976, 0.00047544707])
    t1_ea1 = np.array([0.19946572, 0.5066996, 0.2004617])
    t1_eb0 = np.array([1.3507714e-05, 0.5027302, -6.3285814e-05])
    t1_eb1 = np.array([0.50007135, 0.5027298, 0.49987888])
    
    distance_calculator = EdgeEdgeDistance()
    
    print("=== 边-边距离详细分析 ===")
    
    # 分析t0时刻
    print("\n--- t0时刻分析 ---")
    result_t0 = distance_calculator.analyze_edge_edge_relationship(t0_ea0, t0_ea1, t0_eb0, t0_eb1)
    print(f"距离: {result_t0['distance']:.8f}")
    print(f"是否几乎平行: {result_t0['is_parallel']}")
    print(f"平行度: {result_t0['parallel_dot']:.8f}")
    print(f"最近点类型: {result_t0['type']}")
    print(f"系数: {result_t0['coefficients']}")
    print(f"边A最近点: {result_t0['closest_point_A']}")
    print(f"边B最近点: {result_t0['closest_point_B']}")
    
    # 分析t1时刻
    print("\n--- t1时刻分析 ---")
    result_t1 = distance_calculator.analyze_edge_edge_relationship(t1_ea0, t1_ea1, t1_eb0, t1_eb1)
    print(f"距离: {result_t1['distance']:.8f}")
    print(f"是否几乎平行: {result_t1['is_parallel']}")
    print(f"平行度: {result_t1['parallel_dot']:.8f}")
    print(f"最近点类型: {result_t1['type']}")
    print(f"系数: {result_t1['coefficients']}")
    print(f"边A最近点: {result_t1['closest_point_A']}")
    print(f"边B最近点: {result_t1['closest_point_B']}")
    
    # 距离变化分析
    print(f"\n--- 距离变化分析 ---")
    print(f"距离变化: {result_t1['distance'] - result_t0['distance']:.8f}")
    print(f"平行度变化: {result_t1['parallel_dot'] - result_t0['parallel_dot']:.8f}")
    
    return result_t0, result_t1

def plot_detailed_analysis(result_t0, result_t1):
    """绘制详细的分析结果"""
    # 输入数据
    t0_ea0 = np.array([-0.00052356004, 0.5066976, 0.00046172945])
    t0_ea1 = np.array([0.1994806, 0.50669956, 0.20045757])
    t0_eb0 = np.array([1.4930345e-05, 0.5027302, -6.327331e-05])
    t0_eb1 = np.array([0.50006753, 0.5027298, 0.4998841])

    t1_ea0 = np.array([-0.0005480349, 0.5066976, 0.00047544707])
    t1_ea1 = np.array([0.19946572, 0.5066996, 0.2004617])
    t1_eb0 = np.array([1.3507714e-05, 0.5027302, -6.3285814e-05])
    t1_eb1 = np.array([0.50007135, 0.5027298, 0.49987888])
    
    fig = plt.figure(figsize=(16, 12))
    
    # 3D视图，包含最近点
    ax1 = fig.add_subplot(221, projection='3d')
    
    # 绘制边
    ax1.plot([t0_ea0[0], t0_ea1[0]], [t0_ea0[1], t0_ea1[1]], [t0_ea0[2], t0_ea1[2]], 
             'b-', linewidth=3, label='Edge A (t0)')
    ax1.plot([t0_eb0[0], t0_eb1[0]], [t0_eb0[1], t0_eb1[1]], [t0_eb0[2], t0_eb1[2]], 
             'r-', linewidth=3, label='Edge B (t0)')
    
    # 绘制最近点连线
    cpA_t0 = result_t0['closest_point_A']
    cpB_t0 = result_t0['closest_point_B']
    ax1.plot([cpA_t0[0], cpB_t0[0]], [cpA_t0[1], cpB_t0[1]], [cpA_t0[2], cpB_t0[2]], 
             'g--', linewidth=2, label='Closest Points (t0)')
    
    # 标记最近点
    ax1.scatter(*cpA_t0, color='blue', s=100, marker='o')
    ax1.scatter(*cpB_t0, color='red', s=100, marker='s')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View with Closest Points (t0)')
    ax1.legend()
    
    # t1时刻的类似图
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot([t1_ea0[0], t1_ea1[0]], [t1_ea0[1], t1_ea1[1]], [t1_ea0[2], t1_ea1[2]], 
             'c-', linewidth=3, label='Edge A (t1)')
    ax2.plot([t1_eb0[0], t1_eb1[0]], [t1_eb0[1], t1_eb1[1]], [t1_eb0[2], t1_eb1[2]], 
             'm-', linewidth=3, label='Edge B (t1)')
    
    cpA_t1 = result_t1['closest_point_A']
    cpB_t1 = result_t1['closest_point_B']
    ax2.plot([cpA_t1[0], cpB_t1[0]], [cpA_t1[1], cpB_t1[1]], [cpA_t1[2], cpB_t1[2]], 
             'y--', linewidth=2, label='Closest Points (t1)')
    
    ax2.scatter(*cpA_t1, color='cyan', s=100, marker='o')
    ax2.scatter(*cpB_t1, color='magenta', s=100, marker='s')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D View with Closest Points (t1)')
    ax2.legend()
    
    # 距离变化图
    ax3 = fig.add_subplot(223)
    times = ['t0', 't1']
    distances = [result_t0['distance'], result_t1['distance']]
    parallel_dots = [result_t0['parallel_dot'], result_t1['parallel_dot']]
    
    ax3.plot(times, distances, 'bo-', linewidth=2, markersize=8, label='Distance')
    ax3.set_ylabel('Distance')
    ax3.set_title('Distance Change')
    ax3.grid(True)
    ax3.legend()
    
    ax4 = ax3.twinx()
    ax4.plot(times, parallel_dots, 'ro-', linewidth=2, markersize=8, label='Parallel Dot')
    ax4.set_ylabel('Parallel Dot Product')
    ax4.legend(loc='upper right')
    
    # 系数变化图
    ax5 = fig.add_subplot(224)
    coeff_labels = ['A0', 'A1', 'B0', 'B1']
    coeff_t0 = result_t0['coefficients']
    coeff_t1 = result_t1['coefficients']
    
    x = np.arange(len(coeff_labels))
    width = 0.35
    
    ax5.bar(x - width/2, coeff_t0, width, label='t0', alpha=0.7)
    ax5.bar(x + width/2, coeff_t1, width, label='t1', alpha=0.7)
    
    ax5.set_xlabel('Coefficient Type')
    ax5.set_ylabel('Value')
    ax5.set_title('Barycentric Coefficients')
    ax5.set_xticks(x)
    ax5.set_xticklabels(coeff_labels)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 执行分析
if __name__ == "__main__":
    result_t0, result_t1 = comprehensive_analysis()
    plot_detailed_analysis(result_t0, result_t1)