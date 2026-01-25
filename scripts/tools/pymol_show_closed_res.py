from pymol import cmd
import math

# 注册命令，使其可以在 PyMOL 命令行直接调用
@cmd.extend
def select_nearest_residues(target_sel, n_neighbors=15, out_sel="nearest15", protein_obj="all"):
    """
    DESCRIPTION
        根据 CA 原子的距离，选择目标区域周围最近的 N 个氨基酸（不包含目标自身）。

    USAGE
        select_nearest_residues target_sel, [n_neighbors], [out_sel], [protein_obj]

    ARGS
        target_sel  : 目标选区（例如：resi 100）
        n_neighbors : 要选出的最近残基数量（默认 15）
        out_sel     : 输出的选区名称（默认 nearest15）
        protein_obj : 搜索范围，默认是所有蛋白（all）
    """
    
    # 确保 n_neighbors 是整数
    n_neighbors = int(n_neighbors)

    # 1. 获取目标区域的所有 CA 原子坐标
    # 逻辑：目标选区 -> 限制为 CA -> 获取模型
    target_ca_sel = f"({target_sel}) and name CA"
    target_model = cmd.get_model(target_ca_sel)

    if len(target_model.atom) == 0:
        print(f"[Error] Target selection '{target_sel}' contains no CA atoms.")
        return

    # 提取目标坐标列表
    target_coords = [a.coord for a in target_model.atom]

    # 2. 获取候选原子（蛋白中所有 CA，但排除目标区域自身）
    # 关键优化：使用 PyMOL 选择逻辑 'and not' 直接排除自身，避免在循环中检查
    # byres 确保排除的是整个残基，不仅仅是重叠的原子
    candidate_sel = f"({protein_obj}) and polymer.protein and name CA and not (byres ({target_sel}))"
    candidate_model = cmd.get_model(candidate_sel)
    
    if len(candidate_model.atom) == 0:
        print("[Warning] No candidate residues found around target.")
        return

    neighbors = []

    # 3. 计算距离
    # 遍历所有候选 CA 原子
    for atom in candidate_model.atom:
        # 计算该原子到目标区域所有 CA 的距离，取最小值
        # 兼容性写法，防止旧版 Python 没有 math.dist
        if hasattr(math, 'dist'):
            min_dist = min(math.dist(atom.coord, t_coord) for t_coord in target_coords)
        else:
            min_dist = min(
                math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(atom.coord, t_coord)))
                for t_coord in target_coords
            )
        
        # 存储元组: (原子ID, 最小距离, 残基信息用于打印)
        neighbors.append((atom.id, min_dist, f"{atom.chain}/{atom.resi}{atom.resn}"))

    # 4. 排序并截取前 N 个
    # 按距离从小到大排序
    neighbors.sort(key=lambda x: x[1])
    top_n = neighbors[:n_neighbors]

    if not top_n:
        print("[Info] No residues found.")
        return

    # 5. 构建 Selection
    # 使用原子 ID (id) 构建选择集最快且最准确，格式为: id 1+2+3...
    id_list = [str(item[0]) for item in top_n]
    id_sel_str = "+".join(id_list)
    
    # 先选中这些 CA 原子
    cmd.select(out_sel, f"id {id_sel_str}")
    # 扩展选择集到完整的残基 (byres)
    cmd.select(out_sel, f"byres {out_sel}")

    # 6. 打印结果信息
    print(f"=================================================")
    print(f"[SUCCESS] Selected {len(top_n)} nearest residues into '{out_sel}'.")
    print(f"Target: {target_sel}")
    print(f"Closest residues:")
    for i, (atom_id, dist, res_info) in enumerate(top_n):
        print(f"  {i+1}. {res_info} \t(Dist: {dist:.2f} A)")
    print(f"=================================================")

# === 示例调用（如果是脚本模式直接运行以下行） ===
# 如果你在 PyMOL 界面中加载此脚本，不需要取消注释下面的行，
# 直接在 PyMOL 命令行输入: select_nearest_residues target_region
'''
select_nearest_residues(
    target_sel="target_region",
    n_neighbors=15,
    out_sel="nearest15"
)
'''