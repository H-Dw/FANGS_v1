import os
import shutil
import sys

def copy_files(base_dir, output_dir):
    """
    从 base_dir 中提取所有一级子文件夹中的指定文件，
    并分别保存到 output_dir 下的 pdb 和 json 子文件夹中。
    """

    # 检查基础目录是否存在
    if not os.path.exists(base_dir):
        raise RuntimeError(f"[错误] 基础目录不存在: {base_dir}")
    if not os.path.isdir(base_dir):
        raise RuntimeError(f"[错误] 基础路径不是一个目录: {base_dir}")

    # 创建输出目录结构
    pdb_dir = os.path.join(output_dir, "pdb")
    json_dir = os.path.join(output_dir, "json")
    try:
        os.makedirs(pdb_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"[错误] 无法创建输出目录: {e}")

    valid = False  # 是否至少复制了一个文件

    # 遍历一级子文件夹
    for entry in os.scandir(base_dir):
        if entry.is_dir():
            subfolder_name = entry.name
            subfolder_path = entry.path

            # 构建源文件路径
            pdb_src = os.path.join(subfolder_path, f"{subfolder_name}_model_0.pdb")
            json_src = os.path.join(subfolder_path, f"confidence_{subfolder_name}_model_0.json")

            # 构建目标文件路径
            # pdb_dst = os.path.join(pdb_dir, f"{subfolder_name}_model_0.pdb")
            # json_dst = os.path.join(json_dir, f"confidence_{subfolder_name}_model_0.json")
            pdb_dst = os.path.join(pdb_dir, f"{subfolder_name}.pdb")
            json_dst = os.path.join(json_dir, f"confidence_{subfolder_name}.json")

            # 复制 PDB 文件
            if os.path.exists(pdb_src):
                try:
                    shutil.copy2(pdb_src, pdb_dst)
                    valid = True
                except Exception as e:
                    print(f"[错误] 复制 PDB 文件失败: {pdb_src} -> {pdb_dst}, 原因: {e}")
            else:
                print(f"[警告] PDB 文件不存在，跳过: {pdb_src}")

            # 复制 JSON 文件
            if os.path.exists(json_src):
                try:
                    shutil.copy2(json_src, json_dst)
                    valid = True
                except Exception as e:
                    print(f"[错误] 复制 JSON 文件失败: {json_src} -> {json_dst}, 原因: {e}")
            else:
                print(f"[警告] JSON 文件不存在，跳过: {json_src}")

    # 检查是否复制了至少一个文件
    if not valid:
        raise RuntimeError("[错误] 没有复制任何有效文件，退出。")

    print("[完成] 所有符合条件的文件已成功复制。")

# 主程序入口
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python extract_boltz_predictions.py <base_directory> <output_directory>")
        sys.exit(1)

    base_dir = sys.argv[1]
    output_dir = sys.argv[2]

    try:
        copy_files(base_dir, output_dir)
    except RuntimeError as e:
        print(e)
        sys.exit(1)