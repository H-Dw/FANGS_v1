import os
import glob
import argparse
import sys
from typing import Optional

def extract_sequence_from_pdb(file_path: str) -> Optional[str]:
    """
    提取PDB文件的序列，处理异常并返回标准化结果。
    
    Args:
        file_path: PDB文件路径
        
    Returns:
        格式化为FASTA格式的字符串，失败时返回None
    """
    try:
        from readPDBSeq import get_sequence_from_pdb
        sequence = get_sequence_from_pdb(file_path)
        if not sequence:
            return None
            
        header = os.path.splitext(os.path.basename(file_path))[0]
        return f">{header}\n{sequence}"
        
    except ImportError:
        print("错误：找不到readPDBSeq模块，请确保已正确安装", file=sys.stderr)
        return None
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}", file=sys.stderr)
        return None

def main(db_path: str, output_file: str) -> None:
    """
    主函数：处理PDB文件并生成FASTA序列文件
    
    Args:
        db_path: 输入路径（支持通配符或目录）
        output_file: 输出FASTA文件路径
    """
    # 验证输入路径
    if not os.path.exists(db_path):
        print(f"输入路径不存在: {db_path}", file=sys.stderr)
        return

    # 构建文件列表
    if os.path.isdir(db_path):
        pattern = os.path.join(db_path, "*.pdb")
    else:
        pattern = db_path
        
    pdb_files = glob.glob(pattern)
    if not pdb_files:
        print(f"未找到匹配的PDB文件: {pattern}", file=sys.stderr)
        return

    # 处理文件并写入结果
    try:
        with open(output_file, 'w') as f_out:
            for file in pdb_files:
                fasta_entry = extract_sequence_from_pdb(file)
                if fasta_entry:
                    f_out.write(fasta_entry + '\n')
        print(f"成功处理 {len(pdb_files)} 个文件，输出保存至 {output_file}")
                    
    except IOError as e:
        print(f"写入输出文件失败: {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从PDB文件提取蛋白质序列并生成FASTA格式文件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("db_path", help="PDB文件路径（支持通配符）或包含PDB的目录")
    parser.add_argument("output_file", help="输出FASTA文件的路径")
    
    args = parser.parse_args()
    
    # 添加readPDBSeq模块路径到系统路径（如需要）
    # sys.path.append('/path/to/readPDBSeq')
    
    main(args.db_path, args.output_file)