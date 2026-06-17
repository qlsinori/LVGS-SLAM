import numpy as np


def parse_opencv_matrix(block):
    """手动解析opencv-matrix数据块"""
    matrix_data = {}
    lines = block.split('\n')

    for line in lines:
        line = line.strip()
        if not line: continue

        # 解析键值对
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # 处理数据类型
            if key == 'rows' or key == 'cols':
                matrix_data[key] = int(value)
            elif key == 'dt':  # 数据类型标记
                matrix_data['dtype'] = np.float64 if value == 'd' else np.float32
            elif key == 'data':
                # 处理多行数组数据
                array_str = value
                # 合并后续的数据行
                for next_line in lines[lines.index(line) + 1:]:
                    if next_line.strip().startswith('[') or not next_line.strip():
                        break
                    array_str += next_line.strip()

                # 清理并转换数据
                array_str = array_str.replace('[', '').replace(']', '')
                array_str = array_str.replace(',', ' ').strip()
                # 处理科学计数法中的空格问题
                array_str = array_str.replace('e -', 'e-').replace('e +', 'e+')

                # 转换为浮点数列表
                data_list = []
                for num_str in array_str.split():
                    if num_str:  # 跳过空字符串
                        try:
                            data_list.append(float(num_str))
                        except ValueError:
                            # 处理特殊格式如 "8.9253240841861001e-02"
                            if 'e' in num_str:
                                base, exp = num_str.split('e')
                                data_list.append(float(base) * 10 ** float(exp))
                matrix_data['data'] = data_list

    # 验证并创建矩阵
    if 'rows' in matrix_data and 'cols' in matrix_data and 'data' in matrix_data:
        expected_size = matrix_data['rows'] * matrix_data['cols']
        if len(matrix_data['data']) != expected_size:
            raise ValueError(f"数据大小不匹配: 期望 {expected_size} 个元素, 实际 {len(matrix_data['data'])}")

        return np.array(matrix_data['data'], dtype=matrix_data.get('dtype', np.float64)).reshape(matrix_data['rows'], matrix_data['cols'])
    return None


def load_yaml_matrix(file_path):
    """手动加载YAML文件并解析opencv-matrix结构"""
    with open(file_path, 'r') as f:
        content = f.read()

    # 分割YAML文档为独立块
    blocks = []
    current_block = []
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped.startswith('!!opencv-matrix'):
            if current_block:
                blocks.append('\n'.join(current_block))
            current_block = [line]
        elif current_block:
            current_block.append(line)
    if current_block:
        blocks.append('\n'.join(current_block))

    # 解析所有矩阵块
    results = {}
    for block in blocks:
        if '!!opencv-matrix' in block:
            # 提取矩阵名称
            name_line = block.split('\n')[0].strip()
            if ':' in name_line:
                name = name_line.split(':', 1)[0].strip()
                matrix = parse_opencv_matrix(block)
                if matrix is not None:
                    results[name] = matrix

    return results


if __name__ == "__main__":
    file_path = "/qls/code/neurad-studio/nerfstudio/data/dataparsers/calib/extrinsics/calib_chain.yaml"

    try:
        matrices = load_yaml_matrix(file_path)
        print("成功加载矩阵:")
        for name, matrix in matrices.items():
            print(f"\n{name}:")
            print(matrix)

    except Exception as e:
        print(f"加载失败: {str(e)}")
        # 显示错误位置
        if hasattr(e, 'lineno'):
            print(f"错误发生在第 {e.lineno} 行")
