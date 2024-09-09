import os
import pandas as pd
from numpy import vstack

def import_from_files():
    """
        读取 .csv 文件并将数据存储到数组中
        格式: |LOS|NLOS|data...|
    """
    rootdir = 'C:/Users/Doraemon/Desktop/LOS-NLOS-Classification-CNN-master/dataset/'  # 文件路径

    output_arr = []
    first = 1
    for dirpath, dirnames, filenames in os.walk(rootdir):
        for file in filenames:
            filename = os.path.join(dirpath, file)
            print(f"当前处理的文件: {filename}")

            # 如果文件不是 .csv 文件，则跳过
            if not filename.endswith('.csv'):
                print(f"跳过非 .csv 文件: {filename}")
                continue

            try:
                # 读取 csv 文件
                df = pd.read_csv(filename, sep=',', header=0)
                input_data = df.values  # 使用 .values
            except Exception as e:
                print(f"读取文件时出错: {filename}, 错误信息: {e}")
                continue

            # 将数据附加到数组中
            if first > 0:
                first = 0
                output_arr = input_data
            else:
                output_arr = vstack((output_arr, input_data))

    return output_arr

if __name__ == '__main__':
    print("正在将数据集导入到 numpy 数组")
    print("-------------------------------")
    data = import_from_files()
    print("-------------------------------")
    print("数据集样本数: %d" % len(data))
    print("单个样本长度: %d" % len(data[0]))
    print("-------------------------------")
    print("数据集内容:")
    print(data)
