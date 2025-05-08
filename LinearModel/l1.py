import sys # 导入 sys 模块来处理命令行参数

# 定义 main 函数，接收一个参数 a
def main(a):
    """
    主函数，打印接收到的参数。

    Args:
        a: 从命令行接收到的参数。
    """
    print("i get a:",a)

if __name__ == "__main__":
    # 检查是否有命令行参数传入
    if len(sys.argv) > 1:
        print("sys.argv:",sys.argv)
        # 将第一个命令行参数传递给 main 函数
        main(sys.argv[1])
    else:
        # 如果没有参数，可以打印提示或使用默认值
        print("请提供一个参数！")
        # 或者使用默认值调用 main(default_value)