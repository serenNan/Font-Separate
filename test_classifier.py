"""
测试手写体分类器
"""
import sys
from utils.text_classifier import TextClassifier

def main():
    if len(sys.argv) < 2:
        print("用法: python test_classifier.py <图像路径>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("=" * 60)
    print("手写体/印刷体分类器测试")
    print("=" * 60)

    # 创建分类器(开启debug模式)
    classifier = TextClassifier(debug=True)

    # 执行分类
    result = classifier.classify_and_separate(
        image_path,
        output_dir='results',
        table_mask=None
    )

    print("\n" + "=" * 60)
    print("分类结果:")
    print(f"  手写体区域: {result['handwritten_count']}")
    print(f"  印刷体区域: {result['printed_count']}")
    print(f"  总区域数: {len(result['regions'])}")
    print("\n输出文件:")
    print(f"  手写体: {result['handwritten_path']}")
    print(f"  印刷体: {result['printed_path']}")
    print(f"  标注图: {result['annotated_path']}")
    print("=" * 60)

if __name__ == '__main__':
    main()
