#!/usr/bin/env python3
"""
完整测试流程: 表格分离 + 文字分类
"""
import sys
import os
import shutil
import cv2
from utils.table_detector import TableDetector
from utils.text_classifier import TextClassifier

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python test_classifier.py <图片路径>")
        print("\n示例:")
        print("  python test_classifier.py Pictures/1.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    # 0. 清空 results 目录
    results_dir = "results"
    if os.path.exists(results_dir):
        print(f"清空 {results_dir} 目录...")
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("完整测试流程: 表格分离 + 文字分类")
    print("=" * 70)

    # 1. 表格分离
    print("\n[1/2] 表格分离...")
    table_detector = TableDetector(debug=True)
    table_result = table_detector.separate_and_save(
        image_path=image_path,
        output_dir=results_dir
    )
    print(f"  ✓ 检测到 {table_result['table_count']} 个表格区域")

    # 2. 文字分类
    print("\n[2/2] 文字分类...")

    # 读取表格掩码(避免重复处理表格区域)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    table_mask_path = os.path.join(results_dir, f"{base_name}_table.jpg")
    table_mask = cv2.imread(table_mask_path, cv2.IMREAD_GRAYSCALE)

    if table_mask is not None:
        # 创建表格区域掩码(非零区域为表格)
        _, table_mask = cv2.threshold(table_mask, 10, 255, cv2.THRESH_BINARY)
        print(f"  已加载表格掩码，将排除表格区域")

    # 执行文字分类
    classifier = TextClassifier(debug=True, split_ratio=0.45)
    text_result = classifier.classify_and_separate(
        image_path=image_path,
        output_dir=results_dir,
        table_mask=table_mask
    )

    print("\n" + "=" * 70)
    print("处理完成！")
    print("\n[表格分离结果]")
    print(f"  表格区域: {table_result['table_count']} 个")
    print(f"  表格内容: {results_dir}/{base_name}_table.jpg")
    print(f"  非表格内容: {results_dir}/{base_name}_non_table.jpg")
    print(f"  表格标注: {results_dir}/{base_name}_table_annotated.jpg")

    print("\n[文字分类结果]")
    print(f"  手写体区域: {text_result['handwritten_count']} 个")
    print(f"  印刷体区域: {text_result['printed_count']} 个")
    print(f"  手写体: {text_result['handwritten_path']}")
    print(f"  印刷体: {text_result['printed_path']}")
    print(f"  标注图: {text_result['annotated_path']}")
    print("=" * 70)
