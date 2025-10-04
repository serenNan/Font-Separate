#!/usr/bin/env python3
"""
颜色分类演示脚本
基于自适应颜色聚类对文档中不同颜色的文字进行分类
"""
import sys
import os
import argparse
import json
from utils.color_classifier import ColorClassifier


def main():
    parser = argparse.ArgumentParser(
        description='基于颜色的文字自适应分类工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 自动检测颜色类别数（推荐）
  python color_classify_demo.py Pictures/原始.jpg

  # 指定3个颜色类别
  python color_classify_demo.py Pictures/原始.jpg --n-clusters 3

  # 使用 HSV 颜色空间
  python color_classify_demo.py Pictures/原始.jpg --color-space hsv

  # 调试模式（输出详细信息）
  python color_classify_demo.py Pictures/原始.jpg --debug

  # 自定义输出目录
  python color_classify_demo.py Pictures/原始.jpg --output results_custom
        '''
    )

    parser.add_argument(
        'image',
        help='输入图像路径'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=None,
        help='指定颜色聚类数（默认: 自动检测最佳值）'
    )
    parser.add_argument(
        '--color-space',
        choices=['rgb', 'hsv', 'lab'],
        default='lab',
        help='颜色空间 (默认: lab，推荐用于聚类)'
    )
    parser.add_argument(
        '--auto-k-range',
        nargs=2,
        type=int,
        default=[2, 6],
        metavar=('MIN', 'MAX'),
        help='自动检测时的聚类数范围 (默认: 2 6)'
    )
    parser.add_argument(
        '--min-saturation',
        type=int,
        default=10,
        help='最小饱和度阈值，过滤背景噪声 (默认: 10)'
    )
    parser.add_argument(
        '--output',
        default='results',
        help='输出目录 (默认: results)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式，输出详细信息'
    )

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.image):
        print(f"✗ 错误: 文件不存在 - {args.image}")
        sys.exit(1)

    print("=" * 70)
    print("颜色自适应分类 - 文档图像智能分离")
    print("=" * 70)
    print(f"输入图像: {args.image}")
    print(f"输出目录: {args.output}")
    print(f"颜色空间: {args.color_space.upper()}")

    if args.n_clusters is None:
        print(f"聚类模式: 自动检测 (范围 {args.auto_k_range[0]}-{args.auto_k_range[1]})")
    else:
        print(f"聚类模式: 手动指定 K={args.n_clusters}")

    print("=" * 70)

    # 初始化分类器
    classifier = ColorClassifier(
        debug=args.debug,
        n_clusters=args.n_clusters,
        auto_k_range=tuple(args.auto_k_range),
        color_space=args.color_space,
        min_saturation=args.min_saturation
    )

    try:
        # 执行分类
        result = classifier.classify_by_color(args.image, args.output)

        # 输出结果
        print("\n" + "=" * 70)
        print("处理完成！")
        print("=" * 70)
        print(f"检测到 {result['n_clusters']} 种颜色类别：")
        print()

        for cluster_id, info in result['clusters'].items():
            print(f"  类别 {cluster_id} ({info['description']}):")
            print(f"    文字区域数: {info['count']}")
            print(f"    代表颜色 RGB: {info['color_rgb']}")
            print(f"    代表颜色 HSV: {info['color_hsv']}")
            print()

        print("输出文件:")
        print(f"  标注图: {result['annotated_path']}")
        print(f"  颜色色板: {result['palette_path']}")

        for i, path in enumerate(result['cluster_paths']):
            print(f"  类别 {i}: {path}")

        # 保存统计信息到 JSON
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        stats_path = os.path.join(args.output, f"{base_name}_color_stats.json")

        stats = {
            'n_clusters': result['n_clusters'],
            'clusters': result['clusters'],
            'color_space': args.color_space,
            'input_image': args.image
        }

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"  统计信息: {stats_path}")
        print("=" * 70)
        print("✓ 全部完成")

    except Exception as e:
        print(f"\n✗ 处理失败: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
