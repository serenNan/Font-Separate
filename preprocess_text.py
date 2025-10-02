"""
文字预处理测试脚本
去除表格线、印章、污渍,只保留纯文字
"""
import sys
from utils.text_extractor import TextExtractor


def main():
    if len(sys.argv) < 2:
        print("用法: python preprocess_text.py <图像路径>")
        print("\n功能:")
        print("  - 去除表格线")
        print("  - 去除印章(红色区域)")
        print("  - 去除噪点和污渍")
        print("  - 增强模糊文字")
        print("\n输出:")
        print("  results/原文件名_text_only.jpg  - 纯文字图像")
        print("  results/原文件名_text_overlay.jpg - 文字标注图")
        sys.exit(1)

    image_path = sys.argv[1]

    print("=" * 70)
    print("文字提取预处理工具")
    print("=" * 70)
    print(f"输入图像: {image_path}")
    print()

    # 创建提取器(开启debug)
    extractor = TextExtractor(debug=True)

    # 执行提取
    text_only, stats = extractor.extract_text_only(
        image_path,
        output_dir='results'
    )

    print("\n" + "=" * 70)
    print("✓ 预处理完成!")
    print("=" * 70)
    print("\n查看结果:")
    print("  1. results/*_text_only.jpg    - 纯文字(黑白)")
    print("  2. results/*_text_overlay.jpg - 文字标注(绿色)")
    print("  3. results/*_1_enhanced.jpg   - 增强后的灰度图")
    print("  4. results/*_2_binary.jpg     - 二值化结果")
    print("  5. results/*_3_no_lines.jpg   - 去除表格线后")
    print("  6. results/*_4_no_stamps.jpg  - 去除印章后")
    print("  7. results/*_5_no_noise.jpg   - 去除噪点后")
    print()


if __name__ == '__main__':
    main()
