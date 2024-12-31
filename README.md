# 分水岭算法的应用    
大二上 数字图像处理课程实习 基于分水岭算法的地物分割与识别、粘连细胞的分割与计数的应用

## 简介
本项目是武汉大学遥感信息工程学院学生在数字图像处理课程中完成的实习任务。通过实现和应用分水岭算法，我们深入研究了地物分割与识别及粘连细胞分割与计数的应用场景。我们的目标是探索并优化分水岭算法的效果，提高图像分割性能。

## 目录结构
项目文件按照以下结构组织：
.
├── 20241223_dip_teamwork_watershed
│   ├── 20241223_dip_teamwork_watershed.vcxproj
│   ├── 20241223_dip_teamwork_watershed.vcxproj.filters
│   ├── 20241223_dip_teamwork_watershed.vcxproj.user
│   ├── cells.png
│   ├── data                             # 中间数据文件夹
│   │   ├── 3_gray_hist.png
│   │   └── histogram_stats.txt
│   ├── opencv_world450d.lib             # OpenCV 4.5版本 库文件
│   ├── pic                              # 遥感影像文件夹（已进行灰度拉伸）
│   │   ├── stretched_output_mss.tif
│   │   └── stretched_output_pan.tif
│   ├── res                              # 细胞分割与计数的结果图像
│   │   ├── 01_original.png
│   │   ├── 02_grayscale.png
│   │   ├── 03_binary.png
│   │   ├── 04_cleaned_binary.png
│   │   ├── 05_smoothed_binary.png
│   │   ├── 06_inverted_distance_transform.png
│   │   ├── 07_watershed_lines.png
│   │   ├── 08_watershed_result.png
│   │   ├── 09_segmented_binary.png
│   │   └── 10_numbered_cells.png
│   ├── res_g                           # 地物分割与识别的结果图像
│   │   ├── 1_original_scaled.png
│   │   ├── 2_denoised.png
│   │   ├── 3_thresholded.png
│   │   ├── 4_gradient.png
│   │   ├── 5_gradient_cleaned.png
│   │   └── 6_colored.png
│   ├── cell.cpp                        # 细胞分割与计数的程序
│   ├── rs_img.cpp                      # 地物分割与识别的程序
│   └── ...
├── 20241223_dip_teamwork_watershed.sln
├── README.md                           # 当前文件
└── include                             # 包含 OpenCV 4.5版本 额外资源的文件夹


## 运行方式
请分别运行地物分割与识别的程序 `rs_img.cpp` 和细胞分割与计数的程序 `cell.cpp`，并且在运行其中一个程序的时候，另一个程序需要全部注释掉。

## 致谢
感谢指导教师杨代琴老师提供的宝贵指导和支持。同时感谢小组成员杨丹阳、徐佳蕾、王瑞冰和吴安民的紧密配合与共同努力。也感谢同班同学以及其他同学在遇到困难时给予的宝贵启示和鼓励。

## 版权声明与使用授权
为了便于他人对本项目成果的复现与继续研究，本项目已提交至组长杨丹阳的 GitHub 远程仓库。    
本项目《分水岭算法》实习报告及相关代码由武汉大学遥感信息工程学院的学生杨丹阳、徐佳蕾、王瑞冰和吴安民共同创作完成。    
本项目的所有成员授权用户按照以下条款使用本项目的资料：
 - 非商业用途：除非得到版权所有者的书面许可，否则不得将本项目用于任何形式的商业目的。
 - 保留版权说明：使用本项目内容时必须保留原有的版权声明和链接到原始项目的位置。
 - 不可转让性：您不可转让此授权给第三方。
 - 限制复制：您可以复制本项目的部分内容用于个人学习或研究，但不得大量复制或发行。
 - 修改与衍生作品：允许对本项目进行修改以适应您的需求，但必须注明修改部分，并且不得移除原有作者的署名。
 - 责任限制：对于因使用本项目而产生的任何直接或间接损失，版权所有者概不负责。
 - 如果您有意向超出上述授权范围使用本项目，请联系原作者获取进一步的许可。    

**开源协议**：本项目遵循开源精神，源代码托管于组长杨丹阳的 GitHub 远程仓库。我们鼓励学术交流和技术进步，欢迎您访问并参与讨论和贡献。同时，我们也希望使用者尊重我们的劳动成果，遵守相关法律法规，共同维护良好的科研环境。    
