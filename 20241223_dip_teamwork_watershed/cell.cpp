#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <queue>
#include <filesystem>
#include <vector>
#include <stack>

#define PI acos(-1)

using namespace cv;
using namespace std;

//--------------------------------二值化------------------------------------

//转换为灰度图像
void cvtColorToGray(const cv::Mat& image, cv::Mat& gray) {
    // 检查输入图像是否为三通道彩色图像
    if (image.channels() != 3) {
        std::cerr << "Input image is not a 3-channel color image!" << std::endl;
        return;
    }

    // 创建一个单通道的灰度图像，大小与输入图像相同
    gray.create(image.rows, image.cols, CV_8UC1);

    // 遍历每个像素，按照BGR转灰度公式计算灰度值
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            // 获取当前像素的BGR值
            cv::Vec3b bgr = image.at<cv::Vec3b>(i, j);

            // 使用BGR到灰度的转换公式
            unsigned char grayValue = static_cast<unsigned char>(
                0.299 * bgr[2] + 0.587 * bgr[1] + 0.114 * bgr[0]
                );

            // 将计算出的灰度值赋值到输出图像的相应位置
            gray.at<uchar>(i, j) = grayValue;
        }
    }
}

// 计算 Otsu 阈值
double otsuThreshold(const Mat& gray) {
    // 计算图像直方图
    vector<int> hist(256, 0);
    for (int y = 0; y < gray.rows; ++y) {
        for (int x = 0; x < gray.cols; ++x) {
            hist[gray.at<uchar>(y, x)]++;
        }
    }

    // 总像素数
    int total = gray.rows * gray.cols;

    // 计算 Otsu 阈值
    double sum = 0;
    for (int i = 0; i < 256; ++i) {
        sum += i * hist[i];
    }

    double sumB = 0, wB = 0, wF = 0;
    double varMax = 0, threshold = 0;

    for (int t = 0; t < 256; ++t) {
        wB += hist[t];  // 背景的像素权重
        if (wB == 0) continue;
        wF = total - wB;  // 前景的像素权重
        if (wF == 0) break;

        sumB += t * hist[t];
        double mB = sumB / wB;  // 背景的平均灰度
        double mF = (sum - sumB) / wF;  // 前景的平均灰度

        double varBetween = wB * wF * pow(mB - mF, 2);  // 类间方差
        if (varBetween > varMax) {
            varMax = varBetween;
            threshold = t;
        }
    }

    return threshold;
}

// 手动实现二值化（反转并使用 Otsu 阈值）
void thresholdBinaryInvOtsu(const Mat& gray, Mat& binary) {
    // 计算 Otsu 阈值
    double otsuThresh = otsuThreshold(gray);

    // 创建二值图像并应用反转阈值操作
    binary.create(gray.size(), CV_8UC1);
    for (int y = 0; y < gray.rows; ++y) {
        for (int x = 0; x < gray.cols; ++x) {
            // 使用 Otsu 阈值进行反转二值化
            binary.at<uchar>(y, x) = (gray.at<uchar>(y, x) > otsuThresh) ? 0 : 255;
        }
    }
}

//-----------------------------------去除小面积区域----------------------------------
// 用来检查图像内的连通区域
void floodFill(const Mat& binary, Mat& visited, int x, int y, vector<Point>& contour) {
    int rows = binary.rows, cols = binary.cols;
    queue<Point> q;
    q.push(Point(x, y));
    visited.at<uchar>(y, x) = 1;

    while (!q.empty()) {
        Point p = q.front();
        q.pop();
        contour.push_back(p);

        // 访问四个邻域
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                if (dx == 0 && dy == 0) continue;

                int nx = p.x + dx, ny = p.y + dy;

                // 检查边界条件
                if (nx >= 0 && nx < cols && ny >= 0 && ny < rows &&
                    binary.at<uchar>(ny, nx) == 255 && visited.at<uchar>(ny, nx) == 0) {
                    visited.at<uchar>(ny, nx) = 1;
                    q.push(Point(nx, ny));
                }
            }
        }
    }
}

// 计算轮廓的面积（简单地计算轮廓内的像素点数量）
int contourArea(const vector<Point>& contour) 
{
    int n = contour.size();
    if (n < 3) {
        // 不是一个有效的多边形，无法计算面积
        return 0;
    }

    double area = 0.0;
    for (int i = 0; i < n; i++) {
        // 使用Shoelace公式计算面积
        int j = (i + 1) % n;  // 下一点，最后一点的下一点是第一点
        area += contour[i].x * contour[j].y;
        area -= contour[j].x * contour[i].y;
    }
    area = fabs(area) / 2.0;  // 计算绝对值并除以2

    return area;
    
}

//------------------------------形态学操作平滑边缘------------------------------
// 自定义腐蚀操作
void erodeImage(const Mat& input, Mat& output, const Mat& kernel) {
    int kernelRows = kernel.rows;
    int kernelCols = kernel.cols;
    int kRowOffset = kernelRows / 2;
    int kColOffset = kernelCols / 2;

    output = input.clone(); // 复制输入图像到输出图像

    for (int i = kRowOffset; i < input.rows - kRowOffset; ++i) {
        for (int j = kColOffset; j < input.cols - kColOffset; ++j) {
            bool minPixel = true;
            // 遍历结构元素的每一个位置
            for (int ki = 0; ki < kernelRows; ++ki) {
                for (int kj = 0; kj < kernelCols; ++kj) {
                    int imgRow = i + ki - kRowOffset;
                    int imgCol = j + kj - kColOffset;
                    if (kernel.at<uchar>(ki, kj) == 255 && input.at<uchar>(imgRow, imgCol) == 0) {
                        minPixel = false;
                        break;
                    }
                }
                if (!minPixel) break;
            }
            output.at<uchar>(i, j) = minPixel ? 255 : 0;
        }
    }
}

// 自定义膨胀操作
void dilateImage(const Mat& input, Mat& output, const Mat& kernel) {
    int kernelRows = kernel.rows;
    int kernelCols = kernel.cols;
    int kRowOffset = kernelRows / 2;
    int kColOffset = kernelCols / 2;

    output = input.clone(); // 复制输入图像到输出图像

    for (int i = kRowOffset; i < input.rows - kRowOffset; ++i) {
        for (int j = kColOffset; j < input.cols - kColOffset; ++j) {
            bool maxPixel = false;
            // 遍历结构元素的每一个位置
            for (int ki = 0; ki < kernelRows; ++ki) {
                for (int kj = 0; kj < kernelCols; ++kj) {
                    int imgRow = i + ki - kRowOffset;
                    int imgCol = j + kj - kColOffset;
                    if (kernel.at<uchar>(ki, kj) == 255 && input.at<uchar>(imgRow, imgCol) == 255) {
                        maxPixel = true;
                        break;
                    }
                }
                if (maxPixel) break;
            }
            output.at<uchar>(i, j) = maxPixel ? 255 : 0;
        }
    }
}

// 自定义开运算（先腐蚀再膨胀）
void openImage(const Mat& input, Mat& output, const Mat& kernel) {
    Mat eroded;
    erodeImage(input, eroded, kernel);  // 先腐蚀
    dilateImage(eroded, output, kernel); // 后膨胀
}

// 自定义闭运算（先膨胀再腐蚀）
void closeImage(const Mat& input, Mat& output, const Mat& kernel) {
    Mat dilated;
    dilateImage(input, dilated, kernel);  // 先膨胀
    erodeImage(dilated, output, kernel);  // 后腐蚀
}

// 创建结构元素
Mat getStructureElement(int shape, Size ksize) {
    Mat kernel = Mat::zeros(ksize, CV_8UC1);
    int centerX = ksize.width / 2;
    int centerY = ksize.height / 2;

    if (shape == MORPH_ELLIPSE) {
        // 创建椭圆形结构元素
        for (int i = 0; i < ksize.height; ++i) {
            for (int j = 0; j < ksize.width; ++j) {
                int dx = j - centerX;
                int dy = i - centerY;
                if (dx * dx + dy * dy <= (centerX * centerX)) {
                    kernel.at<uchar>(i, j) = 255;
                }
            }
        }
    }
    return kernel;
}

//中值滤波
void medianBlurCustom(const Mat& input, Mat& output, int ksize) {
    // 图像大小和窗口大小
    int rows = input.rows;
    int cols = input.cols;
    int half_ksize = ksize / 2;

    // 输出图像初始化为输入图像，确保类型匹配
    output = input.clone();

    // 遍历图像中的每个像素，跳过边缘
    for (int i = half_ksize; i < rows - half_ksize; ++i) {
        for (int j = half_ksize; j < cols - half_ksize; ++j) {
            // 收集 3x3 窗口内的像素值
            std::vector<uchar> window;
            for (int di = -half_ksize; di <= half_ksize; ++di) {
                for (int dj = -half_ksize; dj <= half_ksize; ++dj) {
                    window.push_back(input.at<uchar>(i + di, j + dj));
                }
            }

            // 对窗口内的像素值进行排序
            std::sort(window.begin(), window.end());

            // 取中值
            output.at<uchar>(i, j) = window[window.size() / 2]; // 中值是排序后的中间元素
        }
    }
}

//------------------------距离变换的手动实现----------------------------
//距离变换
void distanceTransform(const Mat& input, Mat& output) {
    // 确保输入是二值图像
    CV_Assert(input.type() == CV_8UC1);

    int rows = input.rows;
    int cols = input.cols;

    // 初始化输出矩阵
    output = Mat(rows, cols, CV_32FC1, Scalar(FLT_MAX));

    // 第一遍扫描 - 从左上到右下
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (input.at<uchar>(i, j) > 0) { // 如果是前景像素
                float min_dist = FLT_MAX;

                // 检查左上方的像素
                if (i > 0) {
                    min_dist = min(min_dist, output.at<float>(i - 1, j) + 1);
                    if (j > 0)
                        min_dist = min(min_dist, output.at<float>(i - 1, j - 1) + 1.414f);
                    if (j < cols - 1)
                        min_dist = min(min_dist, output.at<float>(i - 1, j + 1) + 1.414f);
                }
                if (j > 0)
                    min_dist = min(min_dist, output.at<float>(i, j - 1) + 1);

                output.at<float>(i, j) = min_dist;
            }
            else { // 如果是背景像素
                output.at<float>(i, j) = 0;
            }
        }
    }

    // 第二遍扫描 - 从右下到左上
    for (int i = rows - 1; i >= 0; i--) {
        for (int j = cols - 1; j >= 0; j--) {
            if (input.at<uchar>(i, j) > 0) { // 只处理前景像素
                float min_dist = output.at<float>(i, j);

                // 检查右下方的像素
                if (i < rows - 1) {
                    min_dist = min(min_dist, output.at<float>(i + 1, j) + 1);
                    if (j > 0)
                        min_dist = min(min_dist, output.at<float>(i + 1, j - 1) + 1.414f);
                    if (j < cols - 1)
                        min_dist = min(min_dist, output.at<float>(i + 1, j + 1) + 1.414f);
                }
                if (j < cols - 1)
                    min_dist = min(min_dist, output.at<float>(i, j + 1) + 1);

                output.at<float>(i, j) = min_dist;
            }
        }
    }
}

// 双边滤波函数
void BilateralFilter(const Mat& src, Mat& dst, int d, double sigmaColor, double sigmaSpace) {
    CV_Assert(src.channels() == 1); // 确保输入为单通道灰度图像

    // 初始化输出图像
    dst = Mat::zeros(src.size(), src.type());

    // 滤波窗口的半径
    int radius = d / 2;

    // 预计算空间高斯权重
    vector<vector<double>> gaussianSpace(d, vector<double>(d));
    double spaceCoeff = -0.5 / (sigmaSpace * sigmaSpace);
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            gaussianSpace[i + radius][j + radius] = exp((i * i + j * j) * spaceCoeff);
        }
    }

    // 遍历每个像素
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            double sumWeights = 0.0;
            double sumFiltered = 0.0;

            // 遍历窗口
            for (int i = -radius; i <= radius; ++i) {
                for (int j = -radius; j <= radius; ++j) {
                    int neighborY = y + i;
                    int neighborX = x + j;

                    // 边界处理：镜像填充
                    neighborY = max(0, min(neighborY, src.rows - 1));
                    neighborX = max(0, min(neighborX, src.cols - 1));

                    // 当前像素值和邻域像素值
                    uchar centerVal = src.at<uchar>(y, x);
                    uchar neighborVal = src.at<uchar>(neighborY, neighborX);

                    // 计算颜色高斯权重
                    double intensityDiff = neighborVal - centerVal;
                    double gaussianColor = exp(-0.5 * (intensityDiff * intensityDiff) / (sigmaColor * sigmaColor));

                    // 组合总权重
                    double weight = gaussianSpace[i + radius][j + radius] * gaussianColor;

                    // 加权累加
                    sumWeights += weight;
                    sumFiltered += weight * neighborVal;
                }
            }

            // 归一化并更新像素值
            dst.at<uchar>(y, x) = static_cast<uchar>(sumFiltered / sumWeights);
        }
    }
}

// 清理文件夹函数（Windows版本）
void clearDirectory(const string& path) {
    string cmd = "del /Q " + path + "\\*.*";
    system(cmd.c_str());
}

// 定义像素点结构
struct Pixel {
    int x, y;     // 坐标
    float height; // 高度值（灰度值）

    Pixel(int x_, int y_, float height_) : x(x_), y(y_), height(height_) {}

    // 用于优先队列的比较
    bool operator<(const Pixel& other) const {
        return height > other.height; // 最小堆
    }
};

// 细胞统计结构体
struct CellStats {
    int total_cells;
    double avg_area;
    double min_area;
    double max_area;
    double std_dev;
    vector<double> cell_areas;
};

// 分水岭算法实现
void customWatershed(const Mat& image, Mat& markers, const Mat& cleanedBinary, const Mat& dist_inv, float minDist = 0.005, int windowSize = 14) {
    int label = 1;// 标记值从1开始
    int halfWindow = windowSize / 2;

    /******************* 第一阶段：标记点检测 *******************/
    // 在距离变换图（反相后）上寻找局部最小值作为初始标记点
    for (int i = halfWindow; i < dist_inv.rows - halfWindow; i++) {
        for (int j = halfWindow; j < dist_inv.cols - halfWindow; j++) {

            // 安全措施：跳过背景区域（去噪之后的二值化图像中不是细胞的黑色区域），
            // 只从【距离变换后反相得到的图片】中细胞所在的区域中寻找局部最小值点
            if (cleanedBinary.at<uchar>(i, j) == 0) continue;

            // 获取当前点的距离值：距离变换图的像素灰度值就是距离值
            float center = dist_inv.at<float>(i, j);

            // 跳过距离值小于阈值（设置的参数）的点，避免局部最小值的噪声干扰
            if (center < minDist) continue;

            /******************* 局部最小值检验 *******************/
            bool is_min = true;
            // 在指定大小的窗口内检查是否是局部最小值
            for (int di = -halfWindow; di <= halfWindow && is_min; di++) {
                for (int dj = -halfWindow; dj <= halfWindow; dj++) {
                    if (di == 0 && dj == 0) continue;
                    // 如果存在比当前点center更小的值，则不是局部最小值
                    if (dist_inv.at<float>(i + di, j + dj) < center) 
                    {
                        is_min = false;
                        break;
                    }
                }
            }

            /******************* 标记点间距检验 *******************/
            if (is_min) {
                // 检查周围是否已经存在其他标记点
                bool too_close = false;
                for (int di = -windowSize; di <= windowSize && !too_close; di++) {
                    for (int dj = -windowSize; dj <= windowSize; dj++) {
                        int ni = i + di;
                        int nj = j + dj;
                        // 如果在指定范围内发现其他标记点，则当前点不设为标记点
                        if (ni >= 0 && ni < markers.rows &&
                            nj >= 0 && nj < markers.cols &&
                            markers.at<int>(ni, nj) > 0) {
                            too_close = true;
                            break;
                        }
                    }
                }

                // 如果满足所有条件，设置为新的标记点
                if (!too_close) {
                    markers.at<int>(i, j) = label++;
                }
            }
        }
    }

    /******************* 第二阶段：分水岭生长 *******************/
    // 定义8邻域搜索方向
    const int dx[] = { -1, 1, 0, 0, -1, -1, 1, 1 };  // 8邻域x方向偏移
    const int dy[] = { 0, 0, -1, 1, -1, 1, -1, 1 };  // 8邻域y方向偏移
    /*
    上面这段代码定义了8邻域搜索方向：
    邻域示意图：
        P7  P2  P6
        P3  P0  P1
        P5  P4  P8
    其中P0是中心像素，其他P1-P8是8个邻域像素

    dx[]和dy[]数组定义了从中心像素到8个邻域像素的坐标偏移：
    dx[] = {-1,  1,  0,  0, -1, -1,  1,  1}
    dy[] = { 0,  0, -1,  1, -1,  1, -1,  1}
    对应的8个方向是：
    索引  dx  dy   方向
    0:   -1   0   左
    1:    1   0   右
    2:    0  -1   上
    3:    0   1   下
    4:   -1  -1   左上
    5:   -1   1   左下
    6:    1  -1   右上
    7:    1   1   右下

    这种方式比直接写8个if语句更简洁，通过数组索引就能访问到所有8个方向的相邻像素。
    */

    /******************* 初始化优先队列 *******************/
    // 使用优先队列进行基于高度的区域生长
    priority_queue<Pixel> pq;

    // 将所有初始标记点加入优先队列
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            if (markers.at<int>(i, j) > 0) {
                float height = image.at<uchar>(i, j);
                pq.push(Pixel(i, j, height));
            }
        }
    }

    /******************* 区域生长过程 *******************/
    while (!pq.empty()) {
        // 获取当前最低高度的像素
        Pixel current = pq.top();
        pq.pop();

        // 检查8邻域的每个方向
        // 遍历8个邻域
        for (int k = 0; k < 8; k++) {
            int newX = current.x + dx[k]; // 计算邻域像素的x坐标
            int newY = current.y + dy[k]; // 计算邻域像素的y坐标
            // 得到的(newX, newY)就是8个邻域像素中的一个

            // 检查是否在图像边界内
            if (newX >= 0 && newX < markers.rows &&
                newY >= 0 && newY < markers.cols) {

                // 标记未访问的邻域点
                if (markers.at<int>(newX, newY) == 0) {
                    // 获取当前区域的标记值，未访问过的邻域点设置为当前标记值
                    int currentLabel = markers.at<int>(current.x, current.y);

                    /******************* 分水岭判定 *******************/
                    // 检查是否是分水岭点（是否与不同标记区域相邻）
                    bool hasConflict = false;
                    for (int m = 0; m < 8; m++) {
                        int checkX = newX + dx[m];
                        int checkY = newY + dy[m];

                        if (checkX >= 0 && checkX < markers.rows &&
                            checkY >= 0 && checkY < markers.cols) {
                            int neighborLabel = markers.at<int>(checkX, checkY);
                            // 如果邻域存在不同的标记，则为分水岭点
                            if (neighborLabel > 0 && neighborLabel != currentLabel) {
                                hasConflict = true;
                                break;
                            }
                        }
                    }

                    // 根据判定结果进行标记
                    if (hasConflict) {
                        // 标记为分水岭点
                        markers.at<int>(newX, newY) = -1;
                    }
                    else {
                        // 延续当前区域的标记并加入生长队列
                        markers.at<int>(newX, newY) = currentLabel;
                        float newHeight = image.at<uchar>(newX, newY);
                        pq.push(Pixel(newX, newY, newHeight));
                    }
                }
            }
        }
    }
}

//合并小区域
vector<int> mergeSmallRegions(Mat& markers, int minArea) {
    // 1. 统计每个标记区域的大小
    map<int, int> regionSizes;
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int label = markers.at<int>(i, j);
            if (label > 0) {
                regionSizes[label]++;
            }
        }
    }
    cout << "区域大小统计：" << endl;
    for (const auto& region : regionSizes) {
        cout << "区域 " << region.first << " 大小: " << region.second << " 像素" << endl;
    }

    // 2. 找出需要合并的小区域
    vector<int> smallRegions;
    for (const auto& region : regionSizes) {
        if (region.second < minArea) {
            smallRegions.push_back(region.first);
        }
    }
    cout << "\n需要合并的小区域：" << endl;
    for (int label : smallRegions) {
        cout << "区域 " << label << " (大小: " << regionSizes[label] << " 像素)" << endl;
    }

    // 3. 为每个小区域找到最佳合并目标
    for (int smallLabel : smallRegions) {
        // 存储这个小区域的所有像素位置
        vector<Point> regionPixels;
        // 存储邻接的大区域标签及其边界接触次数
        map<int, int> neighborLabels;

        // 找出该区域的所有像素和边界信息
        for (int i = 0; i < markers.rows; i++) {
            for (int j = 0; j < markers.cols; j++) {
                if (markers.at<int>(i, j) == smallLabel) {
                    regionPixels.push_back(Point(j, i));

                    // 检查8邻域
                    for (int di = -1; di <= 1; di++) {
                        for (int dj = -1; dj <= 1; dj++) {
                            if (di == 0 && dj == 0) continue;

                            int ni = i + di;
                            int nj = j + dj;

                            if (ni >= 0 && ni < markers.rows &&
                                nj >= 0 && nj < markers.cols) {
                                int neighborLabel = markers.at<int>(ni, nj);
                                if (neighborLabel > 0 &&
                                    neighborLabel != smallLabel &&
                                    find(smallRegions.begin(), smallRegions.end(), neighborLabel) == smallRegions.end()) {
                                    neighborLabels[neighborLabel]++;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 找出接触最多的大区域
        int bestLabel = -1;
        int maxContact = 0;
        for (const auto& neighbor : neighborLabels) {
            if (neighbor.second > maxContact) {
                maxContact = neighbor.second;
                bestLabel = neighbor.first;
            }
        }

        // 如果找到合适的大区域，执行合并
        if (bestLabel != -1) {
            cout << "合并：区域 " << smallLabel << " -> 区域 " << bestLabel << endl;
            cout << "接触点数量: " << maxContact << endl;
            for (const Point& p : regionPixels) {
                markers.at<int>(p.y, p.x) = bestLabel;
            }
        }
    }

    // 4. 再次检查并合并孤立的小区域（迭代处理）
    bool needAnotherPass = true;
    int maxIterations = 3;  // 限制迭代次数
    int iteration = 0;

    while (needAnotherPass && iteration < maxIterations) {
        needAnotherPass = false;
        iteration++;

        // 更新区域大小统计
        regionSizes.clear();
        for (int i = 0; i < markers.rows; i++) {
            for (int j = 0; j < markers.cols; j++) {
                int label = markers.at<int>(i, j);
                if (label > 0) {
                    regionSizes[label]++;
                }
            }
        }

        // 处理剩余的小区域
        for (const auto& region : regionSizes) {
            if (region.second < 4000) {
                needAnotherPass = true;
                int currentLabel = region.first;

                // 对该小区域的每个像素
                for (int i = 0; i < markers.rows; i++) {
                    for (int j = 0; j < markers.cols; j++) {
                        if (markers.at<int>(i, j) == currentLabel) {
                            // 在更大的邻域中寻找最佳合并目标
                            map<int, int> neighborCounts;
                            int searchRadius = 3;  // 增加搜索范围

                            for (int di = -searchRadius; di <= searchRadius; di++) {
                                for (int dj = -searchRadius; dj <= searchRadius; dj++) {
                                    int ni = i + di;
                                    int nj = j + dj;

                                    if (ni >= 0 && ni < markers.rows &&
                                        nj >= 0 && nj < markers.cols) {
                                        int neighborLabel = markers.at<int>(ni, nj);
                                        if (neighborLabel > 0 &&
                                            neighborLabel != currentLabel &&
                                            regionSizes[neighborLabel] >= 4000) {
                                            neighborCounts[neighborLabel]++;
                                        }
                                    }
                                }
                            }

                            // 找出最频繁的邻居标签
                            int bestLabel = currentLabel;
                            int maxCount = 0;
                            for (const auto& neighbor : neighborCounts) {
                                if (neighbor.second > maxCount) {
                                    maxCount = neighbor.second;
                                    bestLabel = neighbor.first;
                                }
                            }

                            if (bestLabel != currentLabel) {
                                markers.at<int>(i, j) = bestLabel;
                            }
                        }
                    }
                }
            }
        }
    }

    // 统计最终的区域信息
    map<int, int> finalRegionSizes;
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int label = markers.at<int>(i, j);
            if (label > 0) {
                finalRegionSizes[label]++;
            }
        }
    }

    cout << "\n合并后的区域统计：" << endl;
    for (const auto& region : finalRegionSizes) {
        cout << "区域 " << region.first << " 最终大小: " << region.second << " 像素" << endl;
    }

    return smallRegions;
}

// 绘制轮廓
void DrawContours(Mat& img, const vector<vector<Point>>& contours, int contourIdx, const Scalar& color, int thickness) {
    if (contourIdx < 0 || contourIdx >= contours.size()) {
        cout << "轮廓索引超出范围!" << endl;
        return;
    }

    const vector<Point>& contour = contours[contourIdx];

    // 遍历每一对相邻点并绘制线条
    for (size_t i = 0; i < contour.size(); i++) {
        Point p1 = contour[i];
        Point p2 = contour[(i + 1) % contour.size()]; // 环绕连接首尾点

        // 绘制线段（Bresenham 算法）
        LineIterator it(img, p1, p2, 8); // 使用八邻域连接
        for (int j = 0; j < it.count; j++, ++it) {
            // 设置颜色
            if (thickness == 1) {
                img.at<Vec3b>(it.pos()) = Vec3b(color[0], color[1], color[2]);
            }
            else {
                // 增加线条厚度
                for (int dx = -thickness / 2; dx <= thickness / 2; dx++) {
                    for (int dy = -thickness / 2; dy <= thickness / 2; dy++) {
                        Point pt(it.pos().x + dx, it.pos().y + dy);
                        if (pt.x >= 0 && pt.y >= 0 && pt.x < img.cols && pt.y < img.rows) {
                            img.at<Vec3b>(pt) = Vec3b(color[0], color[1], color[2]);
                        }
                    }
                }
            }
        }
    }
}

// 计算细胞的统计信息
CellStats analyzeCells(const Mat& segmented_binary, const Mat& original_image, Mat& numbered_result) {
    CellStats stats;

    // 查找轮廓
    vector<vector<Point>> cell_contours;
    findContours(segmented_binary, cell_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 统计基本信息
    stats.total_cells = cell_contours.size();
    double total_area = 0;
    stats.min_area = DBL_MAX;
    stats.max_area = 0;

    // 计算每个细胞的面积
    for (const auto& contour : cell_contours) {
        double area = contourArea(contour);
        stats.cell_areas.push_back(area);
        total_area += area;
        stats.min_area = min(stats.min_area, area);
        stats.max_area = max(stats.max_area, area);
    }

    // 计算平均面积
    stats.avg_area = total_area / stats.total_cells;

    // 计算标准差
    double variance = 0;
    for (double area : stats.cell_areas) {
        variance += pow(area - stats.avg_area, 2);
    }
    stats.std_dev = sqrt(variance / stats.total_cells);

    // 输出统计结果
    cout << "\n细胞统计结果：" << endl;
    cout << "总细胞数量: " << stats.total_cells << endl;
    cout << "平均面积: " << stats.avg_area << " 像素" << endl;
    cout << "最小面积: " << stats.min_area << " 像素" << endl;
    cout << "最大面积: " << stats.max_area << " 像素" << endl;
    cout << "面积标准差: " << stats.std_dev << " 像素" << endl;

    for (int i = 0; i < cell_contours.size(); i++) {
        // 计算轮廓的中心点
        Moments m = moments(cell_contours[i]);
        Point center(m.m10 / m.m00, m.m01 / m.m00);

        // 绘制轮廓
        DrawContours(numbered_result, cell_contours, i, Scalar(0, 255, 0), 2);

        // 添加编号
        putText(numbered_result,
            to_string(i + 1),
            center,
            FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar(0, 255, 255),
            2);
    }
    return stats;
}

// 打印图像信息
void printMatInfo(const string& filename) {
    Mat img = imread("res//" + filename, IMREAD_UNCHANGED);
    if (img.empty()) {
        cout << "无法读取图像: " << filename << endl;
        return;
    }

    string type;
    switch (img.type()) {
    case CV_8UC1:
        type = "CV_8UC1";
        break;
    case CV_8UC3:
        type = "CV_8UC3";
        break;
    case CV_32FC1:
        type = "CV_32FC1";
        break;
    case CV_32SC1:
        type = "CV_32SC1";
        break;
    default:
        type = "其他类型(" + to_string(img.type()) + ")";
    }

    cout << "图像: " << filename << endl;
    cout << "类型: " << type << endl;
    cout << "尺寸: " << img.size() << endl;
    cout << "通道数: " << img.channels() << endl;
    cout << "-------------------" << endl;
}

// 检查图像类型
void checkImageTypes() {
    vector<string> filenames = {
        "01_original.png",
        "02_grayscale.png",
        "03_binary.png",
        "04_cleaned_binary.png",
        "05_smoothed_binary.png",
        "06_inverted_distance_transform.png",
        "07_watershed_lines.png",
        "08_final_result.png",
        "09_segmented_binary.png",
        "10_numbered_cells.png"
    };

    cout << "\n检查图像类型：" << endl;
    for (const auto& filename : filenames) {
        printMatInfo(filename);
    }
}

// 手动实现形态学开运算，用于可视化处理之后
void morphologicalOpen(Mat& image) {
    // 创建3x3结构元素
    const int kernelSize = 3;
    const int offset = kernelSize / 2;

    // 先进行腐蚀
    Mat eroded = Mat::zeros(image.size(), CV_8U);
    for (int i = offset; i < image.rows - offset; i++) {
        for (int j = offset; j < image.cols - offset; j++) {
            bool isValid = true;
            // 检查3x3邻域
            for (int di = -offset; di <= offset && isValid; di++) {
                for (int dj = -offset; dj <= offset && isValid; dj++) {
                    if (image.at<uchar>(i + di, j + dj) == 0) {
                        isValid = false;
                    }
                }
            }
            eroded.at<uchar>(i, j) = isValid ? 255 : 0;
        }
    }

    // 再进行膨胀
    Mat dilated = Mat::zeros(image.size(), CV_8U);
    for (int i = offset; i < image.rows - offset; i++) {
        for (int j = offset; j < image.cols - offset; j++) {
            bool hasWhite = false;
            // 检查3x3邻域
            for (int di = -offset; di <= offset && !hasWhite; di++) {
                for (int dj = -offset; dj <= offset && !hasWhite; dj++) {
                    if (eroded.at<uchar>(i + di, j + dj) == 255) {
                        hasWhite = true;
                    }
                }
            }
            dilated.at<uchar>(i, j) = hasWhite ? 255 : 0;
        }
    }

    image = dilated.clone();
}

// 可视化处理函数
void visualizeResults(const Mat& image, const Mat& markers, const Mat& cleanedBinary,
    const vector<int>& smallRegions, Mat& result, Mat& watershed_lines, Mat& segmented) {
    result = image.clone();
    watershed_lines = Mat::zeros(markers.size(), CV_8U);
    segmented = cleanedBinary.clone();

    // 创建一个集合来存储已合并的标签对
    set<pair<int, int>> mergedPairs;

    // 收集合并信息
    for (int smallLabel : smallRegions) {
        for (int i = 0; i < markers.rows; i++) {
            for (int j = 0; j < markers.cols; j++) {
                if (markers.at<int>(i, j) == smallLabel) {
                    for (int di = -1; di <= 1; di++) {
                        for (int dj = -1; dj <= 1; dj++) {
                            if (di == 0 && dj == 0) continue;

                            int ni = i + di;
                            int nj = j + dj;

                            if (ni >= 0 && ni < markers.rows &&
                                nj >= 0 && nj < markers.cols) {
                                int neighborLabel = markers.at<int>(ni, nj);
                                if (neighborLabel > 0 && neighborLabel != smallLabel) {
                                    mergedPairs.insert({ min(smallLabel, neighborLabel),
                                                       max(smallLabel, neighborLabel) });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 绘制分水岭线
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            if (markers.at<int>(i, j) == -1) {
                int label1 = -1, label2 = -1;

                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        if (di == 0 && dj == 0) continue;

                        int ni = i + di;
                        int nj = j + dj;

                        if (ni >= 0 && ni < markers.rows &&
                            nj >= 0 && nj < markers.cols) {
                            int neighborLabel = markers.at<int>(ni, nj);
                            if (neighborLabel > 0) {
                                if (label1 == -1) {
                                    label1 = neighborLabel;
                                }
                                else if (label2 == -1 && neighborLabel != label1) {
                                    label2 = neighborLabel;
                                    break;
                                }
                            }
                        }
                    }
                }

                if (label1 != -1 && label2 != -1) {
                    pair<int, int> currentPair = { min(label1, label2), max(label1, label2) };
                    if (mergedPairs.find(currentPair) == mergedPairs.end()) {
                        result.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
                        watershed_lines.at<uchar>(i, j) = 255;
                        segmented.at<uchar>(i, j) = 0;
                    }
                }
            }
        }
    }

    // 对分割结果进行形态学开运算
    morphologicalOpen(segmented);
}

// 主函数
int main() {
    // 创建输出目录
    system("mkdir res");

    // 清理res文件夹
    clearDirectory("res");

    // 1. 读取图像
    Mat image = imread("cells.png");
    if (image.empty()) {
        cout << "无法读取图像" << endl;
        return -1;
    }

    // 2. 转换为灰度图
    Mat gray;
    cvtColorToGray(image, gray);

    // 3. 二值化 (确保细胞是黑色，背景是白色)
    Mat binary;
    thresholdBinaryInvOtsu(gray, binary);

    // 4. 去除小面积区域
    
    // 创建一个用于标记已访问位置的Mat
    Mat visited = Mat::zeros(binary.size(), CV_8U);
    vector<vector<Point>> contours;
    double minArea = 100;  // 小区域的面积阈值

    // 遍历每个像素，寻找连通区域
    for (int y = 0; y < binary.rows; ++y) {
        for (int x = 0; x < binary.cols; ++x) {
            if (binary.at<uchar>(y, x) == 255 && visited.at<uchar>(y, x) == 0) {
                // 如果像素点是前景且未被访问过，执行洪泛填充（Flood Fill）
                vector<Point> contour;
                floodFill(binary, visited, x, y, contour);

                // 计算该轮廓的面积
                int area = contourArea(contour);

                // 如果面积大于阈值，则保留该轮廓
                if (area > minArea) {
                    contours.push_back(contour);
                }
            }
        }
    }

    // 创建清理后的二值图像
    Mat cleanedBinary = Mat::zeros(binary.size(), CV_8U);

    // 将大于阈值的轮廓绘制到清理后的图像上
    for (const auto& contour : contours) {
        for (const Point& pt : contour) {
            cleanedBinary.at<uchar>(pt.y, pt.x) = 255;
        }
    }
    
    // 4.1 对二值图像进行形态学操作来平滑边缘
    
    // 定义结构元素
    Mat kernel = getStructureElement(2, Size(3, 3));
    // 进行开运算
    Mat smoothedBinary;
    openImage(cleanedBinary, smoothedBinary, kernel);
    // 进行闭运算
    closeImage(smoothedBinary, smoothedBinary, kernel);

    // 4.2 使用中值滤波进一步平滑
    medianBlurCustom(smoothedBinary, smoothedBinary, 3);

    // 5. 距离变换 (使用平滑后的二值图像)
    Mat dist;
    distanceTransform(smoothedBinary, dist);
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);

    // 使用较大的核进行高斯模糊
    GaussianBlur(dist, dist, Size(5, 5), 0);

    // 6. 距离变换取反
    Mat dist_inv = 1- dist; // 距离变换取反的结果
    Mat dist_inv_vis;// 距离变换取反的可视化结果
    dist_inv.convertTo(dist_inv_vis, CV_8U, 255.0);

    // 对距离变换结果进行边缘保持滤波
    Mat filtered_dist;
    BilateralFilter(dist_inv_vis, filtered_dist, 5, 50, 50);
    dist_inv_vis = filtered_dist;

    // 7. 应用分水岭算法
    Mat markers = Mat::zeros(dist_inv.size(), CV_32S);
    customWatershed(dist_inv_vis, markers, cleanedBinary, dist_inv, 0.005, 14);    /* 参数设置如下： */    // 修改参数以适应平滑后的图像 float minDist = 0.005;    // 两个涨水种子点之间的最小距离，值越大筛选越严格，默认参数0.005    // 注意图像的像素值的范围是0-1    // int windowSize = 14;     // 搜索窗口大小，影响标记点的密度，默认参数14
    // 8. 合并小区域
    vector<int> smallRegions = mergeSmallRegions(markers, 4000);
   
    // 9. 可视化结果
    Mat result = image.clone();
    Mat watershed_lines = Mat::zeros(markers.size(), CV_8U);
    Mat segmented = cleanedBinary.clone();

    visualizeResults(image, markers, cleanedBinary, smallRegions,
        result, watershed_lines, segmented);

    // 10. 调用细胞统计分析函数
    Mat numbered_result = image.clone(); // 直接使用原始图像的副本
    CellStats cell_statistics = analyzeCells(segmented, image, numbered_result);

    // 输出细胞面积分布
    if (cell_statistics.total_cells > 0) {
        cout << "\n细胞面积分布：" << endl;
        for (int i = 0; i < cell_statistics.total_cells; i++) {
            cout << "细胞 " << (i + 1) << " 面积: " << cell_statistics.cell_areas[i] << " 像素" << endl;
        }
    }

    // 文件保存
    imwrite("res//01_original.png", image);                          // 原始图像
    imwrite("res//02_grayscale.png", gray);                         // 灰度图
    imwrite("res//03_binary.png", binary);                          // 二值图
    imwrite("res//04_cleaned_binary.png", cleanedBinary);           // 去除小区域后的二值图
    imwrite("res//05_smoothed_binary.png", smoothedBinary);         // 平滑后的二值图
    imwrite("res//06_inverted_distance_transform.png", dist_inv_vis); // 距离变换取反
    imwrite("res//07_watershed_lines.png", watershed_lines);         // 分水岭线
    imwrite("res//08_watershed_result.png", result);                 // 分水岭算法结果
    imwrite("res//09_segmented_binary.png", segmented);             // 分割后的二值图
    imwrite("res//10_numbered_cells.png", numbered_result);// 保存带编号的结果图

    // 检查图像类型
    // checkImageTypes();
    /*
    检查图像类型：
    图像: 01_original.png
    类型: CV_8UC3
    尺寸: [1361 x 654]
    通道数: 3
    -------------------
    图像: 02_grayscale.png
    类型: CV_8UC1
    尺寸: [1361 x 654]
    通道数: 1
    -------------------
    图像: 03_binary.png
    类型: CV_8UC1
    尺寸: [1361 x 654]
    通道数: 1
    -------------------
    图像: 04_cleaned_binary.png
    类型: CV_8UC1
    尺寸: [1361 x 654]
    通道数: 1
    -------------------
    图像: 05_smoothed_binary.png
    类型: CV_8UC1
    尺寸: [1361 x 654]
    通道数: 1
    -------------------
    图像: 06_inverted_distance_transform.png
    类型: CV_8UC1
    尺寸: [1361 x 654]
    通道数: 1
    -------------------
    图像: 07_watershed_lines.png
    类型: CV_8UC1
    尺寸: [1361 x 654]
    通道数: 1
    -------------------
    图像: 08_final_result.png
    类型: CV_8UC3
    尺寸: [1361 x 654]
    通道数: 3
    -------------------
    图像: 09_segmented_binary.png
    类型: CV_8UC1
    尺寸: [1361 x 654]
    通道数: 1
    -------------------
    图像: 10_numbered_cells.png
    类型: CV_8UC3
    尺寸: [1361 x 654]
    通道数: 3
    -------------------
    */

    return 0;
}