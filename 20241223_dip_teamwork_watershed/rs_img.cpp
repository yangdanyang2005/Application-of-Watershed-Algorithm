//#define _CRT_SECURE_NO_WARNINGS 1
//#define PI acos(-1)
//
//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <queue>
//#include <stack>
//#include <fstream>  // 用于文件操作
//#include <iomanip>  // 用于设置输出精度
//
//using namespace cv;
//using namespace std;
//
//// 清理文件夹函数（Windows版本）
//void clearDirectory(const string& path) {
//	string cmd = "del /Q " + path + "\\*.*";
//	system(cmd.c_str());
//}
//
////---------------超分辨率---------------
//
//// 双三次插值核函数
//double cubicKernel(double x) {
//	x = abs(x);
//	if (x <= 1)
//		return 1.5 * x * x * x - 2.5 * x * x + 1;
//	else if (x < 2)
//		return -0.5 * x * x * x + 2.5 * x * x - 4 * x + 2;
//	else
//		return 0;
//}
//
//// 双三次插值实现
//Mat bicubicinterpolation(const Mat& src, double scale) {
//	int newRows = round(src.rows * scale);
//	int newCols = round(src.cols * scale);
//	Mat dst = Mat::zeros(newRows, newCols, src.type());
//
//	// 对每个目标像素进行插值
//	for (int y = 0; y < newRows; y++) {
//		for (int x = 0; x < newCols; x++) {
//			// 计算在原图中的对应位置
//			double srcX = x / scale;
//			double srcY = y / scale;
//
//			// 计算整数部分和小数部分
//			int x0 = floor(srcX);
//			int y0 = floor(srcY);
//			double dx = srcX - x0;
//			double dy = srcY - y0;
//
//			double sum = 0;
//			double weightSum = 0;
//
//			// 16个邻近点的加权和
//			for (int i = -1; i <= 2; i++) {
//				for (int j = -1; j <= 2; j++) {
//					int xi = x0 + j;
//					int yi = y0 + i;
//
//					// 边界检查
//					if (xi >= 0 && xi < src.cols && yi >= 0 && yi < src.rows) {
//						double weight = cubicKernel(j - dx) * cubicKernel(i - dy);
//						sum += src.at<uchar>(yi, xi) * weight;
//						weightSum += weight;
//					}
//				}
//			}
//
//			// 归一化并确保值在有效范围内
//			int value = round(sum / weightSum);
//			value = max(0, min(255, value));
//			dst.at<uchar>(y, x) = value;
//		}
//	}
//
//	return dst;
//}
//
////---------------噪声去除---------------
//
//// 生成高斯核函数
//Mat generateGaussianKernel(int ksize, double sigma) {
//	int radius = ksize / 2;
//	Mat kernel(ksize, ksize, CV_64F);
//	double sum = 0.0;
//
//	// 计算高斯核的每个值
//	for (int i = -radius; i <= radius; ++i) {
//		for (int j = -radius; j <= radius; ++j) {
//			double value = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * PI * sigma * sigma);
//			kernel.at<double>(i + radius, j + radius) = value;
//			sum += value;
//		}
//	}
//
//	// 归一化高斯核，使得所有值的和为1
//	kernel /= sum;
//	return kernel;
//}
//
//// 应用高斯滤波器
//Mat applyGaussianFilter(const Mat& src, const Mat& kernel) {
//	int ksize = kernel.rows;
//	int radius = ksize / 2;
//	Mat dst = src.clone();
//
//	// 遍历每一个像素点
//	for (int i = radius; i < src.rows - radius; ++i) {
//		for (int j = radius; j < src.cols - radius; ++j) {
//			double sum = 0.0;
//
//			// 卷积操作
//			for (int m = -radius; m <= radius; ++m) {
//				for (int n = -radius; n <= radius; ++n) {
//					int pixel = src.at<uchar>(i + m, j + n);
//					double kernel_value = kernel.at<double>(m + radius, n + radius);
//					sum += pixel * kernel_value;
//				}
//			}
//
//			dst.at<uchar>(i, j) = static_cast<uchar>(sum);
//		}
//	}
//
//	return dst;
//}
//
//// 高斯去噪函数
//Mat denoiseImage(const Mat& src) {
//	// 设置高斯核大小和标准差
//	int ksize = 5;    // 高斯核的大小
//	double sigma = 1.0;  // 高斯核的标准差
//
//	// 生成高斯核
//	Mat gaussianKernel = generateGaussianKernel(ksize, sigma);
//
//	// 应用高斯滤波
//	Mat dst = applyGaussianFilter(src, gaussianKernel);
//
//	return dst;
//}
//
////---------------三值化---------------
//
//// 统计灰度值分布
//void calcGrayHist(const Mat& image, Mat& histImage, const string& outputPath) {
//	// 1. 统计灰度值频率
//	vector<int> histogram(256, 0);
//	for (int i = 0; i < image.rows; i++) {
//		for (int j = 0; j < image.cols; j++) {
//			histogram[image.at<uchar>(i, j)]++;
//		}
//	}
//
//	// 2. 找出最大频率
//	int maxFreq = *max_element(histogram.begin(), histogram.end());
//
//	// 3. 创建美观的直方图图像
//	int hist_h = 600;    // 增加高度使图像更清晰
//	int hist_w = 800;    // 增加宽度
//	int margin = 50;     // 边距
//	int graph_h = hist_h - 2 * margin;  // 实际图形区域高度
//	int graph_w = hist_w - 2 * margin;  // 实际图形区域宽度
//	int bin_w = graph_w / 256;          // 每个灰度级的宽度
//
//	// 创建白色背景的图像
//	histImage = Mat(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
//
//	// 4. 绘制坐标轴
//	line(histImage,
//		Point(margin, hist_h - margin),
//		Point(hist_w - margin, hist_h - margin),
//		Scalar(0, 0, 0), 2);  // X轴
//	line(histImage,
//		Point(margin, hist_h - margin),
//		Point(margin, margin),
//		Scalar(0, 0, 0), 2);  // Y轴
//
//	// 5. 绘制直方图柱状
//	for (int i = 0; i < 256; i++) {
//		int height = cvRound((double)histogram[i] * graph_h / maxFreq);
//		rectangle(histImage,
//			Point(margin + i * bin_w, hist_h - margin),
//			Point(margin + (i + 1) * bin_w - 1, hist_h - margin - height),
//			Scalar(100, 100, 250),  // 淡红色填充
//			FILLED);
//		rectangle(histImage,
//			Point(margin + i * bin_w, hist_h - margin),
//			Point(margin + (i + 1) * bin_w - 1, hist_h - margin - height),
//			Scalar(0, 0, 0),  // 黑色边框
//			1);
//	}
//
//	// 6. 添加刻度和标签
//	// X轴刻度
//	for (int i = 0; i <= 256; i += 32) {
//		line(histImage,
//			Point(margin + i * bin_w, hist_h - margin),
//			Point(margin + i * bin_w, hist_h - margin + 5),
//			Scalar(0, 0, 0), 1);
//		putText(histImage,
//			to_string(i),
//			Point(margin + i * bin_w - 10, hist_h - margin + 20),
//			FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0));
//	}
//
//	// Y轴刻度
//	int numYTicks = 10;
//	for (int i = 0; i <= numYTicks; i++) {
//		int y = hist_h - margin - (i * graph_h / numYTicks);
//		line(histImage,
//			Point(margin - 5, y),
//			Point(margin, y),
//			Scalar(0, 0, 0), 1);
//		putText(histImage,
//			to_string(i * maxFreq / numYTicks),
//			Point(5, y + 5),
//			FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0));
//	}
//
//	// 7. 添加标题和轴标签
//	putText(histImage,
//		"Grayscale Histogram",
//		Point(hist_w / 2 - 100, 30),
//		FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 2);
//	putText(histImage,
//		"Grayscale Value",
//		Point(hist_w / 2 - 50, hist_h - 10),
//		FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
//	putText(histImage,
//		"Frequency",
//		Point(10, hist_h / 2),
//		FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
//
//	// 8. 将统计结果写入文件
//	ofstream outFile(outputPath + "/histogram_stats.txt");
//	if (outFile.is_open()) {
//		outFile << "灰度直方图统计结果\n" << endl;
//		outFile << "灰度值\t频率\t归一化频率\n" << endl;
//
//		// 计算一些统计量
//		long long totalPixels = image.rows * image.cols;
//		double sum = 0;
//		for (int i = 0; i < 256; i++) {
//			if (histogram[i] > 0) {
//				sum += i * histogram[i];
//				double normalizedFreq = (double)histogram[i] / totalPixels;
//				outFile << i << "\t" << histogram[i] << "\t" << fixed
//					<< setprecision(6) << normalizedFreq << endl;
//			}
//		}
//
//		// 输出统计信息
//		outFile << "\n统计信息：" << endl;
//		outFile << "总像素数：" << totalPixels << endl;
//		outFile << "最大频率：" << maxFreq << endl;
//		outFile << "平均灰度值：" << sum / totalPixels << endl;
//
//		outFile.close();
//	}
//}
//
//// 自定义三值化处理函数，返回背景、前景和过渡区域
//Mat threeValuedThreshold(const Mat& src, int lowThresh, int highThresh) {
//	Mat result = src.clone();
//	for (int y = 0; y < src.rows; ++y) {
//		for (int x = 0; x < src.cols; ++x) {
//			int pixel = src.at<uchar>(y, x);
//			if (pixel < lowThresh) {
//				result.at<uchar>(y, x) = 0;    // 背景
//			}
//			else if (pixel > highThresh) {
//				result.at<uchar>(y, x) = 255;  // 前景
//			}
//			else {
//				result.at<uchar>(y, x) = 127;  // 过渡区域
//			}
//		}
//	}
//	return result;
//}
//
////---------------梯度化---------------
//
//// 计算图像梯度：使用Prewitt算子
//Mat computeGradient(const Mat& src) {
//	// 确保图像是灰度图
//	CV_Assert(src.channels() == 1);
//
//	int rows = src.rows;
//	int cols = src.cols;
//
//	// Prewitt算子的水平和垂直卷积核
//	int sobelX[3][3] = { {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1} };
//	int sobelY[3][3] = { {-1, -1, -1}, {0, 0, 0}, {1, 1, 1} };
//
//	Mat grad_x = Mat::zeros(src.size(), CV_32F);
//	Mat grad_y = Mat::zeros(src.size(), CV_32F);
//	Mat grad = Mat::zeros(src.size(), CV_32F);
//
//	// 对图像进行 Prewitt 卷积
//	for (int i = 1; i < rows - 1; ++i) {
//		for (int j = 1; j < cols - 1; ++j) {
//			float gradX = 0.0f;
//			float gradY = 0.0f;
//
//			// 卷积计算水平梯度 grad_x
//			for (int k = -1; k <= 1; ++k) {
//				for (int l = -1; l <= 1; ++l) {
//					gradX += src.at<uchar>(i + k, j + l) * sobelX[k + 1][l + 1];
//					gradY += src.at<uchar>(i + k, j + l) * sobelY[k + 1][l + 1];
//				}
//			}
//
//			grad_x.at<float>(i, j) = gradX;
//			grad_y.at<float>(i, j) = gradY;
//
//			// 计算梯度的幅度
//			grad.at<float>(i, j) = sqrt(gradX * gradX + gradY * gradY);
//		}
//	}
//
//	// 将计算结果转换为 8 位图像
//	Mat grad_8u;
//	grad.convertTo(grad_8u, CV_8U);
//
//	return grad_8u;
//}
//
////---------------去除小区域---------------
//
//// 检查图像内的连通区域
//void floodFill(const Mat& binary, Mat& visited, int x, int y, vector<Point>& contour) {
//	int rows = binary.rows, cols = binary.cols;
//	queue<Point> q;
//	q.push(Point(x, y));
//	visited.at<uchar>(y, x) = 1;
//
//	while (!q.empty()) {
//		Point p = q.front();
//		q.pop();
//		contour.push_back(p);
//
//		// 访问四个邻域
//		for (int dx = -1; dx <= 1; ++dx) {
//			for (int dy = -1; dy <= 1; ++dy) {
//				if (dx == 0 && dy == 0) continue;
//
//				int nx = p.x + dx, ny = p.y + dy;
//
//				// 检查边界条件
//				if (nx >= 0 && nx < cols && ny >= 0 && ny < rows &&
//					binary.at<uchar>(ny, nx) == 255 && visited.at<uchar>(ny, nx) == 0) {
//					visited.at<uchar>(ny, nx) = 1;
//					q.push(Point(nx, ny));
//				}
//			}
//		}
//	}
//}
//
//// 计算轮廓的面积（简单地计算轮廓内的像素点数量）
//int contourArea(const vector<Point>& contour)
//{
//	int n = contour.size();
//	if (n < 3) {
//		// 不是一个有效的多边形，无法计算面积
//		return 0;
//	}
//
//	double area = 0.0;
//	for (int i = 0; i < n; i++) {
//		// 使用Shoelace公式计算面积
//		int j = (i + 1) % n;  // 下一点，最后一点的下一点是第一点
//		area += contour[i].x * contour[j].y;
//		area -= contour[j].x * contour[i].y;
//	}
//	area = fabs(area) / 2.0;  // 计算绝对值并除以2
//
//	return area;
//
//}
//
//// 移除小区域
//Mat removeSmallRegions(const Mat& input, int minArea) {
//	// 创建一个用于标记已访问位置的Mat
//	Mat visited = Mat::zeros(input.size(), CV_8U);
//	vector<vector<Point>> contours;
//
//
//	// 遍历每个像素，寻找连通区域
//	for (int y = 0; y < input.rows; ++y) {
//		for (int x = 0; x < input.cols; ++x) {
//			if (input.at<uchar>(y, x) == 255 && visited.at<uchar>(y, x) == 0) {
//				// 如果像素点是前景且未被访问过，执行洪泛填充（Flood Fill）
//				vector<Point> contour;
//				floodFill(input, visited, x, y, contour);
//
//				// 计算该轮廓的面积
//				int area = contourArea(contour);
//
//				// 如果面积大于阈值，则保留该轮廓
//				if (area > minArea) {
//					contours.push_back(contour);
//				}
//			}
//		}
//	}
//
//	// 创建清理后的二值图像
//	Mat cleanedBinary = Mat::zeros(input.size(), CV_8U);
//
//	// 将大于阈值的轮廓绘制到清理后的图像上
//	for (const auto& contour : contours) {
//		for (const Point& pt : contour) {
//			cleanedBinary.at<uchar>(pt.y, pt.x) = 255;
//		}
//	}
//
//	return cleanedBinary;
//}
//
////---------------分水岭算法---------------
//
//// 将图像转换为灰度图
//void convertGrayToBGRA(const Mat& original, Mat& output) {
//	// 确保原始图像是灰度图
//	if (original.empty() || original.channels() != 1) {
//		std::cerr << "Input image must be a grayscale image!" << std::endl;
//		return;
//	}
//
//	// 创建与输入图像大小相同的BGRA图像
//	output.create(original.rows, original.cols, CV_8UC4);
//
//	// 遍历每个像素，将灰度值映射到BGRA格式
//	for (int y = 0; y < original.rows; ++y) {
//		for (int x = 0; x < original.cols; ++x) {
//			// 获取灰度值
//			uchar gray = original.at<uchar>(y, x);
//
//			// 设置BGRA图像的每个通道
//			output.at<Vec4b>(y, x)[0] = gray;  // Blue channel
//			output.at<Vec4b>(y, x)[1] = gray;  // Green channel
//			output.at<Vec4b>(y, x)[2] = gray;  // Red channel
//			output.at<Vec4b>(y, x)[3] = 255;  // Alpha channel (fully opaque)
//		}
//	}
//}
//
//// 定义一个函数来实现连通区域标记
//int getConnectedComponents(const Mat& binaryImage, Mat& labels, Mat& stats, Mat& centroids) {
//	// 获取图像的尺寸
//	int rows = binaryImage.rows;
//	int cols = binaryImage.cols;
//
//	// 初始化labels，stats，centroids
//	labels = Mat::zeros(binaryImage.size(), CV_32S);  // 每个像素的标签
//	stats = Mat::zeros(1, 5, CV_32S);  // 每个连通区域的统计信息
//	centroids = Mat::zeros(1, 2, CV_32F);  // 每个连通区域的质心
//
//	// 用于标记连通区域的标签
//	int labelCount = 0;
//
//	// 方向数组，8连通：上下左右和四个对角线方向
//	int dx[8] = { -1, 1, 0, 0, -1, -1, 1, 1 };
//	int dy[8] = { 0, 0, -1, 1, -1, 1, -1, 1 };
//
//	// 用于存储区域的统计信息
//	vector<Rect> boundingBoxes;
//	vector<Point> regionCentroids;
//	vector<int> pixelCounts;
//
//	// 深度优先搜索 (DFS)
//	auto dfs = [&](int x, int y, int label, vector<Point>& regionPixels) {
//		vector<Point> stack;
//		stack.push_back(Point(x, y));
//		regionPixels.push_back(Point(x, y));
//		labels.at<int>(x, y) = label;
//
//		while (!stack.empty()) {
//			Point p = stack.back();
//			stack.pop_back();
//
//			// 遍历所有8个邻域
//			for (int i = 0; i < 8; i++) {
//				int nx = p.x + dx[i];
//				int ny = p.y + dy[i];
//
//				// 检查边界和前景像素
//				if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && labels.at<int>(nx, ny) == 0 && binaryImage.at<uchar>(nx, ny) == 255) {
//					labels.at<int>(nx, ny) = label;
//					stack.push_back(Point(nx, ny));
//					regionPixels.push_back(Point(nx, ny));
//				}
//			}
//		}
//		};
//
//	// 遍历整个图像，查找连通区域
//	for (int i = 0; i < rows; i++) {
//		for (int j = 0; j < cols; j++) {
//			if (binaryImage.at<uchar>(i, j) == 255 && labels.at<int>(i, j) == 0) {
//				// 找到一个新的连通区域
//				labelCount++;
//				vector<Point> regionPixels;
//				dfs(i, j, labelCount, regionPixels);
//
//				// 计算连通区域的边界框
//				int minX = rows, minY = cols, maxX = 0, maxY = 0;
//				float sumX = 0, sumY = 0;
//				int pixelCount = regionPixels.size();
//
//				for (const Point& p : regionPixels) {
//					minX = min(minX, p.x);
//					minY = min(minY, p.y);
//					maxX = max(maxX, p.x);
//					maxY = max(maxY, p.y);
//					sumX += p.x;
//					sumY += p.y;
//				}
//
//				// 存储区域统计信息
//				boundingBoxes.push_back(Rect(minY, minX, maxY - minY + 1, maxX - minX + 1));
//				pixelCounts.push_back(pixelCount);
//				regionCentroids.push_back(Point2f(sumY / pixelCount, sumX / pixelCount));
//			}
//		}
//	}
//
//	// 将统计信息存入stats和centroids
//	stats.create(labelCount, 5, CV_32S);
//	centroids.create(labelCount, 2, CV_32F);
//
//	for (int i = 0; i < labelCount; i++) {
//		stats.at<int>(i, 0) = boundingBoxes[i].x;
//		stats.at<int>(i, 1) = boundingBoxes[i].y;
//		stats.at<int>(i, 2) = boundingBoxes[i].width;
//		stats.at<int>(i, 3) = boundingBoxes[i].height;
//		stats.at<int>(i, 4) = pixelCounts[i];
//
//		centroids.at<float>(i, 0) = regionCentroids[i].x;
//		centroids.at<float>(i, 1) = regionCentroids[i].y;
//	}
//
//	return labelCount;
//}
//
//// 添加函数用于确定区域颜色
//Vec4b determineRegionColor(const Mat& Region, const Mat& thresholdedImage,
//	const Vec4b& darkGreen, const Vec4b& lightGreen, const Vec4b& yellow) {
//	// 统计该区域的灰度值分布
//	int count0 = 0, count127 = 0, count255 = 0;
//
//	for (int i = 0; i < Region.rows; i++) {
//		for (int j = 0; j < Region.cols; j++) {
//			if (Region.at<uchar>(i, j)) {
//				uchar val = thresholdedImage.at<uchar>(i, j);
//				if (val == 0) count0++;
//				else if (val == 127) count127++;
//				else if (val == 255) count255++;
//			}
//		}
//	}
//
//	// 确定区域颜色
//	Vec4b color;
//	if (count0 >= count127 && count0 >= count255) {
//		color = darkGreen;
//	}
//	else if (count127 >= count0 && count127 >= count255) {
//		color = lightGreen;
//	}
//	else {
//		color = yellow;
//	}
//
//	return color;
//}
//
//// 添加函数用于获取被红色线分割的区域
//vector<Mat> getRegionsFromRedLines(const Mat& overlay) {
//	// 创建二值图像表示红色边界
//	Mat redMask = Mat::zeros(overlay.size(), CV_8UC1);
//	for (int i = 0; i < overlay.rows; i++) {
//		for (int j = 0; j < overlay.cols; j++) {
//			Vec4b pixel = overlay.at<Vec4b>(i, j);
//			// 检测红色像素 (BGR: 0,0,255)
//			if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 255) {
//				redMask.at<uchar>(i, j) = 255;
//			}
//		}
//	}
//
//	// 创建用于标记的图像
//	Mat labels, stats, centroids;
//	Mat invRedMask = 255 - redMask; // 反转掩码，使区域为白色，边界为黑色
//	int nLabels = connectedComponentsWithStats(invRedMask, labels, stats, centroids, 8);
//
//	// 存储每个区域的掩码
//	vector<Mat> regions;
//	for (int label = 1; label < nLabels; label++) {
//		Mat regionMask = (labels == label);
//		regions.push_back(regionMask);
//	}
//
//	return regions;
//}
//
//// 修改后的颜色确定函数
//void colorRegionsBasedOnContent(Mat& overlay, const Mat& thresholdedImage,
//	const Vec4b& darkGreen, const Vec4b& lightGreen,
//	const Vec4b& yellow, const Vec4b& red) {
//	// 获取所有被红线分割的区域
//	vector<Mat> regions = getRegionsFromRedLines(overlay);
//
//	// 处理每个区域
//	for (const Mat& region : regions) {
//		// 统计该区域的灰度值分布
//		int count0 = 0, count127 = 0, count255 = 0;
//
//		for (int i = 0; i < region.rows; i++) {
//			for (int j = 0; j < region.cols; j++) {
//				if (region.at<uchar>(i, j)) {
//					uchar val = thresholdedImage.at<uchar>(i, j);
//					if (val == 0) count0++;
//					else if (val == 127) count127++;
//					else if (val == 255) count255++;
//				}
//			}
//		}
//
//		// 确定区域颜色
//		Vec4b regionColor;
//		if (count0 >= count127 && count0 >= count255) {
//			regionColor = darkGreen;
//		}
//		else if (count127 >= count0 && count127 >= count255) {
//			regionColor = lightGreen;
//		}
//		else {
//			regionColor = yellow;
//		}
//
//		// 应用颜色到区域
//		for (int i = 0; i < region.rows; i++) {
//			for (int j = 0; j < region.cols; j++) {
//				if (region.at<uchar>(i, j)) {
//					Vec4b& pixel = overlay.at<Vec4b>(i, j);
//					// 只对非红色边界像素应用新颜色
//					if (!(pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 255)) {
//						overlay.at<Vec4b>(i, j) = regionColor;
//					}
//				}
//			}
//		}
//	}
//}
//
//// 手动实现的分水岭算法
//Mat manualWatershed(const Mat& cleaned_grad, const Mat& thresholdedImage, const Mat& original) {
//
//	Mat output;
//	convertGrayToBGRA(original, output);
//
//	// 创建带边框的梯度图副本
//	Mat bordered_grad = cleaned_grad.clone();
//	rectangle(bordered_grad, Point(0, 0), Point(bordered_grad.cols - 1, bordered_grad.rows - 1), Scalar(255), 1);
//
//	// 创建遮罩层
//	Mat overlay = Mat::zeros(output.size(), CV_8UC4);
//
//	// 定义颜色
//	Vec4b darkGreen(50, 150, 50, 64);
//	Vec4b lightGreen(100, 255, 100, 64);
//	Vec4b yellow(0, 255, 255, 64);
//	Vec4b red(0, 0, 255, 255);
//
//	// 获取连通区域
//	Mat labels, stats, centroids;
//	int nLabels = getConnectedComponents(255 - bordered_grad, labels, stats, centroids);
//
//	// 创建分水岭标记图
//	Mat watershedMarkers = Mat::zeros(labels.size(), CV_32S);
//	Mat processed = Mat::zeros(labels.size(), CV_8U);
//
//	// 8邻域方向
//	int dx[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
//	int dy[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
//
//	// 处理每个连通区域
//	for (int label = 1; label < nLabels; label++) {
//		// 为当前区域创建队列
//		queue<Point> q;
//		Mat currentRegion = (labels == label);
//
//		// 找到区域中心点作为种子点
//		Point seed;
//		bool found = false;
//		for (int i = 0; i < currentRegion.rows && !found; i++) {
//			for (int j = 0; j < currentRegion.cols && !found; j++) {
//				if (currentRegion.at<uchar>(i, j)) {
//					seed = Point(j, i);
//					found = true;
//				}
//			}
//		}
//
//		// 将种子点加入队列
//		q.push(seed);
//		watershedMarkers.at<int>(seed) = label;
//		processed.at<uchar>(seed) = 255;
//
//		// 分水岭扩张过程
//		while (!q.empty()) {
//			Point current = q.front();
//			q.pop();
//
//			// 检查8邻域
//			for (int k = 0; k < 8; k++) {
//				int newX = current.x + dx[k];
//				int newY = current.y + dy[k];
//
//				// 检查边界
//				if (newX >= 0 && newX < labels.cols &&
//					newY >= 0 && newY < labels.rows) {
//
//					// 如果该点未被处理
//					if (!processed.at<uchar>(newY, newX)) {
//						// 检查是否与其他已扩张区域相邻
//						bool isBoundary = false;
//						for (int m = 0; m < 8; m++) {
//							int checkX = newX + dx[m];
//							int checkY = newY + dy[m];
//
//							if (checkX >= 0 && checkX < labels.cols &&
//								checkY >= 0 && checkY < labels.rows) {
//								// 如果邻点属于其他已处理区域，则当前点为边界
//								if (processed.at<uchar>(checkY, checkX) == 255 &&
//									watershedMarkers.at<int>(checkY, checkX) != label) {
//									isBoundary = true;
//									break;
//								}
//							}
//						}
//
//						if (isBoundary) {
//							// 标记为边界点
//							overlay.at<Vec4b>(newY, newX) = red;
//							processed.at<uchar>(newY, newX) = 255;
//						}
//						else {
//							// 将点加入当前区域
//							watershedMarkers.at<int>(newY, newX) = label;
//							processed.at<uchar>(newY, newX) = 255;
//							q.push(Point(newX, newY));
//						}
//					}
//				}
//			}
//		}
//	}
//
//	// 添加红色边界
//	for (int i = 0; i < bordered_grad.rows; i++) {
//		for (int j = 0; j < bordered_grad.cols; j++) {
//			if (bordered_grad.at<uchar>(i, j) == 255) {
//				overlay.at<Vec4b>(i, j) = red;
//			}
//		}
//	}
//
//	// 在完成分水岭扩张后，处理区域颜色
//
//	// 在处理每个区域时
//	colorRegionsBasedOnContent(overlay, thresholdedImage, darkGreen, lightGreen, yellow, red);
//
//	// 混合遮罩层与原图
//	for (int i = 0; i < output.rows; i++) {
//		for (int j = 0; j < output.cols; j++) {
//			Vec4b& pixel = overlay.at<Vec4b>(i, j);
//			Vec4b& outPixel = output.at<Vec4b>(i, j);
//
//			if (pixel[3] > 0) {
//				float alpha = pixel[3] / 255.0f * 0.7f;
//				for (int c = 0; c < 3; c++) {
//					outPixel[c] = saturate_cast<uchar>(
//						(1 - alpha) * outPixel[c] + alpha * pixel[c]
//					);
//				}
//				outPixel[3] = 255;
//			}
//		}
//	}
//
//	return output;
//}
//
//int main() {
//	// 读取图像并转换为灰度图像
//	Mat src = imread("教五草坪.tif", IMREAD_GRAYSCALE);
//	if (src.empty()) {
//		cout << "无法加载图像!" << endl;
//		return -1;
//	}
//
//	// 使用双三次插值将图像放大4倍
//	double scale = 4.0;
//	src = bicubicinterpolation(src, scale);
//
//	// 创建输出目录
//	system("mkdir res_g");
//
//	// 清理res文件夹
//	clearDirectory("res_g");
//
//	// 保存原图（已放大）
//	imwrite("res_g/1_original_scaled.png", src);
//
//	// 图像去噪
//	Mat denoisedImage = denoiseImage(src);
//	imwrite("res_g/2_denoised.png", denoisedImage);
//
//	// 统计灰度直方图
//	Mat hist;
//	calcGrayHist(denoisedImage, hist, "data");
//	imwrite("data/3_gray_hist.png", hist);
//
//	// 图像三值化
//	Mat thresholdedImage = threeValuedThreshold(denoisedImage, 100, 200);
//	imwrite("res_g/3_thresholded.png", thresholdedImage);
//
//	// 计算梯度图
//	Mat grad = computeGradient(thresholdedImage);
//	imwrite("res_g/4_gradient.png", grad);
//
//	// 移除小面积区域
//	// minArea参数需要根据实际情况调整
//	Mat cleaned_grad = removeSmallRegions(grad, 30);
//	imwrite("res_g/5_gradient_cleaned.png", cleaned_grad);
//
//	// 在移除小面积区域后添加
//	Mat coloredImage = manualWatershed(cleaned_grad, thresholdedImage, src);
//	imwrite("res_g/6_colored.png", coloredImage);
//
//	return 0;
//}