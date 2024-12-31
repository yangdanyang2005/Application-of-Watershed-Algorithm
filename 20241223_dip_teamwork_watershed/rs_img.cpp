//#define _CRT_SECURE_NO_WARNINGS 1
//#define PI acos(-1)
//
//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <queue>
//#include <stack>
//#include <fstream>  // �����ļ�����
//#include <iomanip>  // ���������������
//
//using namespace cv;
//using namespace std;
//
//// �����ļ��к�����Windows�汾��
//void clearDirectory(const string& path) {
//	string cmd = "del /Q " + path + "\\*.*";
//	system(cmd.c_str());
//}
//
////---------------���ֱ���---------------
//
//// ˫���β�ֵ�˺���
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
//// ˫���β�ֵʵ��
//Mat bicubicinterpolation(const Mat& src, double scale) {
//	int newRows = round(src.rows * scale);
//	int newCols = round(src.cols * scale);
//	Mat dst = Mat::zeros(newRows, newCols, src.type());
//
//	// ��ÿ��Ŀ�����ؽ��в�ֵ
//	for (int y = 0; y < newRows; y++) {
//		for (int x = 0; x < newCols; x++) {
//			// ������ԭͼ�еĶ�Ӧλ��
//			double srcX = x / scale;
//			double srcY = y / scale;
//
//			// �����������ֺ�С������
//			int x0 = floor(srcX);
//			int y0 = floor(srcY);
//			double dx = srcX - x0;
//			double dy = srcY - y0;
//
//			double sum = 0;
//			double weightSum = 0;
//
//			// 16���ڽ���ļ�Ȩ��
//			for (int i = -1; i <= 2; i++) {
//				for (int j = -1; j <= 2; j++) {
//					int xi = x0 + j;
//					int yi = y0 + i;
//
//					// �߽���
//					if (xi >= 0 && xi < src.cols && yi >= 0 && yi < src.rows) {
//						double weight = cubicKernel(j - dx) * cubicKernel(i - dy);
//						sum += src.at<uchar>(yi, xi) * weight;
//						weightSum += weight;
//					}
//				}
//			}
//
//			// ��һ����ȷ��ֵ����Ч��Χ��
//			int value = round(sum / weightSum);
//			value = max(0, min(255, value));
//			dst.at<uchar>(y, x) = value;
//		}
//	}
//
//	return dst;
//}
//
////---------------����ȥ��---------------
//
//// ���ɸ�˹�˺���
//Mat generateGaussianKernel(int ksize, double sigma) {
//	int radius = ksize / 2;
//	Mat kernel(ksize, ksize, CV_64F);
//	double sum = 0.0;
//
//	// �����˹�˵�ÿ��ֵ
//	for (int i = -radius; i <= radius; ++i) {
//		for (int j = -radius; j <= radius; ++j) {
//			double value = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * PI * sigma * sigma);
//			kernel.at<double>(i + radius, j + radius) = value;
//			sum += value;
//		}
//	}
//
//	// ��һ����˹�ˣ�ʹ������ֵ�ĺ�Ϊ1
//	kernel /= sum;
//	return kernel;
//}
//
//// Ӧ�ø�˹�˲���
//Mat applyGaussianFilter(const Mat& src, const Mat& kernel) {
//	int ksize = kernel.rows;
//	int radius = ksize / 2;
//	Mat dst = src.clone();
//
//	// ����ÿһ�����ص�
//	for (int i = radius; i < src.rows - radius; ++i) {
//		for (int j = radius; j < src.cols - radius; ++j) {
//			double sum = 0.0;
//
//			// �������
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
//// ��˹ȥ�뺯��
//Mat denoiseImage(const Mat& src) {
//	// ���ø�˹�˴�С�ͱ�׼��
//	int ksize = 5;    // ��˹�˵Ĵ�С
//	double sigma = 1.0;  // ��˹�˵ı�׼��
//
//	// ���ɸ�˹��
//	Mat gaussianKernel = generateGaussianKernel(ksize, sigma);
//
//	// Ӧ�ø�˹�˲�
//	Mat dst = applyGaussianFilter(src, gaussianKernel);
//
//	return dst;
//}
//
////---------------��ֵ��---------------
//
//// ͳ�ƻҶ�ֵ�ֲ�
//void calcGrayHist(const Mat& image, Mat& histImage, const string& outputPath) {
//	// 1. ͳ�ƻҶ�ֵƵ��
//	vector<int> histogram(256, 0);
//	for (int i = 0; i < image.rows; i++) {
//		for (int j = 0; j < image.cols; j++) {
//			histogram[image.at<uchar>(i, j)]++;
//		}
//	}
//
//	// 2. �ҳ����Ƶ��
//	int maxFreq = *max_element(histogram.begin(), histogram.end());
//
//	// 3. �������۵�ֱ��ͼͼ��
//	int hist_h = 600;    // ���Ӹ߶�ʹͼ�������
//	int hist_w = 800;    // ���ӿ��
//	int margin = 50;     // �߾�
//	int graph_h = hist_h - 2 * margin;  // ʵ��ͼ������߶�
//	int graph_w = hist_w - 2 * margin;  // ʵ��ͼ��������
//	int bin_w = graph_w / 256;          // ÿ���Ҷȼ��Ŀ��
//
//	// ������ɫ������ͼ��
//	histImage = Mat(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
//
//	// 4. ����������
//	line(histImage,
//		Point(margin, hist_h - margin),
//		Point(hist_w - margin, hist_h - margin),
//		Scalar(0, 0, 0), 2);  // X��
//	line(histImage,
//		Point(margin, hist_h - margin),
//		Point(margin, margin),
//		Scalar(0, 0, 0), 2);  // Y��
//
//	// 5. ����ֱ��ͼ��״
//	for (int i = 0; i < 256; i++) {
//		int height = cvRound((double)histogram[i] * graph_h / maxFreq);
//		rectangle(histImage,
//			Point(margin + i * bin_w, hist_h - margin),
//			Point(margin + (i + 1) * bin_w - 1, hist_h - margin - height),
//			Scalar(100, 100, 250),  // ����ɫ���
//			FILLED);
//		rectangle(histImage,
//			Point(margin + i * bin_w, hist_h - margin),
//			Point(margin + (i + 1) * bin_w - 1, hist_h - margin - height),
//			Scalar(0, 0, 0),  // ��ɫ�߿�
//			1);
//	}
//
//	// 6. ��ӿ̶Ⱥͱ�ǩ
//	// X��̶�
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
//	// Y��̶�
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
//	// 7. ��ӱ�������ǩ
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
//	// 8. ��ͳ�ƽ��д���ļ�
//	ofstream outFile(outputPath + "/histogram_stats.txt");
//	if (outFile.is_open()) {
//		outFile << "�Ҷ�ֱ��ͼͳ�ƽ��\n" << endl;
//		outFile << "�Ҷ�ֵ\tƵ��\t��һ��Ƶ��\n" << endl;
//
//		// ����һЩͳ����
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
//		// ���ͳ����Ϣ
//		outFile << "\nͳ����Ϣ��" << endl;
//		outFile << "����������" << totalPixels << endl;
//		outFile << "���Ƶ�ʣ�" << maxFreq << endl;
//		outFile << "ƽ���Ҷ�ֵ��" << sum / totalPixels << endl;
//
//		outFile.close();
//	}
//}
//
//// �Զ�����ֵ�������������ر�����ǰ���͹�������
//Mat threeValuedThreshold(const Mat& src, int lowThresh, int highThresh) {
//	Mat result = src.clone();
//	for (int y = 0; y < src.rows; ++y) {
//		for (int x = 0; x < src.cols; ++x) {
//			int pixel = src.at<uchar>(y, x);
//			if (pixel < lowThresh) {
//				result.at<uchar>(y, x) = 0;    // ����
//			}
//			else if (pixel > highThresh) {
//				result.at<uchar>(y, x) = 255;  // ǰ��
//			}
//			else {
//				result.at<uchar>(y, x) = 127;  // ��������
//			}
//		}
//	}
//	return result;
//}
//
////---------------�ݶȻ�---------------
//
//// ����ͼ���ݶȣ�ʹ��Prewitt����
//Mat computeGradient(const Mat& src) {
//	// ȷ��ͼ���ǻҶ�ͼ
//	CV_Assert(src.channels() == 1);
//
//	int rows = src.rows;
//	int cols = src.cols;
//
//	// Prewitt���ӵ�ˮƽ�ʹ�ֱ�����
//	int sobelX[3][3] = { {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1} };
//	int sobelY[3][3] = { {-1, -1, -1}, {0, 0, 0}, {1, 1, 1} };
//
//	Mat grad_x = Mat::zeros(src.size(), CV_32F);
//	Mat grad_y = Mat::zeros(src.size(), CV_32F);
//	Mat grad = Mat::zeros(src.size(), CV_32F);
//
//	// ��ͼ����� Prewitt ���
//	for (int i = 1; i < rows - 1; ++i) {
//		for (int j = 1; j < cols - 1; ++j) {
//			float gradX = 0.0f;
//			float gradY = 0.0f;
//
//			// �������ˮƽ�ݶ� grad_x
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
//			// �����ݶȵķ���
//			grad.at<float>(i, j) = sqrt(gradX * gradX + gradY * gradY);
//		}
//	}
//
//	// ��������ת��Ϊ 8 λͼ��
//	Mat grad_8u;
//	grad.convertTo(grad_8u, CV_8U);
//
//	return grad_8u;
//}
//
////---------------ȥ��С����---------------
//
//// ���ͼ���ڵ���ͨ����
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
//		// �����ĸ�����
//		for (int dx = -1; dx <= 1; ++dx) {
//			for (int dy = -1; dy <= 1; ++dy) {
//				if (dx == 0 && dy == 0) continue;
//
//				int nx = p.x + dx, ny = p.y + dy;
//
//				// ���߽�����
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
//// ����������������򵥵ؼ��������ڵ����ص�������
//int contourArea(const vector<Point>& contour)
//{
//	int n = contour.size();
//	if (n < 3) {
//		// ����һ����Ч�Ķ���Σ��޷��������
//		return 0;
//	}
//
//	double area = 0.0;
//	for (int i = 0; i < n; i++) {
//		// ʹ��Shoelace��ʽ�������
//		int j = (i + 1) % n;  // ��һ�㣬���һ�����һ���ǵ�һ��
//		area += contour[i].x * contour[j].y;
//		area -= contour[j].x * contour[i].y;
//	}
//	area = fabs(area) / 2.0;  // �������ֵ������2
//
//	return area;
//
//}
//
//// �Ƴ�С����
//Mat removeSmallRegions(const Mat& input, int minArea) {
//	// ����һ�����ڱ���ѷ���λ�õ�Mat
//	Mat visited = Mat::zeros(input.size(), CV_8U);
//	vector<vector<Point>> contours;
//
//
//	// ����ÿ�����أ�Ѱ����ͨ����
//	for (int y = 0; y < input.rows; ++y) {
//		for (int x = 0; x < input.cols; ++x) {
//			if (input.at<uchar>(y, x) == 255 && visited.at<uchar>(y, x) == 0) {
//				// ������ص���ǰ����δ�����ʹ���ִ�к鷺��䣨Flood Fill��
//				vector<Point> contour;
//				floodFill(input, visited, x, y, contour);
//
//				// ��������������
//				int area = contourArea(contour);
//
//				// ������������ֵ������������
//				if (area > minArea) {
//					contours.push_back(contour);
//				}
//			}
//		}
//	}
//
//	// ���������Ķ�ֵͼ��
//	Mat cleanedBinary = Mat::zeros(input.size(), CV_8U);
//
//	// ��������ֵ���������Ƶ�������ͼ����
//	for (const auto& contour : contours) {
//		for (const Point& pt : contour) {
//			cleanedBinary.at<uchar>(pt.y, pt.x) = 255;
//		}
//	}
//
//	return cleanedBinary;
//}
//
////---------------��ˮ���㷨---------------
//
//// ��ͼ��ת��Ϊ�Ҷ�ͼ
//void convertGrayToBGRA(const Mat& original, Mat& output) {
//	// ȷ��ԭʼͼ���ǻҶ�ͼ
//	if (original.empty() || original.channels() != 1) {
//		std::cerr << "Input image must be a grayscale image!" << std::endl;
//		return;
//	}
//
//	// ����������ͼ���С��ͬ��BGRAͼ��
//	output.create(original.rows, original.cols, CV_8UC4);
//
//	// ����ÿ�����أ����Ҷ�ֵӳ�䵽BGRA��ʽ
//	for (int y = 0; y < original.rows; ++y) {
//		for (int x = 0; x < original.cols; ++x) {
//			// ��ȡ�Ҷ�ֵ
//			uchar gray = original.at<uchar>(y, x);
//
//			// ����BGRAͼ���ÿ��ͨ��
//			output.at<Vec4b>(y, x)[0] = gray;  // Blue channel
//			output.at<Vec4b>(y, x)[1] = gray;  // Green channel
//			output.at<Vec4b>(y, x)[2] = gray;  // Red channel
//			output.at<Vec4b>(y, x)[3] = 255;  // Alpha channel (fully opaque)
//		}
//	}
//}
//
//// ����һ��������ʵ����ͨ������
//int getConnectedComponents(const Mat& binaryImage, Mat& labels, Mat& stats, Mat& centroids) {
//	// ��ȡͼ��ĳߴ�
//	int rows = binaryImage.rows;
//	int cols = binaryImage.cols;
//
//	// ��ʼ��labels��stats��centroids
//	labels = Mat::zeros(binaryImage.size(), CV_32S);  // ÿ�����صı�ǩ
//	stats = Mat::zeros(1, 5, CV_32S);  // ÿ����ͨ�����ͳ����Ϣ
//	centroids = Mat::zeros(1, 2, CV_32F);  // ÿ����ͨ���������
//
//	// ���ڱ����ͨ����ı�ǩ
//	int labelCount = 0;
//
//	// �������飬8��ͨ���������Һ��ĸ��Խ��߷���
//	int dx[8] = { -1, 1, 0, 0, -1, -1, 1, 1 };
//	int dy[8] = { 0, 0, -1, 1, -1, 1, -1, 1 };
//
//	// ���ڴ洢�����ͳ����Ϣ
//	vector<Rect> boundingBoxes;
//	vector<Point> regionCentroids;
//	vector<int> pixelCounts;
//
//	// ����������� (DFS)
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
//			// ��������8������
//			for (int i = 0; i < 8; i++) {
//				int nx = p.x + dx[i];
//				int ny = p.y + dy[i];
//
//				// ���߽��ǰ������
//				if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && labels.at<int>(nx, ny) == 0 && binaryImage.at<uchar>(nx, ny) == 255) {
//					labels.at<int>(nx, ny) = label;
//					stack.push_back(Point(nx, ny));
//					regionPixels.push_back(Point(nx, ny));
//				}
//			}
//		}
//		};
//
//	// ��������ͼ�񣬲�����ͨ����
//	for (int i = 0; i < rows; i++) {
//		for (int j = 0; j < cols; j++) {
//			if (binaryImage.at<uchar>(i, j) == 255 && labels.at<int>(i, j) == 0) {
//				// �ҵ�һ���µ���ͨ����
//				labelCount++;
//				vector<Point> regionPixels;
//				dfs(i, j, labelCount, regionPixels);
//
//				// ������ͨ����ı߽��
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
//				// �洢����ͳ����Ϣ
//				boundingBoxes.push_back(Rect(minY, minX, maxY - minY + 1, maxX - minX + 1));
//				pixelCounts.push_back(pixelCount);
//				regionCentroids.push_back(Point2f(sumY / pixelCount, sumX / pixelCount));
//			}
//		}
//	}
//
//	// ��ͳ����Ϣ����stats��centroids
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
//// ��Ӻ�������ȷ��������ɫ
//Vec4b determineRegionColor(const Mat& Region, const Mat& thresholdedImage,
//	const Vec4b& darkGreen, const Vec4b& lightGreen, const Vec4b& yellow) {
//	// ͳ�Ƹ�����ĻҶ�ֵ�ֲ�
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
//	// ȷ��������ɫ
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
//// ��Ӻ������ڻ�ȡ����ɫ�߷ָ������
//vector<Mat> getRegionsFromRedLines(const Mat& overlay) {
//	// ������ֵͼ���ʾ��ɫ�߽�
//	Mat redMask = Mat::zeros(overlay.size(), CV_8UC1);
//	for (int i = 0; i < overlay.rows; i++) {
//		for (int j = 0; j < overlay.cols; j++) {
//			Vec4b pixel = overlay.at<Vec4b>(i, j);
//			// ����ɫ���� (BGR: 0,0,255)
//			if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 255) {
//				redMask.at<uchar>(i, j) = 255;
//			}
//		}
//	}
//
//	// �������ڱ�ǵ�ͼ��
//	Mat labels, stats, centroids;
//	Mat invRedMask = 255 - redMask; // ��ת���룬ʹ����Ϊ��ɫ���߽�Ϊ��ɫ
//	int nLabels = connectedComponentsWithStats(invRedMask, labels, stats, centroids, 8);
//
//	// �洢ÿ�����������
//	vector<Mat> regions;
//	for (int label = 1; label < nLabels; label++) {
//		Mat regionMask = (labels == label);
//		regions.push_back(regionMask);
//	}
//
//	return regions;
//}
//
//// �޸ĺ����ɫȷ������
//void colorRegionsBasedOnContent(Mat& overlay, const Mat& thresholdedImage,
//	const Vec4b& darkGreen, const Vec4b& lightGreen,
//	const Vec4b& yellow, const Vec4b& red) {
//	// ��ȡ���б����߷ָ������
//	vector<Mat> regions = getRegionsFromRedLines(overlay);
//
//	// ����ÿ������
//	for (const Mat& region : regions) {
//		// ͳ�Ƹ�����ĻҶ�ֵ�ֲ�
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
//		// ȷ��������ɫ
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
//		// Ӧ����ɫ������
//		for (int i = 0; i < region.rows; i++) {
//			for (int j = 0; j < region.cols; j++) {
//				if (region.at<uchar>(i, j)) {
//					Vec4b& pixel = overlay.at<Vec4b>(i, j);
//					// ֻ�ԷǺ�ɫ�߽�����Ӧ������ɫ
//					if (!(pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 255)) {
//						overlay.at<Vec4b>(i, j) = regionColor;
//					}
//				}
//			}
//		}
//	}
//}
//
//// �ֶ�ʵ�ֵķ�ˮ���㷨
//Mat manualWatershed(const Mat& cleaned_grad, const Mat& thresholdedImage, const Mat& original) {
//
//	Mat output;
//	convertGrayToBGRA(original, output);
//
//	// �������߿���ݶ�ͼ����
//	Mat bordered_grad = cleaned_grad.clone();
//	rectangle(bordered_grad, Point(0, 0), Point(bordered_grad.cols - 1, bordered_grad.rows - 1), Scalar(255), 1);
//
//	// �������ֲ�
//	Mat overlay = Mat::zeros(output.size(), CV_8UC4);
//
//	// ������ɫ
//	Vec4b darkGreen(50, 150, 50, 64);
//	Vec4b lightGreen(100, 255, 100, 64);
//	Vec4b yellow(0, 255, 255, 64);
//	Vec4b red(0, 0, 255, 255);
//
//	// ��ȡ��ͨ����
//	Mat labels, stats, centroids;
//	int nLabels = getConnectedComponents(255 - bordered_grad, labels, stats, centroids);
//
//	// ������ˮ����ͼ
//	Mat watershedMarkers = Mat::zeros(labels.size(), CV_32S);
//	Mat processed = Mat::zeros(labels.size(), CV_8U);
//
//	// 8������
//	int dx[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
//	int dy[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
//
//	// ����ÿ����ͨ����
//	for (int label = 1; label < nLabels; label++) {
//		// Ϊ��ǰ���򴴽�����
//		queue<Point> q;
//		Mat currentRegion = (labels == label);
//
//		// �ҵ��������ĵ���Ϊ���ӵ�
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
//		// �����ӵ�������
//		q.push(seed);
//		watershedMarkers.at<int>(seed) = label;
//		processed.at<uchar>(seed) = 255;
//
//		// ��ˮ�����Ź���
//		while (!q.empty()) {
//			Point current = q.front();
//			q.pop();
//
//			// ���8����
//			for (int k = 0; k < 8; k++) {
//				int newX = current.x + dx[k];
//				int newY = current.y + dy[k];
//
//				// ���߽�
//				if (newX >= 0 && newX < labels.cols &&
//					newY >= 0 && newY < labels.rows) {
//
//					// ����õ�δ������
//					if (!processed.at<uchar>(newY, newX)) {
//						// ����Ƿ���������������������
//						bool isBoundary = false;
//						for (int m = 0; m < 8; m++) {
//							int checkX = newX + dx[m];
//							int checkY = newY + dy[m];
//
//							if (checkX >= 0 && checkX < labels.cols &&
//								checkY >= 0 && checkY < labels.rows) {
//								// ����ڵ����������Ѵ���������ǰ��Ϊ�߽�
//								if (processed.at<uchar>(checkY, checkX) == 255 &&
//									watershedMarkers.at<int>(checkY, checkX) != label) {
//									isBoundary = true;
//									break;
//								}
//							}
//						}
//
//						if (isBoundary) {
//							// ���Ϊ�߽��
//							overlay.at<Vec4b>(newY, newX) = red;
//							processed.at<uchar>(newY, newX) = 255;
//						}
//						else {
//							// ������뵱ǰ����
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
//	// ��Ӻ�ɫ�߽�
//	for (int i = 0; i < bordered_grad.rows; i++) {
//		for (int j = 0; j < bordered_grad.cols; j++) {
//			if (bordered_grad.at<uchar>(i, j) == 255) {
//				overlay.at<Vec4b>(i, j) = red;
//			}
//		}
//	}
//
//	// ����ɷ�ˮ�����ź󣬴���������ɫ
//
//	// �ڴ���ÿ������ʱ
//	colorRegionsBasedOnContent(overlay, thresholdedImage, darkGreen, lightGreen, yellow, red);
//
//	// ������ֲ���ԭͼ
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
//	// ��ȡͼ��ת��Ϊ�Ҷ�ͼ��
//	Mat src = imread("�����ƺ.tif", IMREAD_GRAYSCALE);
//	if (src.empty()) {
//		cout << "�޷�����ͼ��!" << endl;
//		return -1;
//	}
//
//	// ʹ��˫���β�ֵ��ͼ��Ŵ�4��
//	double scale = 4.0;
//	src = bicubicinterpolation(src, scale);
//
//	// �������Ŀ¼
//	system("mkdir res_g");
//
//	// ����res�ļ���
//	clearDirectory("res_g");
//
//	// ����ԭͼ���ѷŴ�
//	imwrite("res_g/1_original_scaled.png", src);
//
//	// ͼ��ȥ��
//	Mat denoisedImage = denoiseImage(src);
//	imwrite("res_g/2_denoised.png", denoisedImage);
//
//	// ͳ�ƻҶ�ֱ��ͼ
//	Mat hist;
//	calcGrayHist(denoisedImage, hist, "data");
//	imwrite("data/3_gray_hist.png", hist);
//
//	// ͼ����ֵ��
//	Mat thresholdedImage = threeValuedThreshold(denoisedImage, 100, 200);
//	imwrite("res_g/3_thresholded.png", thresholdedImage);
//
//	// �����ݶ�ͼ
//	Mat grad = computeGradient(thresholdedImage);
//	imwrite("res_g/4_gradient.png", grad);
//
//	// �Ƴ�С�������
//	// minArea������Ҫ����ʵ���������
//	Mat cleaned_grad = removeSmallRegions(grad, 30);
//	imwrite("res_g/5_gradient_cleaned.png", cleaned_grad);
//
//	// ���Ƴ�С�����������
//	Mat coloredImage = manualWatershed(cleaned_grad, thresholdedImage, src);
//	imwrite("res_g/6_colored.png", coloredImage);
//
//	return 0;
//}