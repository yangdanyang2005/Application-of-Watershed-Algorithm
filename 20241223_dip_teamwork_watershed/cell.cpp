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

//--------------------------------��ֵ��------------------------------------

//ת��Ϊ�Ҷ�ͼ��
void cvtColorToGray(const cv::Mat& image, cv::Mat& gray) {
    // �������ͼ���Ƿ�Ϊ��ͨ����ɫͼ��
    if (image.channels() != 3) {
        std::cerr << "Input image is not a 3-channel color image!" << std::endl;
        return;
    }

    // ����һ����ͨ���ĻҶ�ͼ�񣬴�С������ͼ����ͬ
    gray.create(image.rows, image.cols, CV_8UC1);

    // ����ÿ�����أ�����BGRת�Ҷȹ�ʽ����Ҷ�ֵ
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            // ��ȡ��ǰ���ص�BGRֵ
            cv::Vec3b bgr = image.at<cv::Vec3b>(i, j);

            // ʹ��BGR���Ҷȵ�ת����ʽ
            unsigned char grayValue = static_cast<unsigned char>(
                0.299 * bgr[2] + 0.587 * bgr[1] + 0.114 * bgr[0]
                );

            // ��������ĻҶ�ֵ��ֵ�����ͼ�����Ӧλ��
            gray.at<uchar>(i, j) = grayValue;
        }
    }
}

// ���� Otsu ��ֵ
double otsuThreshold(const Mat& gray) {
    // ����ͼ��ֱ��ͼ
    vector<int> hist(256, 0);
    for (int y = 0; y < gray.rows; ++y) {
        for (int x = 0; x < gray.cols; ++x) {
            hist[gray.at<uchar>(y, x)]++;
        }
    }

    // ��������
    int total = gray.rows * gray.cols;

    // ���� Otsu ��ֵ
    double sum = 0;
    for (int i = 0; i < 256; ++i) {
        sum += i * hist[i];
    }

    double sumB = 0, wB = 0, wF = 0;
    double varMax = 0, threshold = 0;

    for (int t = 0; t < 256; ++t) {
        wB += hist[t];  // ����������Ȩ��
        if (wB == 0) continue;
        wF = total - wB;  // ǰ��������Ȩ��
        if (wF == 0) break;

        sumB += t * hist[t];
        double mB = sumB / wB;  // ������ƽ���Ҷ�
        double mF = (sum - sumB) / wF;  // ǰ����ƽ���Ҷ�

        double varBetween = wB * wF * pow(mB - mF, 2);  // ��䷽��
        if (varBetween > varMax) {
            varMax = varBetween;
            threshold = t;
        }
    }

    return threshold;
}

// �ֶ�ʵ�ֶ�ֵ������ת��ʹ�� Otsu ��ֵ��
void thresholdBinaryInvOtsu(const Mat& gray, Mat& binary) {
    // ���� Otsu ��ֵ
    double otsuThresh = otsuThreshold(gray);

    // ������ֵͼ��Ӧ�÷�ת��ֵ����
    binary.create(gray.size(), CV_8UC1);
    for (int y = 0; y < gray.rows; ++y) {
        for (int x = 0; x < gray.cols; ++x) {
            // ʹ�� Otsu ��ֵ���з�ת��ֵ��
            binary.at<uchar>(y, x) = (gray.at<uchar>(y, x) > otsuThresh) ? 0 : 255;
        }
    }
}

//-----------------------------------ȥ��С�������----------------------------------
// �������ͼ���ڵ���ͨ����
void floodFill(const Mat& binary, Mat& visited, int x, int y, vector<Point>& contour) {
    int rows = binary.rows, cols = binary.cols;
    queue<Point> q;
    q.push(Point(x, y));
    visited.at<uchar>(y, x) = 1;

    while (!q.empty()) {
        Point p = q.front();
        q.pop();
        contour.push_back(p);

        // �����ĸ�����
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                if (dx == 0 && dy == 0) continue;

                int nx = p.x + dx, ny = p.y + dy;

                // ���߽�����
                if (nx >= 0 && nx < cols && ny >= 0 && ny < rows &&
                    binary.at<uchar>(ny, nx) == 255 && visited.at<uchar>(ny, nx) == 0) {
                    visited.at<uchar>(ny, nx) = 1;
                    q.push(Point(nx, ny));
                }
            }
        }
    }
}

// ����������������򵥵ؼ��������ڵ����ص�������
int contourArea(const vector<Point>& contour) 
{
    int n = contour.size();
    if (n < 3) {
        // ����һ����Ч�Ķ���Σ��޷��������
        return 0;
    }

    double area = 0.0;
    for (int i = 0; i < n; i++) {
        // ʹ��Shoelace��ʽ�������
        int j = (i + 1) % n;  // ��һ�㣬���һ�����һ���ǵ�һ��
        area += contour[i].x * contour[j].y;
        area -= contour[j].x * contour[i].y;
    }
    area = fabs(area) / 2.0;  // �������ֵ������2

    return area;
    
}

//------------------------------��̬ѧ����ƽ����Ե------------------------------
// �Զ��帯ʴ����
void erodeImage(const Mat& input, Mat& output, const Mat& kernel) {
    int kernelRows = kernel.rows;
    int kernelCols = kernel.cols;
    int kRowOffset = kernelRows / 2;
    int kColOffset = kernelCols / 2;

    output = input.clone(); // ��������ͼ�����ͼ��

    for (int i = kRowOffset; i < input.rows - kRowOffset; ++i) {
        for (int j = kColOffset; j < input.cols - kColOffset; ++j) {
            bool minPixel = true;
            // �����ṹԪ�ص�ÿһ��λ��
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

// �Զ������Ͳ���
void dilateImage(const Mat& input, Mat& output, const Mat& kernel) {
    int kernelRows = kernel.rows;
    int kernelCols = kernel.cols;
    int kRowOffset = kernelRows / 2;
    int kColOffset = kernelCols / 2;

    output = input.clone(); // ��������ͼ�����ͼ��

    for (int i = kRowOffset; i < input.rows - kRowOffset; ++i) {
        for (int j = kColOffset; j < input.cols - kColOffset; ++j) {
            bool maxPixel = false;
            // �����ṹԪ�ص�ÿһ��λ��
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

// �Զ��忪���㣨�ȸ�ʴ�����ͣ�
void openImage(const Mat& input, Mat& output, const Mat& kernel) {
    Mat eroded;
    erodeImage(input, eroded, kernel);  // �ȸ�ʴ
    dilateImage(eroded, output, kernel); // ������
}

// �Զ�������㣨�������ٸ�ʴ��
void closeImage(const Mat& input, Mat& output, const Mat& kernel) {
    Mat dilated;
    dilateImage(input, dilated, kernel);  // ������
    erodeImage(dilated, output, kernel);  // ��ʴ
}

// �����ṹԪ��
Mat getStructureElement(int shape, Size ksize) {
    Mat kernel = Mat::zeros(ksize, CV_8UC1);
    int centerX = ksize.width / 2;
    int centerY = ksize.height / 2;

    if (shape == MORPH_ELLIPSE) {
        // ������Բ�νṹԪ��
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

//��ֵ�˲�
void medianBlurCustom(const Mat& input, Mat& output, int ksize) {
    // ͼ���С�ʹ��ڴ�С
    int rows = input.rows;
    int cols = input.cols;
    int half_ksize = ksize / 2;

    // ���ͼ���ʼ��Ϊ����ͼ��ȷ������ƥ��
    output = input.clone();

    // ����ͼ���е�ÿ�����أ�������Ե
    for (int i = half_ksize; i < rows - half_ksize; ++i) {
        for (int j = half_ksize; j < cols - half_ksize; ++j) {
            // �ռ� 3x3 �����ڵ�����ֵ
            std::vector<uchar> window;
            for (int di = -half_ksize; di <= half_ksize; ++di) {
                for (int dj = -half_ksize; dj <= half_ksize; ++dj) {
                    window.push_back(input.at<uchar>(i + di, j + dj));
                }
            }

            // �Դ����ڵ�����ֵ��������
            std::sort(window.begin(), window.end());

            // ȡ��ֵ
            output.at<uchar>(i, j) = window[window.size() / 2]; // ��ֵ���������м�Ԫ��
        }
    }
}

//------------------------����任���ֶ�ʵ��----------------------------
//����任
void distanceTransform(const Mat& input, Mat& output) {
    // ȷ�������Ƕ�ֵͼ��
    CV_Assert(input.type() == CV_8UC1);

    int rows = input.rows;
    int cols = input.cols;

    // ��ʼ���������
    output = Mat(rows, cols, CV_32FC1, Scalar(FLT_MAX));

    // ��һ��ɨ�� - �����ϵ�����
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (input.at<uchar>(i, j) > 0) { // �����ǰ������
                float min_dist = FLT_MAX;

                // ������Ϸ�������
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
            else { // ����Ǳ�������
                output.at<float>(i, j) = 0;
            }
        }
    }

    // �ڶ���ɨ�� - �����µ�����
    for (int i = rows - 1; i >= 0; i--) {
        for (int j = cols - 1; j >= 0; j--) {
            if (input.at<uchar>(i, j) > 0) { // ֻ����ǰ������
                float min_dist = output.at<float>(i, j);

                // ������·�������
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

// ˫���˲�����
void BilateralFilter(const Mat& src, Mat& dst, int d, double sigmaColor, double sigmaSpace) {
    CV_Assert(src.channels() == 1); // ȷ������Ϊ��ͨ���Ҷ�ͼ��

    // ��ʼ�����ͼ��
    dst = Mat::zeros(src.size(), src.type());

    // �˲����ڵİ뾶
    int radius = d / 2;

    // Ԥ����ռ��˹Ȩ��
    vector<vector<double>> gaussianSpace(d, vector<double>(d));
    double spaceCoeff = -0.5 / (sigmaSpace * sigmaSpace);
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            gaussianSpace[i + radius][j + radius] = exp((i * i + j * j) * spaceCoeff);
        }
    }

    // ����ÿ������
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            double sumWeights = 0.0;
            double sumFiltered = 0.0;

            // ��������
            for (int i = -radius; i <= radius; ++i) {
                for (int j = -radius; j <= radius; ++j) {
                    int neighborY = y + i;
                    int neighborX = x + j;

                    // �߽紦���������
                    neighborY = max(0, min(neighborY, src.rows - 1));
                    neighborX = max(0, min(neighborX, src.cols - 1));

                    // ��ǰ����ֵ����������ֵ
                    uchar centerVal = src.at<uchar>(y, x);
                    uchar neighborVal = src.at<uchar>(neighborY, neighborX);

                    // ������ɫ��˹Ȩ��
                    double intensityDiff = neighborVal - centerVal;
                    double gaussianColor = exp(-0.5 * (intensityDiff * intensityDiff) / (sigmaColor * sigmaColor));

                    // �����Ȩ��
                    double weight = gaussianSpace[i + radius][j + radius] * gaussianColor;

                    // ��Ȩ�ۼ�
                    sumWeights += weight;
                    sumFiltered += weight * neighborVal;
                }
            }

            // ��һ������������ֵ
            dst.at<uchar>(y, x) = static_cast<uchar>(sumFiltered / sumWeights);
        }
    }
}

// �����ļ��к�����Windows�汾��
void clearDirectory(const string& path) {
    string cmd = "del /Q " + path + "\\*.*";
    system(cmd.c_str());
}

// �������ص�ṹ
struct Pixel {
    int x, y;     // ����
    float height; // �߶�ֵ���Ҷ�ֵ��

    Pixel(int x_, int y_, float height_) : x(x_), y(y_), height(height_) {}

    // �������ȶ��еıȽ�
    bool operator<(const Pixel& other) const {
        return height > other.height; // ��С��
    }
};

// ϸ��ͳ�ƽṹ��
struct CellStats {
    int total_cells;
    double avg_area;
    double min_area;
    double max_area;
    double std_dev;
    vector<double> cell_areas;
};

// ��ˮ���㷨ʵ��
void customWatershed(const Mat& image, Mat& markers, const Mat& cleanedBinary, const Mat& dist_inv, float minDist = 0.005, int windowSize = 14) {
    int label = 1;// ���ֵ��1��ʼ
    int halfWindow = windowSize / 2;

    /******************* ��һ�׶Σ���ǵ��� *******************/
    // �ھ���任ͼ���������Ѱ�Ҿֲ���Сֵ��Ϊ��ʼ��ǵ�
    for (int i = halfWindow; i < dist_inv.rows - halfWindow; i++) {
        for (int j = halfWindow; j < dist_inv.cols - halfWindow; j++) {

            // ��ȫ��ʩ��������������ȥ��֮��Ķ�ֵ��ͼ���в���ϸ���ĺ�ɫ���򣩣�
            // ֻ�ӡ�����任����õ���ͼƬ����ϸ�����ڵ�������Ѱ�Ҿֲ���Сֵ��
            if (cleanedBinary.at<uchar>(i, j) == 0) continue;

            // ��ȡ��ǰ��ľ���ֵ������任ͼ�����ػҶ�ֵ���Ǿ���ֵ
            float center = dist_inv.at<float>(i, j);

            // ��������ֵС����ֵ�����õĲ������ĵ㣬����ֲ���Сֵ����������
            if (center < minDist) continue;

            /******************* �ֲ���Сֵ���� *******************/
            bool is_min = true;
            // ��ָ����С�Ĵ����ڼ���Ƿ��Ǿֲ���Сֵ
            for (int di = -halfWindow; di <= halfWindow && is_min; di++) {
                for (int dj = -halfWindow; dj <= halfWindow; dj++) {
                    if (di == 0 && dj == 0) continue;
                    // ������ڱȵ�ǰ��center��С��ֵ�����Ǿֲ���Сֵ
                    if (dist_inv.at<float>(i + di, j + dj) < center) 
                    {
                        is_min = false;
                        break;
                    }
                }
            }

            /******************* ��ǵ������ *******************/
            if (is_min) {
                // �����Χ�Ƿ��Ѿ�����������ǵ�
                bool too_close = false;
                for (int di = -windowSize; di <= windowSize && !too_close; di++) {
                    for (int dj = -windowSize; dj <= windowSize; dj++) {
                        int ni = i + di;
                        int nj = j + dj;
                        // �����ָ����Χ�ڷ���������ǵ㣬��ǰ�㲻��Ϊ��ǵ�
                        if (ni >= 0 && ni < markers.rows &&
                            nj >= 0 && nj < markers.cols &&
                            markers.at<int>(ni, nj) > 0) {
                            too_close = true;
                            break;
                        }
                    }
                }

                // ���������������������Ϊ�µı�ǵ�
                if (!too_close) {
                    markers.at<int>(i, j) = label++;
                }
            }
        }
    }

    /******************* �ڶ��׶Σ���ˮ������ *******************/
    // ����8������������
    const int dx[] = { -1, 1, 0, 0, -1, -1, 1, 1 };  // 8����x����ƫ��
    const int dy[] = { 0, 0, -1, 1, -1, 1, -1, 1 };  // 8����y����ƫ��
    /*
    ������δ��붨����8������������
    ����ʾ��ͼ��
        P7  P2  P6
        P3  P0  P1
        P5  P4  P8
    ����P0���������أ�����P1-P8��8����������

    dx[]��dy[]���鶨���˴��������ص�8���������ص�����ƫ�ƣ�
    dx[] = {-1,  1,  0,  0, -1, -1,  1,  1}
    dy[] = { 0,  0, -1,  1, -1,  1, -1,  1}
    ��Ӧ��8�������ǣ�
    ����  dx  dy   ����
    0:   -1   0   ��
    1:    1   0   ��
    2:    0  -1   ��
    3:    0   1   ��
    4:   -1  -1   ����
    5:   -1   1   ����
    6:    1  -1   ����
    7:    1   1   ����

    ���ַ�ʽ��ֱ��д8��if������࣬ͨ�������������ܷ��ʵ�����8��������������ء�
    */

    /******************* ��ʼ�����ȶ��� *******************/
    // ʹ�����ȶ��н��л��ڸ߶ȵ���������
    priority_queue<Pixel> pq;

    // �����г�ʼ��ǵ�������ȶ���
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            if (markers.at<int>(i, j) > 0) {
                float height = image.at<uchar>(i, j);
                pq.push(Pixel(i, j, height));
            }
        }
    }

    /******************* ������������ *******************/
    while (!pq.empty()) {
        // ��ȡ��ǰ��͸߶ȵ�����
        Pixel current = pq.top();
        pq.pop();

        // ���8�����ÿ������
        // ����8������
        for (int k = 0; k < 8; k++) {
            int newX = current.x + dx[k]; // �����������ص�x����
            int newY = current.y + dy[k]; // �����������ص�y����
            // �õ���(newX, newY)����8�����������е�һ��

            // ����Ƿ���ͼ��߽���
            if (newX >= 0 && newX < markers.rows &&
                newY >= 0 && newY < markers.cols) {

                // ���δ���ʵ������
                if (markers.at<int>(newX, newY) == 0) {
                    // ��ȡ��ǰ����ı��ֵ��δ���ʹ������������Ϊ��ǰ���ֵ
                    int currentLabel = markers.at<int>(current.x, current.y);

                    /******************* ��ˮ���ж� *******************/
                    // ����Ƿ��Ƿ�ˮ��㣨�Ƿ��벻ͬ����������ڣ�
                    bool hasConflict = false;
                    for (int m = 0; m < 8; m++) {
                        int checkX = newX + dx[m];
                        int checkY = newY + dy[m];

                        if (checkX >= 0 && checkX < markers.rows &&
                            checkY >= 0 && checkY < markers.cols) {
                            int neighborLabel = markers.at<int>(checkX, checkY);
                            // ���������ڲ�ͬ�ı�ǣ���Ϊ��ˮ���
                            if (neighborLabel > 0 && neighborLabel != currentLabel) {
                                hasConflict = true;
                                break;
                            }
                        }
                    }

                    // �����ж�������б��
                    if (hasConflict) {
                        // ���Ϊ��ˮ���
                        markers.at<int>(newX, newY) = -1;
                    }
                    else {
                        // ������ǰ����ı�ǲ�������������
                        markers.at<int>(newX, newY) = currentLabel;
                        float newHeight = image.at<uchar>(newX, newY);
                        pq.push(Pixel(newX, newY, newHeight));
                    }
                }
            }
        }
    }
}

//�ϲ�С����
vector<int> mergeSmallRegions(Mat& markers, int minArea) {
    // 1. ͳ��ÿ���������Ĵ�С
    map<int, int> regionSizes;
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int label = markers.at<int>(i, j);
            if (label > 0) {
                regionSizes[label]++;
            }
        }
    }
    cout << "�����Сͳ�ƣ�" << endl;
    for (const auto& region : regionSizes) {
        cout << "���� " << region.first << " ��С: " << region.second << " ����" << endl;
    }

    // 2. �ҳ���Ҫ�ϲ���С����
    vector<int> smallRegions;
    for (const auto& region : regionSizes) {
        if (region.second < minArea) {
            smallRegions.push_back(region.first);
        }
    }
    cout << "\n��Ҫ�ϲ���С����" << endl;
    for (int label : smallRegions) {
        cout << "���� " << label << " (��С: " << regionSizes[label] << " ����)" << endl;
    }

    // 3. Ϊÿ��С�����ҵ���Ѻϲ�Ŀ��
    for (int smallLabel : smallRegions) {
        // �洢���С�������������λ��
        vector<Point> regionPixels;
        // �洢�ڽӵĴ������ǩ����߽�Ӵ�����
        map<int, int> neighborLabels;

        // �ҳ���������������غͱ߽���Ϣ
        for (int i = 0; i < markers.rows; i++) {
            for (int j = 0; j < markers.cols; j++) {
                if (markers.at<int>(i, j) == smallLabel) {
                    regionPixels.push_back(Point(j, i));

                    // ���8����
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

        // �ҳ��Ӵ����Ĵ�����
        int bestLabel = -1;
        int maxContact = 0;
        for (const auto& neighbor : neighborLabels) {
            if (neighbor.second > maxContact) {
                maxContact = neighbor.second;
                bestLabel = neighbor.first;
            }
        }

        // ����ҵ����ʵĴ�����ִ�кϲ�
        if (bestLabel != -1) {
            cout << "�ϲ������� " << smallLabel << " -> ���� " << bestLabel << endl;
            cout << "�Ӵ�������: " << maxContact << endl;
            for (const Point& p : regionPixels) {
                markers.at<int>(p.y, p.x) = bestLabel;
            }
        }
    }

    // 4. �ٴμ�鲢�ϲ�������С���򣨵�������
    bool needAnotherPass = true;
    int maxIterations = 3;  // ���Ƶ�������
    int iteration = 0;

    while (needAnotherPass && iteration < maxIterations) {
        needAnotherPass = false;
        iteration++;

        // ���������Сͳ��
        regionSizes.clear();
        for (int i = 0; i < markers.rows; i++) {
            for (int j = 0; j < markers.cols; j++) {
                int label = markers.at<int>(i, j);
                if (label > 0) {
                    regionSizes[label]++;
                }
            }
        }

        // ����ʣ���С����
        for (const auto& region : regionSizes) {
            if (region.second < 4000) {
                needAnotherPass = true;
                int currentLabel = region.first;

                // �Ը�С�����ÿ������
                for (int i = 0; i < markers.rows; i++) {
                    for (int j = 0; j < markers.cols; j++) {
                        if (markers.at<int>(i, j) == currentLabel) {
                            // �ڸ����������Ѱ����Ѻϲ�Ŀ��
                            map<int, int> neighborCounts;
                            int searchRadius = 3;  // ����������Χ

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

                            // �ҳ���Ƶ�����ھӱ�ǩ
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

    // ͳ�����յ�������Ϣ
    map<int, int> finalRegionSizes;
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int label = markers.at<int>(i, j);
            if (label > 0) {
                finalRegionSizes[label]++;
            }
        }
    }

    cout << "\n�ϲ��������ͳ�ƣ�" << endl;
    for (const auto& region : finalRegionSizes) {
        cout << "���� " << region.first << " ���մ�С: " << region.second << " ����" << endl;
    }

    return smallRegions;
}

// ��������
void DrawContours(Mat& img, const vector<vector<Point>>& contours, int contourIdx, const Scalar& color, int thickness) {
    if (contourIdx < 0 || contourIdx >= contours.size()) {
        cout << "��������������Χ!" << endl;
        return;
    }

    const vector<Point>& contour = contours[contourIdx];

    // ����ÿһ�����ڵ㲢��������
    for (size_t i = 0; i < contour.size(); i++) {
        Point p1 = contour[i];
        Point p2 = contour[(i + 1) % contour.size()]; // ����������β��

        // �����߶Σ�Bresenham �㷨��
        LineIterator it(img, p1, p2, 8); // ʹ�ð���������
        for (int j = 0; j < it.count; j++, ++it) {
            // ������ɫ
            if (thickness == 1) {
                img.at<Vec3b>(it.pos()) = Vec3b(color[0], color[1], color[2]);
            }
            else {
                // �����������
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

// ����ϸ����ͳ����Ϣ
CellStats analyzeCells(const Mat& segmented_binary, const Mat& original_image, Mat& numbered_result) {
    CellStats stats;

    // ��������
    vector<vector<Point>> cell_contours;
    findContours(segmented_binary, cell_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // ͳ�ƻ�����Ϣ
    stats.total_cells = cell_contours.size();
    double total_area = 0;
    stats.min_area = DBL_MAX;
    stats.max_area = 0;

    // ����ÿ��ϸ�������
    for (const auto& contour : cell_contours) {
        double area = contourArea(contour);
        stats.cell_areas.push_back(area);
        total_area += area;
        stats.min_area = min(stats.min_area, area);
        stats.max_area = max(stats.max_area, area);
    }

    // ����ƽ�����
    stats.avg_area = total_area / stats.total_cells;

    // �����׼��
    double variance = 0;
    for (double area : stats.cell_areas) {
        variance += pow(area - stats.avg_area, 2);
    }
    stats.std_dev = sqrt(variance / stats.total_cells);

    // ���ͳ�ƽ��
    cout << "\nϸ��ͳ�ƽ����" << endl;
    cout << "��ϸ������: " << stats.total_cells << endl;
    cout << "ƽ�����: " << stats.avg_area << " ����" << endl;
    cout << "��С���: " << stats.min_area << " ����" << endl;
    cout << "������: " << stats.max_area << " ����" << endl;
    cout << "�����׼��: " << stats.std_dev << " ����" << endl;

    for (int i = 0; i < cell_contours.size(); i++) {
        // �������������ĵ�
        Moments m = moments(cell_contours[i]);
        Point center(m.m10 / m.m00, m.m01 / m.m00);

        // ��������
        DrawContours(numbered_result, cell_contours, i, Scalar(0, 255, 0), 2);

        // ��ӱ��
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

// ��ӡͼ����Ϣ
void printMatInfo(const string& filename) {
    Mat img = imread("res//" + filename, IMREAD_UNCHANGED);
    if (img.empty()) {
        cout << "�޷���ȡͼ��: " << filename << endl;
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
        type = "��������(" + to_string(img.type()) + ")";
    }

    cout << "ͼ��: " << filename << endl;
    cout << "����: " << type << endl;
    cout << "�ߴ�: " << img.size() << endl;
    cout << "ͨ����: " << img.channels() << endl;
    cout << "-------------------" << endl;
}

// ���ͼ������
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

    cout << "\n���ͼ�����ͣ�" << endl;
    for (const auto& filename : filenames) {
        printMatInfo(filename);
    }
}

// �ֶ�ʵ����̬ѧ�����㣬���ڿ��ӻ�����֮��
void morphologicalOpen(Mat& image) {
    // ����3x3�ṹԪ��
    const int kernelSize = 3;
    const int offset = kernelSize / 2;

    // �Ƚ��и�ʴ
    Mat eroded = Mat::zeros(image.size(), CV_8U);
    for (int i = offset; i < image.rows - offset; i++) {
        for (int j = offset; j < image.cols - offset; j++) {
            bool isValid = true;
            // ���3x3����
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

    // �ٽ�������
    Mat dilated = Mat::zeros(image.size(), CV_8U);
    for (int i = offset; i < image.rows - offset; i++) {
        for (int j = offset; j < image.cols - offset; j++) {
            bool hasWhite = false;
            // ���3x3����
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

// ���ӻ�������
void visualizeResults(const Mat& image, const Mat& markers, const Mat& cleanedBinary,
    const vector<int>& smallRegions, Mat& result, Mat& watershed_lines, Mat& segmented) {
    result = image.clone();
    watershed_lines = Mat::zeros(markers.size(), CV_8U);
    segmented = cleanedBinary.clone();

    // ����һ���������洢�Ѻϲ��ı�ǩ��
    set<pair<int, int>> mergedPairs;

    // �ռ��ϲ���Ϣ
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

    // ���Ʒ�ˮ����
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

    // �Էָ���������̬ѧ������
    morphologicalOpen(segmented);
}

// ������
int main() {
    // �������Ŀ¼
    system("mkdir res");

    // ����res�ļ���
    clearDirectory("res");

    // 1. ��ȡͼ��
    Mat image = imread("cells.png");
    if (image.empty()) {
        cout << "�޷���ȡͼ��" << endl;
        return -1;
    }

    // 2. ת��Ϊ�Ҷ�ͼ
    Mat gray;
    cvtColorToGray(image, gray);

    // 3. ��ֵ�� (ȷ��ϸ���Ǻ�ɫ�������ǰ�ɫ)
    Mat binary;
    thresholdBinaryInvOtsu(gray, binary);

    // 4. ȥ��С�������
    
    // ����һ�����ڱ���ѷ���λ�õ�Mat
    Mat visited = Mat::zeros(binary.size(), CV_8U);
    vector<vector<Point>> contours;
    double minArea = 100;  // С����������ֵ

    // ����ÿ�����أ�Ѱ����ͨ����
    for (int y = 0; y < binary.rows; ++y) {
        for (int x = 0; x < binary.cols; ++x) {
            if (binary.at<uchar>(y, x) == 255 && visited.at<uchar>(y, x) == 0) {
                // ������ص���ǰ����δ�����ʹ���ִ�к鷺��䣨Flood Fill��
                vector<Point> contour;
                floodFill(binary, visited, x, y, contour);

                // ��������������
                int area = contourArea(contour);

                // ������������ֵ������������
                if (area > minArea) {
                    contours.push_back(contour);
                }
            }
        }
    }

    // ���������Ķ�ֵͼ��
    Mat cleanedBinary = Mat::zeros(binary.size(), CV_8U);

    // ��������ֵ���������Ƶ�������ͼ����
    for (const auto& contour : contours) {
        for (const Point& pt : contour) {
            cleanedBinary.at<uchar>(pt.y, pt.x) = 255;
        }
    }
    
    // 4.1 �Զ�ֵͼ�������̬ѧ������ƽ����Ե
    
    // ����ṹԪ��
    Mat kernel = getStructureElement(2, Size(3, 3));
    // ���п�����
    Mat smoothedBinary;
    openImage(cleanedBinary, smoothedBinary, kernel);
    // ���б�����
    closeImage(smoothedBinary, smoothedBinary, kernel);

    // 4.2 ʹ����ֵ�˲���һ��ƽ��
    medianBlurCustom(smoothedBinary, smoothedBinary, 3);

    // 5. ����任 (ʹ��ƽ����Ķ�ֵͼ��)
    Mat dist;
    distanceTransform(smoothedBinary, dist);
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);

    // ʹ�ýϴ�ĺ˽��и�˹ģ��
    GaussianBlur(dist, dist, Size(5, 5), 0);

    // 6. ����任ȡ��
    Mat dist_inv = 1- dist; // ����任ȡ���Ľ��
    Mat dist_inv_vis;// ����任ȡ���Ŀ��ӻ����
    dist_inv.convertTo(dist_inv_vis, CV_8U, 255.0);

    // �Ծ���任������б�Ե�����˲�
    Mat filtered_dist;
    BilateralFilter(dist_inv_vis, filtered_dist, 5, 50, 50);
    dist_inv_vis = filtered_dist;

    // 7. Ӧ�÷�ˮ���㷨
    Mat markers = Mat::zeros(dist_inv.size(), CV_32S);
    customWatershed(dist_inv_vis, markers, cleanedBinary, dist_inv, 0.005, 14);    /* �����������£� */    // �޸Ĳ�������Ӧƽ�����ͼ�� float minDist = 0.005;    // ������ˮ���ӵ�֮�����С���룬ֵԽ��ɸѡԽ�ϸ�Ĭ�ϲ���0.005    // ע��ͼ�������ֵ�ķ�Χ��0-1    // int windowSize = 14;     // �������ڴ�С��Ӱ���ǵ���ܶȣ�Ĭ�ϲ���14
    // 8. �ϲ�С����
    vector<int> smallRegions = mergeSmallRegions(markers, 4000);
   
    // 9. ���ӻ����
    Mat result = image.clone();
    Mat watershed_lines = Mat::zeros(markers.size(), CV_8U);
    Mat segmented = cleanedBinary.clone();

    visualizeResults(image, markers, cleanedBinary, smallRegions,
        result, watershed_lines, segmented);

    // 10. ����ϸ��ͳ�Ʒ�������
    Mat numbered_result = image.clone(); // ֱ��ʹ��ԭʼͼ��ĸ���
    CellStats cell_statistics = analyzeCells(segmented, image, numbered_result);

    // ���ϸ������ֲ�
    if (cell_statistics.total_cells > 0) {
        cout << "\nϸ������ֲ���" << endl;
        for (int i = 0; i < cell_statistics.total_cells; i++) {
            cout << "ϸ�� " << (i + 1) << " ���: " << cell_statistics.cell_areas[i] << " ����" << endl;
        }
    }

    // �ļ�����
    imwrite("res//01_original.png", image);                          // ԭʼͼ��
    imwrite("res//02_grayscale.png", gray);                         // �Ҷ�ͼ
    imwrite("res//03_binary.png", binary);                          // ��ֵͼ
    imwrite("res//04_cleaned_binary.png", cleanedBinary);           // ȥ��С�����Ķ�ֵͼ
    imwrite("res//05_smoothed_binary.png", smoothedBinary);         // ƽ����Ķ�ֵͼ
    imwrite("res//06_inverted_distance_transform.png", dist_inv_vis); // ����任ȡ��
    imwrite("res//07_watershed_lines.png", watershed_lines);         // ��ˮ����
    imwrite("res//08_watershed_result.png", result);                 // ��ˮ���㷨���
    imwrite("res//09_segmented_binary.png", segmented);             // �ָ��Ķ�ֵͼ
    imwrite("res//10_numbered_cells.png", numbered_result);// �������ŵĽ��ͼ

    // ���ͼ������
    // checkImageTypes();
    /*
    ���ͼ�����ͣ�
    ͼ��: 01_original.png
    ����: CV_8UC3
    �ߴ�: [1361 x 654]
    ͨ����: 3
    -------------------
    ͼ��: 02_grayscale.png
    ����: CV_8UC1
    �ߴ�: [1361 x 654]
    ͨ����: 1
    -------------------
    ͼ��: 03_binary.png
    ����: CV_8UC1
    �ߴ�: [1361 x 654]
    ͨ����: 1
    -------------------
    ͼ��: 04_cleaned_binary.png
    ����: CV_8UC1
    �ߴ�: [1361 x 654]
    ͨ����: 1
    -------------------
    ͼ��: 05_smoothed_binary.png
    ����: CV_8UC1
    �ߴ�: [1361 x 654]
    ͨ����: 1
    -------------------
    ͼ��: 06_inverted_distance_transform.png
    ����: CV_8UC1
    �ߴ�: [1361 x 654]
    ͨ����: 1
    -------------------
    ͼ��: 07_watershed_lines.png
    ����: CV_8UC1
    �ߴ�: [1361 x 654]
    ͨ����: 1
    -------------------
    ͼ��: 08_final_result.png
    ����: CV_8UC3
    �ߴ�: [1361 x 654]
    ͨ����: 3
    -------------------
    ͼ��: 09_segmented_binary.png
    ����: CV_8UC1
    �ߴ�: [1361 x 654]
    ͨ����: 1
    -------------------
    ͼ��: 10_numbered_cells.png
    ����: CV_8UC3
    �ߴ�: [1361 x 654]
    ͨ����: 3
    -------------------
    */

    return 0;
}