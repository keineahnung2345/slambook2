#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

string image_file = "./distorted.png";   // 请确保路径正确

int main(int argc, char **argv) {

  // 本程序实现去畸变部分的代码。尽管我们可以调用OpenCV的去畸变，但自己实现一遍有助于理解。
  // 畸变参数
  double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
  // 内参
  double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

  cv::Mat image = cv::imread(image_file, 0);   // 图像是灰度图，CV_8UC1
  int rows = image.rows, cols = image.cols;
  cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1);   // 去畸变以后的图

  // 计算去畸变后图像的内容
  // u是橫向，v是縱向
  for (int v = 0; v < rows; v++) {
    for (int u = 0; u < cols; u++) {
      // 按照公式，计算点(u,v)对应到畸变图像中的坐标(u_distorted, v_distorted)
      /**
       * 由歸一化像平面到圖像坐標
       * u = fx * x_normalized + cx
       * v = fy * y_normalized + cy
       * 由圖像到歸一化像平面坐標
       * x_normalized = (u - cx)/fx
       * y_normalized = (v - cy)/fy
       **/
      /**
       * 對應關係：
       * 橫向：cols, u, cx, fx
       * 縱向：rows, v, cy, fy
       **/
      double x = (u - cx) / fx, y = (v - cy) / fy;
      // 在歸一化像平面上做畸變
      double r = sqrt(x * x + y * y);
      /**
       * 徑向畸變:x_distorted = x(1 + k1 * r^2 + k2* r^4)
       * 徑向畸變:y_distorted = y(1 + k1 * r^2 + k2* r^4)
       * 切向畸變:x_distorted = x + 2*p1*x*y + p2(r^2+2*x^2)
       * 切向畸變:y_distorted = y + 2*p2*x*y + p1(r^2+2*y^2)
       **/
      double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
      double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
      /**
       * (X,Y,Z):相機坐標系下的三維點坐標
       * (u,v):圖像上的坐標
       * [u]    1  [fx 0 cx] [X]
       * [v] = --- [0 fy cy] [Y]
       * [1]    Z  [0  0  1] [Z]
       * 如果X,Y已經對深度Z歸一化，那麼：
       * u_distorted = fx * x_distorted + cx
       * v_distorted = fy * y_distorted + cy
       **/
      // 做完畸變後由歸一化像平面到圖像坐標
      double u_distorted = fx * x_distorted + cx;
      double v_distorted = fy * y_distorted + cy;
      /**
       * 整體思想如下：
       * 現在有一張畸變的圖，我們希望把其中的像素值搬到正確（未畸變）的位置上
       * 因此我們需要知道畸變圖上的哪個像素點會對應到還原圖上的給定像素點（v,u）
       * 這邊的做法是先從還原圖上的點映射到歸一化像平面上，然後對它做畸變，得到畸變圖上的像素坐標
       * 如此一來，我們就得到了還原圖與畸變圖的像素對應
       * 最後再為還原圖上的像素點（u,v），尋找它在畸變圖上對應像素點坐標，以該處的顏色來填補
       **/

      // 赋值 (最近邻插值)
      if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {
        image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
      } else {
        image_undistort.at<uchar>(v, u) = 0;
      }
    }
  }

  // 画图去畸变后图像
  cv::imshow("distorted", image);
  cv::imshow("undistorted", image_undistort);
  cv::waitKey();
  return 0;
}
