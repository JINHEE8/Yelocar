// Written by DsLiner 2021
// Ros & RealSense
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tf/transform_broadcaster.h>
#include <librealsense2/rs.hpp>
#include <ros/console.h>
#include <vector>

// TCP for python server
// http://www.linuxhowtos.org/C_C++/socket.htm
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

using namespace std;

#define PI 3.141592
#define BUFFER_SIZE 4096
#define ENDIAN_BYTES 1
#define HEADER_BYTES 4

//initial min and max HSV filter values.
//these will be changed using trackbars
//setting focus is black
#define H_MIN 15
#define H_MAX 35
#define S_MIN 128
#define S_MAX 256
#define V_MIN 128
#define V_MAX 256

#define STEER_LIMIT 0.8
#define THROTTLE_LIMIT 0.5

#define STEER_LEVEL 101
#define THROTTLE_LEVEL STEER_LEVEL

class LineRecognizer
{
private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher masked_pub_;
  image_transport::Publisher resized_pub_;
  cv::Mat HSV;
  cv::Mat mask;
  cv::Mat control_mask_view;
  cv::Mat control_mask;

  int sockfd, portno, n;
  struct sockaddr_in serv_addr;
  struct hostent *server;

  char buffer[256];

public:
  LineRecognizer(int argc, char **argv)
      : it_(nh_) //, drone(), obstacle(), obstacle_detection()
  {

    // Subscribe to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/camera/color/image_raw", 1,
                               &LineRecognizer::imageCb, this);
    masked_pub_ = it_.advertise("/yello_truck/masked", 1);
    resized_pub_ = it_.advertise("/yello_truck/resized", 1);

    portno = 12345;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
      error("ERROR opening socket");
    server = gethostbyname("127.0.0.1");
    if (server == NULL)
    {
      fprintf(stderr, "ERROR, no such host\n");
      exit(0);
    }
    bzero((char *)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr,
          (char *)&serv_addr.sin_addr.s_addr,
          server->h_length);
    serv_addr.sin_port = htons(portno);
    if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
      error("ERROR connecting");
  }

  ~LineRecognizer()
  {
    close(sockfd);
  }

  void imageCb(const sensor_msgs::ImageConstPtr &msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // Resize video stream to 1280 * 720
    cv::Mat resized;

    if (cv_ptr->image.cols == 640 && cv_ptr->image.rows == 480)
      resized = cv_ptr->image.clone();
    else
      cv::resize(cv_ptr->image, resized, cv::Size(640, 480));

    //convert frame from BGR to HSV colorspace
    cvtColor(resized, HSV, cv::COLOR_BGR2HSV);
    inRange(HSV, cv::Scalar(H_MIN, S_MIN, V_MIN), cv::Scalar(H_MAX, S_MAX, V_MAX), mask);

    cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
    cv_ptr->image = mask;
    masked_pub_.publish(cv_ptr->toImageMsg());
    cvtColor(mask, mask, cv::COLOR_BGR2GRAY);

    cv::resize(mask, control_mask, cv::Size(STEER_LEVEL, THROTTLE_LEVEL));
    cv::resize(control_mask, control_mask_view, cv::Size(640, 480));

    cvtColor(control_mask_view, control_mask_view, cv::COLOR_GRAY2BGR);
    cv_ptr->image = control_mask_view;
    resized_pub_.publish(cv_ptr->toImageMsg());

    double raw_steer[STEER_LEVEL] = {};
    double raw_throttle[THROTTLE_LEVEL] = {};

    // row = i, col = j
    for (int i = 0; i < STEER_LEVEL; i++)
    {
      double temp_steer = 0;
      double temp_throttle = 0;
      for (int j = 0; j < THROTTLE_LEVEL; j++)
      {
        temp_steer = temp_steer < control_mask.at<double>(j, i) ? control_mask.at<double>(j, i) : temp_steer;
        temp_throttle = temp_throttle < control_mask.at<double>(i, j) ? control_mask.at<double>(i, j) : temp_throttle;
      }
      raw_steer[i] = temp_steer;
      raw_throttle[i] = temp_throttle;
    }

    // ROS_INFO("");

    double steer_left = 0;
    double steer_right = 0;

    for (int i = 0; i < STEER_LEVEL / 2; i++)
    {
      steer_left += raw_steer[i];
      steer_right += raw_steer[STEER_LEVEL - i];
    }

    int max_steer_index, max_throttle_index;

    if (steer_left > steer_right)
    {
      max_steer_index = 0;
      double max = raw_steer[0];
      for (int i = 1; i < STEER_LEVEL / 2; i++)
      {
        if (raw_steer[i] > max)
        {
          max_steer_index = i;
          max = raw_steer[i];
        }
      }
      max_throttle_index = max_steer_index * 2 + 1;
    }
    else if (steer_left < steer_right)
    {
      max_steer_index = STEER_LEVEL - 1;
      double max = raw_steer[STEER_LEVEL - 1];
      for (int i = STEER_LEVEL - 2; i > STEER_LEVEL / 2 + 1; i--)
      {
        if (raw_steer[i] > max)
        {
          max_steer_index = i;
          max = raw_steer[i];
        }
      }
      max_throttle_index = (THROTTLE_LEVEL - max_steer_index - 1) * 2 + 1;
    }
    else
    {
      max_steer_index = STEER_LEVEL / 2;
      max_throttle_index = THROTTLE_LEVEL;
    }

    double steer, throttle;

    steer = (max_steer_index - STEER_LEVEL / 2) / (double)(STEER_LEVEL / 2) * STEER_LIMIT;
    throttle = max_throttle_index / (double)THROTTLE_LEVEL * THROTTLE_LIMIT;

    socket_control(steer, throttle);
  }

  void error(const char *msg)
  {
    perror(msg);
    exit(0);
  }

  int socket_control(double steer, double throttle)
  {
    bzero(buffer, 256);

    string steer_s = to_string(steer / 0.01 * 0.01);
    string throttle_s = to_string(throttle / 0.01 * 0.01);
    string data = string("control_data, ") + steer_s + string(", ") + throttle_s + string(", end");
    strcpy(buffer, data.c_str());

    n = write(sockfd, buffer, strlen(buffer));
    if (n < 0)
      error("ERROR writing to socket");
    return 0;
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "yello_truck");
  LineRecognizer lc(argc, argv);
  ros::spin();
  return 0;
}