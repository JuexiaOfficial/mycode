#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
public:
  FeatureTracker();

  void readImage(const cv::Mat &_img);

  void readImage(const cv::Mat &_img, double _cur_time);

  void setMask();

  void addPoints();

  bool updateID(unsigned int i);

  void readIntrinsicParameter(const string &calib_file);

  void showUndistortion(const string &name);

  void showUndistortion(); // pl-vins

  void rejectWithF();

  vector<cv::Point2f> undistortedPoints(); // pl-vins

  void myUndistortedPoints();

  cv::Mat mask;
  cv::Mat fisheye_mask;
  cv::Mat prev_img, cur_img, forw_img; //  prev : i-1 时刻，  cur: i 时刻， forw： i+1时刻
  vector<cv::Point2f> n_pts;

  vector<cv::Point2f> prev_pts, cur_pts, forw_pts;

  vector<cv::Point2f> prev_un_pts, cur_un_pts; // pl-vins定义了但是没用过
  vector<cv::Point2f> pts_velocity;

  vector<int> ids;       //  每个特征点的id
  vector<int> track_cnt; //  记录某个特征已经跟踪多少帧了，即被多少帧看到了
  camodocal::CameraPtr m_camera;

  // 为时偏估计而准备的
  map<int, cv::Point2f> prev_un_pts_map, cur_un_pts_map;
  double cur_time;
  double pre_time;

  static int n_id;
};
