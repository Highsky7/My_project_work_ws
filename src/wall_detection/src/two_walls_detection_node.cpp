/****************************************************
 * two_walls_detection_node.cpp
 *
 * - /velodyne_points 포인트클라우드를 구독
 * - max_range (기본 2m) 이내의 포인트만 필터링
 * - z: [-1,2], y: [-5,5] 범위에서만 사용
 * - x축 기준으로 왼쪽(<=0), 오른쪽(>=0) 클라우드 분리
 * - 각각 RANSAC으로 수직 평면(벽)을 검출
 *   (ransac_thresh=0.05, vertical_thres=0.3, min_inliers=1000)
 * - 검출 결과는 콘솔 로그로만 출력
 * - 마커 등 시각화는 전혀 수행하지 않음
 ****************************************************/

 #include <ros/ros.h>
 #include <sensor_msgs/PointCloud2.h>
 
 #include <pcl_conversions/pcl_conversions.h>
 #include <pcl/point_cloud.h>
 #include <pcl/point_types.h>
 #include <pcl/filters/passthrough.h>
 #include <pcl/filters/extract_indices.h>
 #include <pcl/segmentation/sac_segmentation.h>
 #include <cmath>
 
 class TwoWallsDetectionNode
 {
 public:
   TwoWallsDetectionNode()
   {
     ros::NodeHandle nh, pnh("~");
 
     // 하드코딩된 값들
     input_topic_ = "/velodyne_points";
     z_min_ = -1.0;  
     z_max_ = 2.0;
     y_min_ = -5.0;  
     y_max_ = 5.0;
     left_x_max_ = 0.0;
     right_x_min_ = 0.0;
     ransac_thresh_ = 0.05;
     vertical_thres_ = 0.3;
     min_inliers_ = 1000;
 
     // max_range만 파라미터로 받고 기본값 2.0m
     pnh.param<double>("max_range", max_range_, 2.0);
 
     sub_cloud_ = nh.subscribe(input_topic_, 1, &TwoWallsDetectionNode::cloudCallback, this);
 
     ROS_INFO("TwoWallsDetectionNode initialized (no marker).");
   }
 
 private:
   ros::Subscriber sub_cloud_;
 
   std::string input_topic_;
   double z_min_, z_max_;
   double y_min_, y_max_;
   double left_x_max_, right_x_min_;
   double ransac_thresh_;
   double vertical_thres_;
   int min_inliers_;
   double max_range_;
 
   // max_range 필터
   void rangeFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_cloud,
                    double max_dist,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr &out_cloud)
   {
     out_cloud->points.clear();
     for (const auto &pt : in_cloud->points)
     {
       float d = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
       if(d <= max_dist)
         out_cloud->points.push_back(pt);
     }
     out_cloud->width = out_cloud->points.size();
     out_cloud->height = 1;
     out_cloud->is_dense = true;
   }
 
   // PassThrough 필터
   void passFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                   const std::string &field,
                   double min_val, double max_val)
   {
     pcl::PassThrough<pcl::PointXYZ> pass;
     pass.setInputCloud(cloud);
     pass.setFilterFieldName(field);
     pass.setFilterLimits(min_val, max_val);
     pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
     pass.filter(*tmp);
     cloud.swap(tmp);
   }
 
   // RANSAC으로 수직 평면(벽) 검출
   bool segmentWall(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr &wall_out)
   {
     if(in->empty()) return false;
 
     pcl::SACSegmentation<pcl::PointXYZ> seg;
     seg.setOptimizeCoefficients(true);
     seg.setModelType(pcl::SACMODEL_PLANE);
     seg.setMethodType(pcl::SAC_RANSAC);
     seg.setDistanceThreshold(ransac_thresh_);
     seg.setInputCloud(in);
 
     pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
     pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
     seg.segment(*inliers, *coeff);
 
     if(inliers->indices.empty()) return false;
 
     // 수직 평면 => z축 성분 |C| < vertical_thres_
     float C = coeff->values[2];
     if(std::fabs(C) > vertical_thres_) return false;
 
     pcl::ExtractIndices<pcl::PointXYZ> extract;
     extract.setInputCloud(in);
     extract.setIndices(inliers);
     extract.setNegative(false);
     extract.filter(*wall_out);
 
     if((int)wall_out->size() < min_inliers_) return false;
     return true;
   }
 
   void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
   {
     // 1) ROS -> PCL
     pcl::PointCloud<pcl::PointXYZ>::Ptr raw(new pcl::PointCloud<pcl::PointXYZ>);
     pcl::fromROSMsg(*msg, *raw);
     if(raw->empty()) {
       ROS_INFO_THROTTLE(1.0, "No points in cloud.");
       return;
     }
 
     // 2) max_range 필터
     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
     rangeFilter(raw, max_range_, cloud);
     if(cloud->empty()) {
       ROS_INFO_THROTTLE(1.0, "No points after rangeFilter.");
       return;
     }
 
     // 3) z, y 축 필터
     passFilter(cloud, "z", z_min_, z_max_);
     passFilter(cloud, "y", y_min_, y_max_);
     if(cloud->empty()) {
       ROS_INFO_THROTTLE(1.0, "No points after z,y filter.");
       return;
     }
 
     // 4) x축 기준 왼/오 분리
     pcl::PointCloud<pcl::PointXYZ>::Ptr left(new pcl::PointCloud<pcl::PointXYZ>);
     pcl::PointCloud<pcl::PointXYZ>::Ptr right(new pcl::PointCloud<pcl::PointXYZ>);
     {
       pcl::PassThrough<pcl::PointXYZ> pass;
       pass.setInputCloud(cloud);
       pass.setFilterFieldName("x");
       pass.setFilterLimits(-1e6, left_x_max_);
       pass.filter(*left);
     }
     {
       pcl::PassThrough<pcl::PointXYZ> pass;
       pass.setInputCloud(cloud);
       pass.setFilterFieldName("x");
       pass.setFilterLimits(right_x_min_, 1e6);
       pass.filter(*right);
     }
 
     // 5) 벽 검출 (왼쪽, 오른쪽)
     pcl::PointCloud<pcl::PointXYZ>::Ptr wall_left (new pcl::PointCloud<pcl::PointXYZ>);
     pcl::PointCloud<pcl::PointXYZ>::Ptr wall_right(new pcl::PointCloud<pcl::PointXYZ>);
     bool found_left  = segmentWall(left,  wall_left);
     bool found_right = segmentWall(right, wall_right);
 
     if(found_left)  ROS_INFO_THROTTLE(1.0, "Left wall detected. inliers=%d", (int)wall_left->size());
     else            ROS_INFO_THROTTLE(1.0, "No left wall.");
 
     if(found_right) ROS_INFO_THROTTLE(1.0, "Right wall detected. inliers=%d",(int)wall_right->size());
     else            ROS_INFO_THROTTLE(1.0, "No right wall.");
 
     // 두 벽 모두 인식된 경우
     if(found_left && found_right) {
       ROS_INFO_THROTTLE(1.0, "Both walls are detected!");
     }
   }
 };
 
 int main(int argc, char** argv)
 {
   ros::init(argc, argv, "two_walls_detection_node");
   TwoWallsDetectionNode node;
   ros::spin();
   return 0;
 }
 