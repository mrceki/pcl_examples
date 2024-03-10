#include "pcl_interface/pcl_interface.h"
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>

class PointCloudHandler
{
public:
    PointCloudHandler(ros::NodeHandle &nh) : nh_(nh)
    {
        sub_ = nh.subscribe("camera1/depth/color/points", 1, &PointCloudHandler::cloudCallback, this);
        pub_ = nh.advertise<sensor_msgs::PointCloud2>("filtered_clusters", 1);
    }

    void setParametersFromYAML(const std::string &yaml_file)
    {
        try
        {
            pcl_.setParametersFromYAML(params_, yaml_file);
        }
        catch (const YAML::Exception &e)
        {
            std::cerr << "YAML parsing error: " << e.what() << std::endl;
        }
    }

private:
    typedef PointCloudInterface::PointT PointT;

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
    {
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*cloud_msg, pcl_pc2);
        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
        pcl::fromPCLPointCloud2(pcl_pc2, *cloud);
        pcl_.cloud = cloud;

        double theta_x = M_PI / 2.0; 
        double theta_y = M_PI / 2.0; 
        double theta_z = M_PI / 1.0; 
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.rotate(Eigen::AngleAxisf(theta_x, Eigen::Vector3f::UnitX()));
        transform.rotate(Eigen::AngleAxisf(theta_y, Eigen::Vector3f::UnitY()));
        transform.rotate(Eigen::AngleAxisf(theta_z, Eigen::Vector3f::UnitZ()));
        pcl::transformPointCloud(*pcl_.cloud, *pcl_.cloud, transform);

        pcl_.passthroughFilterCloud(pcl_.cloud, params_.filter_params);
        pcl_.downsample(pcl_.cloud, params_.downsample_leaf_size);
        pcl_.segmentPlane(pcl_.cloud, params_.sac_params);
        pcl_.regionGrowing(pcl_.cloud, params_.region_growing_params);
        pcl_.extractClusters(pcl_.cloud, params_.ec_params);
        pcl_.createNewCloudFromIndicies(pcl_.cluster_indices, pcl_.cloud_cluster, params_.sac_params.min_indices);
        sensor_msgs::PointCloud2 filtered_clusters_msg;
        std::vector<tf::StampedTransform> transforms;
        pcl_.ros_cloud->clear();
        int i = 0;
        for (const auto &cluster : pcl_.clusters)
        {
            pcl_.segmentPlane(cluster, params_.cluster_sac_params);
            pcl_.performKMeans(cluster, params_.kmeans_cluster_size);
            pcl_.momentOfInertia(cluster, params_.moment_of_inertia_params);
            
            tf::Transform transform;
            transform.setOrigin(tf::Vector3(cluster->points[0].z, cluster->points[0].x, cluster->points[0].y));
            tf::Quaternion q;
            transform.setRotation(q);
            *pcl_.ros_cloud += *cluster;
            std::string cluster_frame_name = "cluster_frame_";
            tf::StampedTransform stampedTransform(transform, ros::Time::now(), "camera_link", cluster_frame_name);
            transforms.push_back(stampedTransform);
            i++;
        }

        static tf::TransformBroadcaster br;
        br.sendTransform(transforms);
        transforms.clear();

        pcl::toROSMsg(*pcl_.ros_cloud, filtered_clusters_msg);
        filtered_clusters_msg.header.frame_id = "camera_link";
        filtered_clusters_msg.header.stamp = ros::Time::now();
        pub_.publish(filtered_clusters_msg);
    }

    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher pub_;
    PointCloudInterface pcl_;
    PointCloudInterface::Parameters params_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pcl_interface");
    ros::NodeHandle nh;

    PointCloudHandler handler(nh);
    handler.setParametersFromYAML("/home/cenk/catkin_ws/src/pcl_interface/config/parameters.yaml");

    ros::spin();

    return 0;
}
