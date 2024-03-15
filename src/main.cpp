#include "pcl_interface/pcl_interface.h"
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <dynamic_reconfigure/server.h>
#include <dolly_pose_estimation/DollyPoseEstimate.h>
class PointCloudHandler
{
public:
    PointCloudHandler(ros::NodeHandle &nh) : nh_(nh)
    {
        sub_ = nh.subscribe("camera1/depth/color/points", 1, &PointCloudHandler::cloudCallback, this);
        pub_ = nh.advertise<sensor_msgs::PointCloud2>("filtered_clusters", 1);
        dolly_client_ = nh.serviceClient<dolly_pose_estimation::DollyPoseEstimate>("/dolly_pose_estimation");
        dynamic_reconfigure::Server<pcl_interface::ParametersConfig> *server = new dynamic_reconfigure::Server<pcl_interface::ParametersConfig>;
        dynamic_reconfigure::Server<pcl_interface::ParametersConfig>::CallbackType f;
        f = boost::bind(&PointCloudHandler::reconfigureCallback, this, _1, _2);
        server->setCallback(f);
    }

    void setParameters(const std::string &yaml_file)
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

    void reconfigureCallback(pcl_interface::ParametersConfig &config, uint32_t level)
    {
        params_.filter_params.passthrough_filter_fields.clear();
        params_.filter_params.passthrough_filter_fields.push_back({"x", config.x_limit_min, config.x_limit_max});
        params_.filter_params.passthrough_filter_fields.push_back({"y", config.y_limit_min, config.y_limit_max});
        params_.filter_params.passthrough_filter_fields.push_back({"z", config.z_limit_min, config.z_limit_max});
        params_.filter_params.conditional_removal_fields.clear();
        params_.filter_params.conditional_removal_fields.push_back({"z", config.conditional_z_limit_min, config.conditional_z_limit_max});

        params_.downsample_leaf_size = config.downsample_leaf_size;
        params_.kmeans_cluster_size = config.kmeans_cluster_size;
        params_.visualization_point_size = config.visualization_point_size;

        params_.sac_params.optimize_coefficients = config.groups.sac_params.optimize_coefficients;
        params_.sac_params.model_type = config.groups.sac_params.model_type;
        params_.sac_params.method_type = config.groups.sac_params.method_type;
        params_.sac_params.max_iterations = config.groups.sac_params.max_iterations;
        params_.sac_params.distance_threshold = config.groups.sac_params.distance_threshold;
        params_.sac_params.filtering = config.groups.sac_params.filtering;
        params_.sac_params.min_indices = config.groups.sac_params.min_indices;
        params_.sac_params.is_cloud_clustered = config.groups.sac_params.is_cloud_clustered;
        params_.sac_params.normal_axis = {config.groups.sac_params.normal_axis_X, config.groups.sac_params.normal_axis_Y, config.groups.sac_params.normal_axis_Z};
        params_.sac_params.angle_threshold = config.groups.sac_params.angle_threshold;

        params_.cluster_sac_params.optimize_coefficients = config.groups.cluster_sac_params.cluster_optimize_coefficients;
        params_.cluster_sac_params.model_type = config.groups.cluster_sac_params.cluster_model_type;
        params_.cluster_sac_params.method_type = config.groups.cluster_sac_params.cluster_method_type;
        params_.cluster_sac_params.distance_threshold = config.groups.cluster_sac_params.cluster_distance_threshold;
        params_.cluster_sac_params.max_iterations = config.groups.cluster_sac_params.cluster_max_iterations;
        params_.cluster_sac_params.filtering = config.groups.cluster_sac_params.cluster_filtering;
        params_.cluster_sac_params.min_indices = config.groups.cluster_sac_params.cluster_min_indices;
        params_.cluster_sac_params.is_cloud_clustered = config.groups.cluster_sac_params.cluster_is_cloud_clustered;
        params_.cluster_sac_params.normal_axis = {config.groups.cluster_sac_params.cluster_normal_axis_X, config.groups.cluster_sac_params.cluster_normal_axis_Y, config.groups.cluster_sac_params.cluster_normal_axis_Z};
        params_.cluster_sac_params.angle_threshold = config.groups.cluster_sac_params.cluster_angle_threshold;

        params_.region_growing_params.min_cluster_size = config.groups.region_growing_params.rg_min_cluster_size;
        params_.region_growing_params.max_cluster_size = config.groups.region_growing_params.rg_max_cluster_size;
        params_.region_growing_params.number_of_neighbours = config.groups.region_growing_params.rg_number_of_neighbours;
        params_.region_growing_params.distance_threshold = config.groups.region_growing_params.rg_distance_threshold;
        params_.region_growing_params.point_color_threshold = config.groups.region_growing_params.rg_point_color_threshold;
        params_.region_growing_params.region_color_threshold = config.groups.region_growing_params.rg_region_color_threshold;
        params_.region_growing_params.smoothness_threshold = config.groups.region_growing_params.rg_smoothness_threshold;
        params_.region_growing_params.curvature_threshold = config.groups.region_growing_params.rg_curvature_threshold;

        params_.ec_params.cluster_tolerance = config.groups.ec_params.ec_cluster_tolerance;
        params_.ec_params.min_cluster_size = config.groups.ec_params.ec_min_cluster_size;
        params_.ec_params.max_cluster_size = config.groups.ec_params.ec_max_cluster_size;

        params_.sor_params.mean_k = config.groups.statistical_outlier_removal.sor_mean_k;
        params_.sor_params.stddev_mul_thresh = config.groups.statistical_outlier_removal.sor_stddev_mul_thresh;

        params_.ror_params.radius_search = config.groups.radius_outlier_removal.ror_radius_search;
        params_.ror_params.min_neighbors_in_radius = config.groups.radius_outlier_removal.ror_min_neighbors_in_radius;
        params_.ror_params.keep_organized = config.groups.radius_outlier_removal.ror_keep_organized;
    }

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
    {
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*cloud_msg, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2, *pcl_.cloud);

        double theta_x = M_PI / 2.0; 
        double theta_y = M_PI / 2.0; 
        double theta_z = M_PI / 1.0; 
        transform = Eigen::Affine3f::Identity();
        transform.rotate(Eigen::AngleAxisf(theta_x, Eigen::Vector3f::UnitX()));
        transform.rotate(Eigen::AngleAxisf(theta_y, Eigen::Vector3f::UnitY()));
        transform.rotate(Eigen::AngleAxisf(theta_z, Eigen::Vector3f::UnitZ()));
        pcl::transformPointCloud(*pcl_.cloud, *pcl_.cloud, transform);

        pcl_.passthroughFilterCloud(pcl_.cloud, params_.filter_params);
        pcl_.downsample(pcl_.cloud, params_.downsample_leaf_size);
        pcl_.segmentPlane(pcl_.cloud, params_.sac_params);
        // pcl_.regionGrowing(pcl_.cloud, params_.region_growing_params);
        pcl_.extractClusters(pcl_.cloud, params_.ec_params);
        pcl_.createNewCloudFromIndicies(pcl_.cluster_indices, pcl_.cloud_cluster, params_.sac_params.min_indices);
        int i = 0;
        geometry_msgs::PoseArray pose_array;
        dolly_pose_estimation::DollyPoseEstimate dolly_srv;
        for (const auto &cluster : pcl_.clusters)
        {
            pcl_.segmentPlane(cluster, params_.cluster_sac_params);
            pcl_.performKMeans(cluster, params_.kmeans_cluster_size);
            pcl_.momentOfInertia(cluster, params_.moment_of_inertia_params);
            transformTf.setOrigin(tf::Vector3(cluster->points[0].x, cluster->points[0].y, cluster->points[0].z));
            q.setRPY(0, 0, 0);
            transformTf.setRotation(q);
            *pcl_.ros_cloud += *cluster;
            cluster_frame_name = "cluster_frame_" + std::to_string(i);
            tf::StampedTransform stampedTransform(transformTf, ros::Time::now(), "camera_link", cluster_frame_name);
            tf_transforms.push_back(stampedTransform);
            pose_array.header.frame_id = "camera_link";
            pose_array.header.stamp = ros::Time::now();
            geometry_msgs::Pose pose;
            pose.position.x = cluster->points[0].x;
            pose.position.y = cluster->points[0].y;
            pose.position.z = cluster->points[0].z;
            pose.orientation.x = 0;
            pose.orientation.y = 0;
            pose.orientation.z = 0;
            pose.orientation.w = 1;
            pose_array.poses.push_back(pose);
            i++;
        }
        dolly_srv.request.cluster_poses = pose_array;
        
        if (!pose_array.poses.empty() && pose_array.poses.size() >= 4){
            if (dolly_client_.call(dolly_srv))
            {
                ROS_INFO("Cluster poses request successful");
            }
            else
            {
                ROS_ERROR("Failed to call cluster poses request");
            }
        }
        else
        {
            ROS_WARN("Not enough cluster poses to call service request");
        }

        br.sendTransform(tf_transforms);
        tf_transforms.clear();

        pcl::toROSMsg(*pcl_.ros_cloud, filtered_clusters_msg);
        filtered_clusters_msg.header.frame_id = "camera_link";
        filtered_clusters_msg.header.stamp = ros::Time::now();
        pub_.publish(filtered_clusters_msg);
        clearClouds(pcl_);

    }

    void clearClouds(class PointCloudInterface &pcl)
    {
        pcl.ros_cloud.reset(new pcl::PointCloud<PointT>());
        pcl.clusters.clear();
        pcl.cluster_indices.clear();
        pcl.cloud_cluster.reset(new pcl::PointCloud<PointT>());
    }

    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher pub_;
    ros::ServiceClient dolly_client_;
    Eigen::Affine3f transform;
    std::vector<tf::StampedTransform> tf_transforms;
    sensor_msgs::PointCloud2 filtered_clusters_msg;
    PointCloudInterface pcl_;
    PointCloudInterface::Parameters params_;
    tf::TransformBroadcaster br;
    tf::Transform transformTf;
    tf::Quaternion q;
    std::string cluster_frame_name;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pcl_interface_node");
    ros::NodeHandle nh;

    PointCloudHandler handler(nh);
    handler.setParameters("/home/cenk/catkin_ws/src/pcl_interface/config/parameters.yaml");

    ros::spin();

    return 0;
}
