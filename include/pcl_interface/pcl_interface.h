#ifndef POINT_CLOUD_INTERFACE_H
#define POINT_CLOUD_INTERFACE_H

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/ml/kmeans.h>
#include <pcl/common/impl/angles.hpp>
#include <yaml-cpp/yaml.h>

class PointCloudInterface
{
public:
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::Kmeans::Centroids Centroids;
    pcl::PointCloud<PointT>::Ptr cloud;
    pcl::PointCloud<PointT>::Ptr cloud_cluster;
    pcl::Kmeans::Centroids centroids;
    std::vector<pcl::PointIndices> cluster_indices;
    std::vector<pcl::PointCloud<PointT>::Ptr> clusters;
    
    struct SACParams
    {
        bool optimize_coefficients;
        bool filtering;
        bool is_cloud_clustered;
        int model_type;
        int method_type;
        float distance_threshold;
        int max_iterations;
        int min_indices;
        std::vector<float> normal_axis;
        float angle_threshold;
    };

    struct EuclideanClusterParams
    {
        float cluster_tolerance;
        int min_cluster_size;
        int max_cluster_size;
    };

    struct FilterField
    {
        std::string field;
        float limit_min;
        float limit_max;
    };

    struct MomentOfInertiaParams
    {
        std::vector<float> moment_of_inertia;
        std::vector<float> eccentricity;
        PointT min_point_AABB;
        PointT max_point_AABB;
        PointT min_point_OBB;
        PointT max_point_OBB;
        PointT position_OBB;
        Eigen::Matrix3f rotational_matrix_OBB;
        float major_value, middle_value, minor_value;
        Eigen::Vector3f major_vector, middle_vector, minor_vector;
        Eigen::Vector3f mass_center;
    };

    struct RegionGrowingParams
    {
        int min_cluster_size;
        int max_cluster_size;
        int number_of_neighbours;
        float distance_threshold;
        float point_color_threshold;
        float region_color_threshold;
        float smoothness_threshold;
        float curvature_threshold;
    };

    struct FilterParams
    {
        std::vector<FilterField> passthrough_filter_fields;
        std::vector<FilterField> conditional_removal_fields;
    };

    struct SORParams
    {
        int mean_k;
        float stddev_mul_thresh;
    };

    struct RORParams
    {
        float radius_search;
        int min_neighbors_in_radius;
        bool keep_organized;
    };

    struct Parameters
    {
        std::string pcd_filepath, output_pcd_filepath;
        SACParams sac_params;
        SACParams cluster_sac_params;
        RegionGrowingParams region_growing_params;
        EuclideanClusterParams ec_params;
        FilterParams filter_params;
        MomentOfInertiaParams moment_of_inertia_params;
        SORParams sor_params;
        RORParams ror_params;
        float downsample_leaf_size;
        int kmeans_cluster_size;
        int visualization_point_size;
    };

    PointCloudInterface();
    ~PointCloudInterface();

    void loadPCD(const std::string &filename);
    void downsample(pcl::PointCloud<PointT>::Ptr point_cloud, float leaf_size);
    void segmentPlane(pcl::PointCloud<PointT>::Ptr point_cloud, SACParams params);
    void regionGrowing(pcl::PointCloud<PointT>::Ptr point_cloud, RegionGrowingParams &params);
    void extractClusters(pcl::PointCloud<PointT>::Ptr point_cloud, EuclideanClusterParams params);
    void createNewCloudFromIndicies(std::vector<pcl::PointIndices> cluster_indices, pcl::PointCloud<PointT>::Ptr cloud_cluster, int min_indices = 100);
    void performKMeans(pcl::PointCloud<PointT>::Ptr point_cloud, int cluster_size);
    void momentOfInertia(pcl::PointCloud<PointT>::Ptr point_cloud, MomentOfInertiaParams &params);
    void passthroughFilterCloud(pcl::PointCloud<PointT>::Ptr point_cloud, FilterParams &params);
    void statisticalOutlierRemoval(pcl::PointCloud<PointT>::Ptr point_cloud, Parameters &params);
    void radiusOutlierRemoval(pcl::PointCloud<PointT>::Ptr point_cloud, Parameters &params);
    void conditionalRemoval(pcl::PointCloud<PointT>::Ptr point_cloud, Parameters &params);
    auto mergeClouds(pcl::PointCloud<PointT>::Ptr cloud1, pcl::PointCloud<PointT>::Ptr cloud2);
    // void visualizeCluster(pcl::PointCloud<PointT>::Ptr point_cloud, Centroids centroids, Parameters &params);
    void visualiseMomentOfInertia(MomentOfInertiaParams p);
    void writeClusters(pcl::PointCloud<PointT>::Ptr cloud_cluster, Parameters &params);
    void setParametersFromYAML(Parameters &params, const std::string &yaml_file);

private:

    pcl::PCDReader reader;
    pcl::PointCloud<PointT>::Ptr cloud_filtered;
    pcl::search::KdTree<PointT>::Ptr tree;
    pcl::EuclideanClusterExtraction<PointT> ec;
    pcl::SACSegmentation<PointT> sac_seg;
    pcl::VoxelGrid<PointT> vg;
    pcl::PCDWriter writer;
    pcl::ModelCoefficients::Ptr coefficients;
    pcl::PointIndices::Ptr inliers;
    pcl::ExtractIndices<PointT> extract;
    // pcl::visualization::PCLVisualizer::Ptr viewer;
    pcl::PassThrough<PointT> pass;
    pcl::StatisticalOutlierRemoval<PointT> sor;
    pcl::RadiusOutlierRemoval<PointT> ror;
    pcl::ConditionalRemoval<PointT> cor;
    pcl::MomentOfInertiaEstimation<PointT> feature_extractor;
    int cluster_i, centroid_i;
};

#endif // POINT_CLOUD_INTERFACE_H
