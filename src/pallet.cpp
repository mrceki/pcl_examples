#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/ml/kmeans.h>

class PointCloudAnalyzer
{
public:
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster;
    pcl::Kmeans::Centroids centroids;


    PointCloudAnalyzer() : cloud(new pcl::PointCloud<pcl::PointXYZ>),
                           cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>),
                           cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>),
                           coefficients(new pcl::ModelCoefficients),
                           inliers(new pcl::PointIndices),
                           tree(new pcl::search::KdTree<pcl::PointXYZ>),
                           viewer(new pcl::visualization::PCLVisualizer("Cluster viewer"))
    {
    }

    struct SACParams
    {
        bool optimize_coefficients;
        int model_type;
        int method_type;
        float distance_threshold;
        int max_iterations;
    };

    struct EuclideanClusterParams
    {
        float cluster_tolerance;
        int min_cluster_size;
        int max_cluster_size;
    };

    void loadPCD(const std::string &filename)
    {
        cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
        reader.read(filename, *cloud);
        std::cout << "Loaded " << cloud->size() << " data points from " << filename << std::endl;
    }

    void downsample(float leaf_size)
    {
        vg.setInputCloud(cloud);
        vg.setLeafSize(leaf_size, leaf_size, leaf_size);
        vg.filter(*cloud);
        std::cout << "PointCloud after filtering has: " << cloud->size() << " data points." << std::endl; //*
    }

    void segmentPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, SACParams params)
    {
        seg.setOptimizeCoefficients(params.optimize_coefficients);
        seg.setModelType(params.model_type);
        seg.setMethodType(params.method_type);
        seg.setDistanceThreshold(params.distance_threshold);
        seg.setMaxIterations(params.max_iterations);

        int nr_points = (int)point_cloud->size();
        while (point_cloud->size() > 0.3 * nr_points)
        {
            seg.setInputCloud(point_cloud);
            seg.segment(*inliers, *coefficients);
            std::cout << "Segmented cloud size: " << inliers->indices.size() << std::endl;
            if (inliers->indices.size() == 0)
            {
                std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
                break;
            }

            extract.setInputCloud(point_cloud);
            extract.setIndices(inliers); //
            extract.setNegative(false);  // Extract the planar inliers from the input cloud
            extract.filter(*point_cloud);

            extract.setNegative(true); // Remove the planar inliers, extract the rest
            extract.filter(*cloud_filtered);
            *point_cloud = *cloud_filtered;
        }
        std::cout << "Plane segmentation completed." << std::endl;
    }

    void extractClusters(EuclideanClusterParams params)
    {
        tree->setInputCloud(cloud);
        ec.setClusterTolerance(params.cluster_tolerance);
        ec.setMinClusterSize(params.min_cluster_size);
        ec.setMaxClusterSize(params.max_cluster_size);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);
        std::cout << "Cluster extraction completed." << std::endl;
    }

    void performKMeans(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, int cluster_size)
    {
        pcl::Kmeans kmeans(point_cloud->size(), 3);
        kmeans.setClusterSize(cluster_size);
        std::vector<std::vector<float>> input_data;
        for (const auto &point : *point_cloud)
        {
            std::vector<float> point_data = {point.x, point.y, point.z};
            input_data.push_back(point_data);
        }
        kmeans.setInputData(input_data);
        kmeans.kMeans();
        centroids = kmeans.get_centroids();
        std::cout << "K-means clustering completed." << std::endl;
        
        // Logging centroids
        std::cout << "Centroids:" << std::endl;
        for (const auto &centroid : centroids)
        {
            std::cout << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << std::endl;
        }
    }

    void visualizeClusters(int point_size = 3)
    {
        for (const auto &cluster : cluster_indices)
        {
            for (const auto &idx : cluster.indices)
            {
                cloud_cluster->push_back((*cloud)[idx]);
            }
            cloud_cluster->width = cloud_cluster->size();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;
            viewer->removeAllPointClouds();
            viewer->addPointCloud<pcl::PointXYZ>(cloud_cluster);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size);
        }
        viewer->spin();
        std::cout << "Cluster visualization completed." << std::endl;
    }

    void writeClusters(const std::string &filename)
    {
        int j = 0;
        for (const auto &cluster : cluster_indices)
        {
            for (const auto &idx : cluster.indices)
            {
                cloud_cluster->push_back((*cloud)[idx]);
            }
            cloud_cluster->width = cloud_cluster->size();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;
            std::stringstream ss;
            ss << std::setw(4) << std::setfill('0') << j;
            writer.write<pcl::PointXYZ>(filename + ss.str() + ".pcd", *cloud_cluster, false);
            j++;
        }
        std::cout << "Cluster writing completed." << std::endl;
    }

private:
    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::visualization::PCLVisualizer::Ptr viewer;
    pcl::PCDWriter writer;
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::ModelCoefficients::Ptr coefficients;
    pcl::PointIndices::Ptr inliers;
    pcl::ExtractIndices<pcl::PointXYZ> extract;
};

int main()
{
    PointCloudAnalyzer pcl;

    pcl.loadPCD("440903000.pcd");
    std::cout << "PointCloud loaded." << std::endl;

    pcl.downsample(0.01f);

    PointCloudAnalyzer::SACParams sac_params;
    sac_params.optimize_coefficients = true;
    sac_params.model_type = pcl::SACMODEL_PLANE;
    sac_params.method_type = pcl::SAC_RANSAC;
    sac_params.distance_threshold = 0.01;
    sac_params.max_iterations = 100;
    pcl.segmentPlane(pcl.cloud, sac_params);

    std::cout << "PointCloud representing the planar component: " << pcl.cloud->size() << " data points." << std::endl;

    PointCloudAnalyzer::EuclideanClusterParams ec_params;
    ec_params.cluster_tolerance = 0.015;
    ec_params.min_cluster_size = 50;
    ec_params.max_cluster_size = 25000;
    pcl.extractClusters(ec_params);

    pcl.segmentPlane(pcl.cloud_cluster, sac_params);

    pcl.performKMeans(pcl.cloud_cluster, 5);

    pcl.visualizeClusters();

    // pcl.writeClusters("cloud_cluster_");

    return 0;
}