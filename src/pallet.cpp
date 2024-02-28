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
#include <yaml-cpp/yaml.h>

class PointCloudAnalyzer
{
public:
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster;
    pcl::Kmeans::Centroids centroids;
    std::vector<pcl::PointIndices> cluster_indices;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
    PointCloudAnalyzer() : cloud(new pcl::PointCloud<pcl::PointXYZ>),
                           cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>),
                           coefficients(new pcl::ModelCoefficients),
                           inliers(new pcl::PointIndices),
                           tree(new pcl::search::KdTree<pcl::PointXYZ>)
    {
    }

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
    };

    struct EuclideanClusterParams
    {
        float cluster_tolerance;
        int min_cluster_size;
        int max_cluster_size;
    };
    struct Parameters
    {
        std::string pcd_filepath;
        SACParams sac_params;
        SACParams cluster_sac_params;
        EuclideanClusterParams ec_params;
        float downsample_leaf_size;
        int kmeans_cluster_size;
        int visualization_point_size;
    };

    void loadPCD(const std::string &filename)
    {
        cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
        reader.read(filename, *cloud);
        std::cout << "Loaded " << cloud->size() << " data points from " << filename << std::endl;
    }

    void downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, float leaf_size)
    {
        vg.setInputCloud(point_cloud);
        vg.setLeafSize(leaf_size, leaf_size, leaf_size);
        vg.filter(*point_cloud);
        std::cout << "PointCloud after filtering has: " << point_cloud->size() << " data points." << std::endl; //*
    }

    void segmentPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, SACParams params)
    {
        seg.setOptimizeCoefficients(params.optimize_coefficients);
        seg.setModelType(params.model_type);
        seg.setMethodType(params.method_type);
        seg.setDistanceThreshold(params.distance_threshold);
        seg.setMaxIterations(params.max_iterations);

        int nr_points = (int)point_cloud->size();
        if (!params.is_cloud_clustered)
        {
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
                extract.setIndices(inliers);
                extract.setNegative(params.filtering);

                extract.filter(*cloud_filtered);
                *point_cloud = *cloud_filtered;
            }
        }
        else
        {
            seg.setInputCloud(point_cloud);
            seg.segment(*inliers, *coefficients);
            std::cout << "Segmented cloud size: " << inliers->indices.size() << std::endl;
            if (inliers->indices.size() == 0)
            {
                std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
            }
            else if (inliers->indices.size() < params.min_indices)
            {
                std::cout << "Not enough points to estimate a planar model for the given dataset." << std::endl;
            }
            else
            {
                extract.setInputCloud(point_cloud);
                extract.setIndices(inliers);
                extract.setNegative(params.filtering);

                extract.filter(*cloud_filtered);
                *point_cloud = *cloud_filtered;
            }
        }
        std::cout << "Plane segmentation completed." << std::endl;
    }

    void extractClusters(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, EuclideanClusterParams params)
    {
        tree->setInputCloud(point_cloud);
        ec.setClusterTolerance(params.cluster_tolerance);
        ec.setMinClusterSize(params.min_cluster_size);
        ec.setMaxClusterSize(params.max_cluster_size);
        ec.setSearchMethod(tree);
        ec.setInputCloud(point_cloud);
        ec.extract(cluster_indices);
        std::cout << "cluster_indices size: " << cluster_indices.size() << std::endl;
        std::cout << "Cluster extraction completed." << std::endl;
    }
    void createNewCloudFromIndicies(std::vector<pcl::PointIndices> cluster_indices, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster, int min_indices = 100)
    {
        std::cout << "Creating new cloud from indices" << std::endl;
        std::cout << "cluster_indices size: " << cluster_indices.size() << std::endl;

        for (const auto &cluster : cluster_indices)
        {
            cloud_cluster.reset(new pcl::PointCloud<pcl::PointXYZ>);
            std::cout << "Cluster Started" << std::endl;
            for (const auto &idx : cluster.indices)
            {
                cloud_cluster->push_back((*cloud)[idx]);
            }
            if (cloud_cluster->size() > min_indices)
            {
                cloud_cluster->width = cloud_cluster->size();
                cloud_cluster->height = 1;
                cloud_cluster->is_dense = true;
                clusters.push_back(cloud_cluster);
                std::cout << "Cluster Pushed" << std::endl;
            }
        }
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

        std::cout << "centroids size: " << centroids.size() << std::endl;
        std::cout << "Centroids:" << std::endl;
        for (const auto &centroid : centroids)
        {
            std::cout << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << std::endl;
        }
    }

    void visualizeCluster(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, pcl::Kmeans::Centroids centroids, int point_size = 3)
    {
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Cluster Viewer"));

        for (int i = 0; i < centroids.size(); i++)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            cluster->points.push_back(pcl::PointXYZ(centroids[i][0], centroids[i][1], centroids[i][2]));
            std::stringstream ss;
            ss << "centroid_" << i;
            viewer->addPointCloud<pcl::PointXYZ>(cluster, ss.str());
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size * 3, ss.str());
        }

        viewer->addPointCloud<pcl::PointXYZ>(point_cloud);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size);
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

    void setParametersFromYAML(PointCloudAnalyzer::Parameters &params,
                               const std::string &yaml_file)
    {
        YAML::Node config = YAML::LoadFile(yaml_file);

        params.pcd_filepath = config["pcd_filepath"].as<std::string>();
        params.downsample_leaf_size = config["downsample_leaf_size"].as<float>();
        params.kmeans_cluster_size = config["kmeans_cluster_size"].as<int>();
        params.visualization_point_size = config["visualization_point_size"].as<int>();

        params.sac_params.optimize_coefficients = config["sac_params"]["optimize_coefficients"].as<bool>();
        params.sac_params.model_type = config["sac_params"]["model_type"].as<int>();
        params.sac_params.method_type = config["sac_params"]["method_type"].as<int>();
        params.sac_params.max_iterations = config["sac_params"]["max_iterations"].as<int>();
        params.sac_params.distance_threshold = config["sac_params"]["distance_threshold"].as<float>();
        params.sac_params.filtering = config["sac_params"]["filtering"].as<bool>();
        params.sac_params.min_indices = config["sac_params"]["min_indices"].as<int>();

        params.cluster_sac_params.optimize_coefficients = config["cluster_sac_params"]["optimize_coefficients"].as<bool>();
        params.cluster_sac_params.model_type = config["cluster_sac_params"]["model_type"].as<int>();
        params.cluster_sac_params.method_type = config["cluster_sac_params"]["method_type"].as<int>();
        params.cluster_sac_params.distance_threshold = config["cluster_sac_params"]["distance_threshold"].as<float>();
        params.cluster_sac_params.max_iterations = config["cluster_sac_params"]["max_iterations"].as<int>();
        params.cluster_sac_params.filtering = config["cluster_sac_params"]["filtering"].as<bool>();
        params.cluster_sac_params.min_indices = config["cluster_sac_params"]["min_indices"].as<int>();
        params.cluster_sac_params.is_cloud_clustered = config["cluster_sac_params"]["is_cloud_clustered"].as<bool>();

        params.ec_params.cluster_tolerance = config["ec_params"]["cluster_tolerance"].as<float>();
        params.ec_params.min_cluster_size = config["ec_params"]["min_cluster_size"].as<int>();
        params.ec_params.max_cluster_size = config["ec_params"]["max_cluster_size"].as<int>();
        std::cout << "filepath: " << params.pcd_filepath << std::endl;
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
    pcl::ModelCoefficients::Ptr coefficients;
    pcl::PointIndices::Ptr inliers;
    pcl::ExtractIndices<pcl::PointXYZ> extract;
};

int main()
{
    PointCloudAnalyzer pcl;
    PointCloudAnalyzer::Parameters params;

    try
    {
        pcl.setParametersFromYAML(params, "/home/cenk/pcl_examples/config/parameters.yaml");
    }
    catch (const YAML::Exception &e)
    {
        std::cerr << "YAML parsing error: " << e.what() << std::endl;
    }

    pcl.loadPCD(params.pcd_filepath);
    std::cout << "PointCloud loaded." << std::endl;

    pcl.downsample(pcl.cloud, params.downsample_leaf_size);
    pcl.segmentPlane(pcl.cloud, params.sac_params);
    std::cout << "PointCloud representing the planar component: " << pcl.cloud->size() << " data points." << std::endl;

    pcl.extractClusters(pcl.cloud, params.ec_params);
    pcl.createNewCloudFromIndicies(pcl.cluster_indices, pcl.cloud_cluster, params.sac_params.min_indices);

    for (const auto &cluster : pcl.clusters)
    {
        std::cout << "Cluster size: " << cluster->size() << std::endl;
        pcl.segmentPlane(cluster, params.cluster_sac_params);
        pcl.performKMeans(cluster, params.kmeans_cluster_size);
        pcl.visualizeCluster(cluster, pcl.centroids, params.visualization_point_size);
    }

    return 0;
}