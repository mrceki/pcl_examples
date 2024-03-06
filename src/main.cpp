#include "pcl_interface/pcl_interface.h"

int main(){
    PointCloudInterface pcl;
    PointCloudInterface::Parameters params;

    try
    {
        pcl.setParametersFromYAML(params, "/home/cenk/catkin_ws/src/pcl_interface/config/parameters.yaml");
    }
    catch (const YAML::Exception &e)
    {
        std::cerr << "YAML parsing error: " << e.what() << std::endl;
    }

    pcl.loadPCD(params.pcd_filepath);
    std::cout << "PointCloud loaded." << std::endl;

    pcl.passthroughFilterCloud(pcl.cloud, params.filter_params);
    pcl.conditionalRemoval(pcl.cloud, params);
    pcl.downsample(pcl.cloud, params.downsample_leaf_size);
    pcl.segmentPlane(pcl.cloud, params.sac_params);
    std::cout << "PointCloud representing the planar component: " << pcl.cloud->size() << " data points." << std::endl;
    pcl.regionGrowing(pcl.cloud, params.region_growing_params);
    pcl.extractClusters(pcl.cloud, params.ec_params);
    pcl.createNewCloudFromIndicies(pcl.cluster_indices, pcl.cloud_cluster, params.sac_params.min_indices);

    for (const auto &cluster : pcl.clusters)
    {
        std::cout << "Cluster size: " << cluster->size() << std::endl;
        pcl.segmentPlane(cluster, params.cluster_sac_params);
        pcl.performKMeans(cluster, params.kmeans_cluster_size);
        pcl.momentOfInertia(cluster, params.moment_of_inertia_params);
        // pcl.writeClusters(cluster, params);
        // pcl.visualizeCluster(cluster, pcl.centroids, params);
    }

    return 0;
}