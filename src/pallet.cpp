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

int main()
{
    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
    reader.read("440903000.pcd", *cloud);

    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.01f, 0.01f, 0.01f);
    vg.filter(*cloud_filtered);

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PCDWriter writer;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.01);

    int nr_points = (int)cloud_filtered->size();
    while (cloud_filtered->size() > 0.3 * nr_points)
    {
        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0)
        {
            break;
        }

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(inliers);
        extract.setNegative(false);

        extract.filter(*cloud_plane);

        extract.setNegative(true);
        extract.filter(*cloud_f);
        *cloud_filtered = *cloud_f;
    }

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.015);
    ec.setMinClusterSize(50);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Cluster viewer"));

    int j = 0;
    for (const auto &cluster : cluster_indices)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto &idx : cluster.indices)
        {
            cloud_cluster->push_back((*cloud_filtered)[idx]);
        }
        cloud_cluster->width = cloud_cluster->size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << j;
        writer.write<pcl::PointXYZ>("cloud_cluster_" + ss.str() + ".pcd", *cloud_cluster, false);
        j++;

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.02);

        seg.setInputCloud(cloud_cluster);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0)
        {
                return (-1);
        }

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_cluster);
        extract.setIndices(inliers);
        extract.setNegative(false);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());
        extract.filter(*cloud_plane);

        pcl::Kmeans kmeans(cloud_plane->size(), 3);
        kmeans.setClusterSize(4);

        std::vector<std::vector<float>> input_data;
        for (const auto& point : *cloud_plane)
        {
                std::vector<float> point_data = {point.x, point.y, point.z};
                input_data.push_back(point_data);
        }

        kmeans.setInputData(input_data);

        kmeans.kMeans();
        pcl::Kmeans::Centroids centroids = kmeans.get_centroids();

        viewer->addPointCloud<pcl::PointXYZ>(cloud_plane, ss.str());
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, ss.str());

        for (int i = 0; i < centroids.size(); i++)
        {
                pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
                cluster->points.push_back(pcl::PointXYZ(centroids[i][0], centroids[i][1], centroids[i][2]));
                std::stringstream ss;
                ss << "centroid_" << i;
                viewer->addPointCloud<pcl::PointXYZ>(cluster, ss.str());
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, ss.str());
        }
        viewer->spin();
    }

    return (0);
}