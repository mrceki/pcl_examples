#include "pcl_interface/pcl_interface.h"

float calculateDistance(float p1, float p2)
{
    return sqrt(pow(p1, 2) + pow(p2, 2));
}

PointCloudInterface::PointCloudInterface() : cloud(new pcl::PointCloud<PointT>),
                                             cloud_filtered(new pcl::PointCloud<PointT>),
                                             coefficients(new pcl::ModelCoefficients),
                                             inliers(new pcl::PointIndices),
                                             tree(new pcl::search::KdTree<PointT>),
                                             ros_cloud(new pcl::PointCloud<PointT>),
                                            //  viewer(new pcl::visualization::PCLVisualizer("Cluster Viewer")),
                                             cluster_i(0), centroid_i(0)
{
}

PointCloudInterface::~PointCloudInterface()
{
    // viewer->close();
}

void PointCloudInterface::loadPCD(const std::string &filename)
{
    cloud.reset(new pcl::PointCloud<PointT>);
    reader.read(filename, *cloud);
    std::cout << "Loaded " << cloud->width * cloud->height << " data points from " << filename << std::endl;
}

void PointCloudInterface::setParametersFromYAML(PointCloudInterface::Parameters &params,
                               const std::string &yaml_file)
    {
        YAML::Node config = YAML::LoadFile(yaml_file);
        YAML::Node passthrougFilter = config["passthrough_filter"];
        YAML::Node conditionalRemoval = config["conditional_removal"];
        params.pcd_filepath = config["pcd_filepath"].as<std::string>();
        params.output_pcd_filepath = config["output_pcd_filepath"].as<std::string>();

        auto parseFilterFields = [&](const YAML::Node &node, std::vector<PointCloudInterface::FilterField> &fields)
        {
            for (const auto &config : node)
            {
                fields.push_back({config["field"].as<std::string>(),
                                  config["limit_min"].as<float>(),
                                  config["limit_max"].as<float>()});
            }
        };

        parseFilterFields(passthrougFilter, params.filter_params.passthrough_filter_fields);
        parseFilterFields(conditionalRemoval, params.filter_params.conditional_removal_fields);

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
        params.sac_params.normal_axis = config["cluster_sac_params"]["normal_axis"].as<std::vector<float>>();
        params.sac_params.angle_threshold = config["cluster_sac_params"]["angle_threshold"].as<float>();

        params.region_growing_params.min_cluster_size = config["region_growing_params"]["min_cluster_size"].as<int>();
        params.region_growing_params.max_cluster_size = config["region_growing_params"]["max_cluster_size"].as<int>();
        params.region_growing_params.number_of_neighbours = config["region_growing_params"]["number_of_neighbours"].as<int>();
        params.region_growing_params.distance_threshold = config["region_growing_params"]["distance_threshold"].as<float>();
        params.region_growing_params.point_color_threshold = config["region_growing_params"]["point_color_threshold"].as<float>();
        params.region_growing_params.region_color_threshold = config["region_growing_params"]["region_color_threshold"].as<float>();
        params.region_growing_params.smoothness_threshold = config["region_growing_params"]["smoothness_threshold"].as<float>();
        params.region_growing_params.curvature_threshold = config["region_growing_params"]["curvature_threshold"].as<float>();

        params.cluster_sac_params.optimize_coefficients = config["cluster_sac_params"]["optimize_coefficients"].as<bool>();
        params.cluster_sac_params.model_type = config["cluster_sac_params"]["model_type"].as<int>();
        params.cluster_sac_params.method_type = config["cluster_sac_params"]["method_type"].as<int>();
        params.cluster_sac_params.distance_threshold = config["cluster_sac_params"]["distance_threshold"].as<float>();
        params.cluster_sac_params.max_iterations = config["cluster_sac_params"]["max_iterations"].as<int>();
        params.cluster_sac_params.filtering = config["cluster_sac_params"]["filtering"].as<bool>();
        params.cluster_sac_params.min_indices = config["cluster_sac_params"]["min_indices"].as<int>();
        params.cluster_sac_params.is_cloud_clustered = config["cluster_sac_params"]["is_cloud_clustered"].as<bool>();
        params.cluster_sac_params.normal_axis = config["cluster_sac_params"]["normal_axis"].as<std::vector<float>>();
        params.cluster_sac_params.angle_threshold = config["cluster_sac_params"]["angle_threshold"].as<float>();

        params.ec_params.cluster_tolerance = config["ec_params"]["cluster_tolerance"].as<float>();
        params.ec_params.min_cluster_size = config["ec_params"]["min_cluster_size"].as<int>();
        params.ec_params.max_cluster_size = config["ec_params"]["max_cluster_size"].as<int>();

        params.sor_params.mean_k = config["statistical_outlier_removal"]["mean_k"].as<int>();
        params.sor_params.stddev_mul_thresh = config["statistical_outlier_removal"]["stddev_mul_thresh"].as<float>();

        params.ror_params.radius_search = config["radius_outlier_removal"]["radius_search"].as<float>();
        params.ror_params.min_neighbors_in_radius = config["radius_outlier_removal"]["min_neighbors_in_radius"].as<int>();
        params.ror_params.keep_organized = config["radius_outlier_removal"]["keep_organized"].as<bool>();
    }

void PointCloudInterface::downsample(pcl::PointCloud<PointT>::Ptr point_cloud, float leaf_size)
{
        vg.setInputCloud(point_cloud);
        vg.setLeafSize(leaf_size, leaf_size, leaf_size);
        vg.filter(*point_cloud);
        std::cout << "PointCloud after filtering has: " << point_cloud->size() << " data points." << std::endl;
}

void PointCloudInterface::segmentPlane(pcl::PointCloud<PointT>::Ptr point_cloud, SACParams params)
{
        sac_seg.setOptimizeCoefficients(params.optimize_coefficients);
        sac_seg.setModelType(params.model_type);
        sac_seg.setMethodType(params.method_type);
        sac_seg.setDistanceThreshold(params.distance_threshold);
        sac_seg.setMaxIterations(params.max_iterations);
        sac_seg.setAxis(Eigen::Vector3f(params.normal_axis[0], params.normal_axis[1], params.normal_axis[2]));
        sac_seg.setEpsAngle(pcl::deg2rad(params.angle_threshold));

        int nr_points = (int)point_cloud->size();
        if (!params.is_cloud_clustered)
        {
            while (point_cloud->size() > 0.3 * nr_points)
            {
                sac_seg.setInputCloud(point_cloud);
                sac_seg.segment(*inliers, *coefficients);
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
            sac_seg.setInputCloud(point_cloud);
            sac_seg.segment(*inliers, *coefficients);
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

void PointCloudInterface::regionGrowing(pcl::PointCloud<PointT>::Ptr point_cloud, RegionGrowingParams &params)
{
    pcl::RegionGrowingRGB<PointT> reg;
    reg.setSearchMethod(tree);
    reg.setMinClusterSize(params.min_cluster_size);
    reg.setMaxClusterSize(params.max_cluster_size);
    reg.setNumberOfNeighbours(params.number_of_neighbours);
    reg.setDistanceThreshold(params.distance_threshold);
    reg.setPointColorThreshold(params.point_color_threshold);
    reg.setRegionColorThreshold(params.region_color_threshold);
    reg.setSmoothnessThreshold(params.smoothness_threshold / 180.0 * M_PI);
    reg.setCurvatureThreshold(params.curvature_threshold);
    reg.setInputCloud(point_cloud);
    // std::vector<pcl::PointIndices> clusters;
    // reg.extract(clusters);
    point_cloud = reg.getColoredCloud();
}

void PointCloudInterface::extractClusters(pcl::PointCloud<PointT>::Ptr point_cloud, EuclideanClusterParams params)
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

void PointCloudInterface::createNewCloudFromIndicies(std::vector<pcl::PointIndices> cluster_indices, pcl::PointCloud<PointT>::Ptr cloud_cluster, int min_indices)
{
    std::cout << "Creating new cloud from indices" << std::endl;
    std::cout << "cluster_indices size: " << cluster_indices.size() << std::endl;
    for (const auto &cluster : cluster_indices)
    {
        cloud_cluster.reset(new pcl::PointCloud<PointT>);
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

void PointCloudInterface::performKMeans(pcl::PointCloud<PointT>::Ptr point_cloud, int cluster_size)
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

void PointCloudInterface::momentOfInertia(pcl::PointCloud<PointT>::Ptr point_cloud, MomentOfInertiaParams &params)
{
    feature_extractor.setInputCloud(point_cloud);
    feature_extractor.compute();
    feature_extractor.getMomentOfInertia(params.moment_of_inertia);
    feature_extractor.getEccentricity(params.eccentricity);
    feature_extractor.getAABB(params.min_point_AABB, params.max_point_AABB);
    feature_extractor.getOBB(params.min_point_OBB, params.max_point_OBB, params.position_OBB, params.rotational_matrix_OBB);
    feature_extractor.getEigenValues(params.major_value, params.middle_value, params.minor_value);
    feature_extractor.getEigenVectors(params.major_vector, params.middle_vector, params.minor_vector);
    feature_extractor.getMassCenter(params.mass_center);
}

void PointCloudInterface::passthroughFilterCloud(pcl::PointCloud<PointT>::Ptr point_cloud, FilterParams &params)
{
    for (auto field : params.passthrough_filter_fields)
    {
        pass.setInputCloud(point_cloud);
        pass.setFilterFieldName(field.field);
        pass.setFilterLimits(field.limit_min, field.limit_max);
        pass.filter(*cloud_filtered);
    }
}

void PointCloudInterface::statisticalOutlierRemoval(pcl::PointCloud<PointT>::Ptr point_cloud, Parameters &params)
{
    sor.setInputCloud(point_cloud);
    sor.setMeanK(params.sor_params.mean_k);
    sor.setStddevMulThresh(params.sor_params.stddev_mul_thresh);
    sor.filter(*cloud_filtered);
}

void PointCloudInterface::radiusOutlierRemoval(pcl::PointCloud<PointT>::Ptr point_cloud, Parameters &params)
{
    ror.setInputCloud(point_cloud);
    ror.setRadiusSearch(params.ror_params.radius_search);
    ror.setMinNeighborsInRadius(params.ror_params.min_neighbors_in_radius);
    ror.setKeepOrganized(params.ror_params.keep_organized);
    ror.filter(*cloud_filtered);
}

void PointCloudInterface::conditionalRemoval(pcl::PointCloud<PointT>::Ptr point_cloud, Parameters &params)
{
    pcl::ConditionAnd<PointT>::Ptr range_cond(new pcl::ConditionAnd<PointT>());
    for (auto field : params.filter_params.conditional_removal_fields)
    {
        pcl::FieldComparison<PointT>::ConstPtr fcomp(new pcl::FieldComparison<PointT>(field.field, pcl::ComparisonOps::GT, field.limit_min));
        pcl::FieldComparison<PointT>::ConstPtr fcomp2(new pcl::FieldComparison<PointT>(field.field, pcl::ComparisonOps::LT, field.limit_max));
        range_cond->addComparison(fcomp);
        range_cond->addComparison(fcomp2);
    }
    cor.setCondition(range_cond);
    cor.setInputCloud(point_cloud);
    cor.setKeepOrganized(true);
    cor.filter(*cloud_filtered);
}

auto PointCloudInterface::mergeClouds(pcl::PointCloud<PointT>::Ptr cloud1, pcl::PointCloud<PointT>::Ptr cloud2)
{
    pcl::PointCloud<PointT>::Ptr combined_cloud(new pcl::PointCloud<PointT>);
    *combined_cloud = *cloud1 + *cloud2;
    return combined_cloud;
}

void PointCloudInterface::writeClusters(pcl::PointCloud<PointT>::Ptr cloud_cluster, Parameters &params)
{
    cloud_cluster->width = cloud_cluster->size();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;
    std::string cluster_index = std::to_string(cluster_i);
    writer.write<PointT>(params.output_pcd_filepath + "cloud_cluster_" + cluster_index + ".pcd", *cloud_cluster, false);
    std::cout << "Cluster writing completed." << std::endl;
}

// void PointCloudInterface::visualizeCluster(pcl::PointCloud<PointT>::Ptr point_cloud, Centroids centroids, Parameters &params)
// {
//     pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(point_cloud);
//     viewer->addPointCloud<PointT>(point_cloud, rgb, "Cluster" + std::to_string(cluster_i));
//     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, params.cluster_params.point_size, "Cluster" + std::to_string(cluster_i));
//     viewer->addSphere(centroids, params.cluster_params.sphere_radius, "Centroid" + std::to_string(cluster_i));
// }

// void PointCloudInterface::displayPointCloud()
// {
//     while (!viewer->wasStopped())
//     {
//         viewer->spinOnce();
//     }
// }
