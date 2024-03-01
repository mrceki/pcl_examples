#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/ml/kmeans.h>
#include <pcl/common/impl/angles.hpp>
#include <yaml-cpp/yaml.h>

float calculateDistance(float p1, float p2)
{
    return sqrt(pow(p1, 2) + pow(p2, 2));
}

class PointCloudAnalyzer
{
public:
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster;
    pcl::Kmeans::Centroids centroids;
    std::vector<pcl::PointIndices> cluster_indices;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
    int cluster_i = 0;
    int centroid_i = 0;
    PointCloudAnalyzer() : cloud(new pcl::PointCloud<pcl::PointXYZ>),
                           cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>),
                           coefficients(new pcl::ModelCoefficients),
                           inliers(new pcl::PointIndices),
                           tree(new pcl::search::KdTree<pcl::PointXYZ>),
                           viewer(new pcl::visualization::PCLVisualizer("Cluster Viewer"))
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
        pcl::PointXYZ min_point_AABB;
        pcl::PointXYZ max_point_AABB;
        pcl::PointXYZ min_point_OBB;
        pcl::PointXYZ max_point_OBB;
        pcl::PointXYZ position_OBB;
        Eigen::Matrix3f rotational_matrix_OBB;
        float major_value, middle_value, minor_value;
        Eigen::Vector3f major_vector, middle_vector, minor_vector;
        Eigen::Vector3f mass_center;
    };

    struct PassThroughFilterParams
    {
        std::vector<FilterField> filter_fields;
    };

    struct Parameters
    {
        std::string pcd_filepath, output_pcd_filepath;
        SACParams sac_params;
        SACParams cluster_sac_params;
        EuclideanClusterParams ec_params;
        PassThroughFilterParams passthrough_filter_params;
        MomentOfInertiaParams moment_of_inertia_params;
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
        seg.setAxis(Eigen::Vector3f(params.normal_axis[0], params.normal_axis[1], params.normal_axis[2]));
        seg.setEpsAngle(pcl::deg2rad(params.angle_threshold));

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

    void momentOfInertia(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, MomentOfInertiaParams &params)
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

    void filterCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, PassThroughFilterParams params)
    {
        for (const auto &filterField : params.filter_fields)
        {
            pass.setInputCloud(point_cloud);
            pass.setFilterFieldName(filterField.field);
            pass.setFilterLimits(filterField.limit_min, filterField.limit_max);
            pass.filter(*point_cloud);
        }
    }

    void visualizeCluster(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, pcl::Kmeans::Centroids centroids, Parameters &params)
    {
        for (int i = 0; i < centroids.size(); i++)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            cluster->points.push_back(pcl::PointXYZ(centroids[i][0], centroids[i][1], centroids[i][2]));
            std::stringstream ss;
            ss << "centroid_" << centroid_i;
            centroid_i++;
            viewer->addPointCloud<pcl::PointXYZ>(cluster, ss.str());
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, params.visualization_point_size * 3, ss.str());
        }
        std::stringstream ss;
        ss << "cluster_" << cluster_i;
        viewer->addPointCloud<pcl::PointXYZ>(point_cloud, ss.str());
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, params.visualization_point_size, ss.str());
        visualiseMomentOfInertia(params.moment_of_inertia_params);
        cluster_i++;
        viewer->spin();
        std::cout << "Cluster visualization completed." << std::endl;
    }

    void visualiseMomentOfInertia(MomentOfInertiaParams p)
    {
        viewer->setBackgroundColor(0, 0, 0);
        viewer->addCoordinateSystem(1.0);
        viewer->initCameraParameters();

        std::stringstream ss;
        ss << "AABB_" << cluster_i;
        viewer->addCube(p.min_point_AABB.x, p.max_point_AABB.x, p.min_point_AABB.y, p.max_point_AABB.y, p.min_point_AABB.z, p.max_point_AABB.z, 1.0, 1.0, 0.0, ss.str());
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, ss.str());

        ss << "OBB_" << cluster_i;
        Eigen::Vector3f position(p.position_OBB.x, p.position_OBB.y, p.position_OBB.z);
        Eigen::Quaternionf quat(p.rotational_matrix_OBB);
        viewer->addCube(position, quat, p.max_point_OBB.x - p.min_point_OBB.x, p.max_point_OBB.y - p.min_point_OBB.y, p.max_point_OBB.z - p.min_point_OBB.z, ss.str());
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, ss.str());

        pcl::PointXYZ center(p.mass_center(0), p.mass_center(1), p.mass_center(2));
        pcl::PointXYZ x_axis(p.major_vector(0) + p.mass_center(0), p.major_vector(1) + p.mass_center(1), p.major_vector(2) + p.mass_center(2));
        pcl::PointXYZ y_axis(p.middle_vector(0) + p.mass_center(0), p.middle_vector(1) + p.mass_center(1), p.middle_vector(2) + p.mass_center(2));
        pcl::PointXYZ z_axis(p.minor_vector(0) + p.mass_center(0), p.minor_vector(1) + p.mass_center(1), p.minor_vector(2) + p.mass_center(2));

        std::stringstream ss_major;
        ss_major << "major_eigen_vector_" << cluster_i;
        std::stringstream ss_middle;
        ss_middle << "middle_eigen_vector_" << cluster_i;
        std::stringstream ss_minor;
        ss_minor << "minor_eigen_vector_" << cluster_i;

        float x_dimension = calculateDistance(p.max_point_AABB.x - p.min_point_AABB.x, p.max_point_AABB.y - p.min_point_AABB.y);
        float y_dimension =  p.max_point_AABB.z - p.min_point_AABB.z;
        std::cout << "max_points_aabb: " << p.max_point_AABB << ", min_points_aabb: " << p.min_point_AABB << std::endl;

        std::cout << "x_dimension: " << x_dimension << std::endl;
        std::cout << "y_dimension: " << y_dimension << std::endl;
        viewer->addLine(center, x_axis, 1.0f, 0.0f, 0.0f, ss_major.str());
        viewer->addLine(center, y_axis, 0.0f, 1.0f, 0.0f, ss_middle.str());
        viewer->addLine(center, z_axis, 0.0f, 0.0f, 1.0f, ss_minor.str());
    }

    void writeClusters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster, Parameters &params)
    {
        cloud_cluster->width = cloud_cluster->size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
    
        std::string cluster_index = std::to_string(cluster_i);
        writer.write<pcl::PointXYZ>(params.output_pcd_filepath + "cloud_cluster_" + cluster_index + ".pcd", *cloud_cluster, false);
        std::cout << "Cluster writing completed." << std::endl;
    }

    void setParametersFromYAML(PointCloudAnalyzer::Parameters &params,
                               const std::string &yaml_file)
    {
        YAML::Node config = YAML::LoadFile(yaml_file);
        YAML::Node filterFieldsNode = config["filter_fields"];

        params.pcd_filepath = config["pcd_filepath"].as<std::string>();
        params.output_pcd_filepath = config["output_pcd_filepath"].as<std::string>();

        for (const auto& fieldNode : filterFieldsNode)
        {
            PointCloudAnalyzer::FilterField filterField;
            filterField.field = fieldNode["name"].as<std::string>();
            filterField.limit_min = fieldNode["limit_min"].as<float>();
            filterField.limit_max = fieldNode["limit_max"].as<float>();

            params.passthrough_filter_params.filter_fields.push_back(filterField);
        }

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
    }

private:
    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PCDWriter writer;
    pcl::ModelCoefficients::Ptr coefficients;
    pcl::PointIndices::Ptr inliers;
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::visualization::PCLVisualizer::Ptr viewer;
    pcl::PassThrough<pcl::PointXYZ> pass;
    pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
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

    pcl.filterCloud(pcl.cloud, params.passthrough_filter_params);

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
        pcl.momentOfInertia(cluster, params.moment_of_inertia_params);
        pcl.writeClusters(cluster, params);
        pcl.visualizeCluster(cluster, pcl.centroids, params);
    }

    return 0;
}