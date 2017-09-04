#include <iostream>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/ply_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/correspondence.h>
#include <pcl/features/board.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/features//fpfh_omp.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/features/vfh.h>
#include <pthread.h>
#include "kinect2grabber.h"
/*this header 'kinect2grabber.h' include opencv headers, which has to be
 * placed in the last line, othrewise the compiler will
 * report an error about "unordered_map has no member
 * named 'serialize', the reason seems to be that the opencv
   header leak #define USE_UNORDERED_MAP 1 in case of std ≥ C++11 */

using namespace std;

typedef pcl::PointXYZRGB pointT;
typedef pcl::PointCloud<pointT> pointCloudT;
typedef pointCloudT::Ptr pointCloudPtr;
typedef pcl::Normal normalT;
typedef pcl::PointCloud<normalT> normalCloudT;
typedef pcl::SHOT352 descriptorT;
typedef pcl::FPFHSignature33 featureT;
typedef pcl::PointCloud<featureT> featureCloudT;
typedef featureCloudT::Ptr featureCloudPtr;

/**
 * @brief downSample
 * @param cloud
 * @param leafsize
 * @return
 */
int downSample(pointCloudPtr & cloud, double leafsize)
{
  pcl::VoxelGrid<pointT> grid;
  grid.setInputCloud(cloud);
  grid.setLeafSize(leafsize,leafsize,leafsize);
  grid.filter(*cloud);
  return 1;
}

/**
 * @brief filterCloud--limit the range of point cloud and remove outliers
 * @param cloud
 */
int filterCloud(pointCloudPtr & cloud)
{
  if(cloud->points.size()==0){
      std::cout<<"the input cloud is empty"<<std::endl;
      return -1;
    }

  std::cout<<"the point cloud size is: "<<cloud->points.size()<<std::endl;

  //remove NANs
//  for(pointCloudT::iterator it=cloud->begin();it<cloud->end();++it){
//      if(!pcl::isFinite(*it))
//        cloud->erase(it);
//    }

  //filter by passthrough
  const double zmin=0, zmax=1.2;
  const double xmin=-0.6,xmax=0.6;
  const double ymin=-0.6,ymax=0.6;
  pcl::PassThrough<pointT> pass;
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(zmin,zmax);
  pass.filter(*cloud);

  pass.setInputCloud(cloud);
  pass.setFilterFieldName("x");
  pass.setFilterLimits(xmin,xmax);
  pass.filter(*cloud);

  pass.setInputCloud(cloud);
  pass.setFilterFieldName("y");
  pass.setFilterLimits(ymin,ymax);
  pass.filter(*cloud);

  downSample(cloud,0.002);

  //outlier removal
  pcl::StatisticalOutlierRemoval<pointT> outlierRemv;
  outlierRemv.setInputCloud(cloud);
  outlierRemv.setMeanK(50);
  outlierRemv.setStddevMulThresh(0.5);
  outlierRemv.filter(*cloud);

  std::cout<<"after filtered, the point cloud size is: "<<cloud->points.size()<<std::endl;
  return 1;
}

/**
 * @brief segmentCloud--segment the cloud into clusters,
 *                      the plane will be removed
 * @param cloud
 * @param clusters
 */
int segmentCloud(pointCloudPtr & cloud,vector<pointCloudPtr> & clusters)
{
  //plane segment
  pcl::SACSegmentation<pointT> plane_seg;
  pcl::ModelCoefficients::Ptr coeffPtr(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  plane_seg.setOptimizeCoefficients(true);
  plane_seg.setModelType(pcl::SACMODEL_PLANE);
  plane_seg.setMethodType(pcl::SAC_RANSAC);
  plane_seg.setInputCloud(cloud);
  plane_seg.setDistanceThreshold(0.01);
  plane_seg.segment(*inliers,*coeffPtr);
  if(inliers->indices.size()==0)
    {
      std::cerr<<"WARNING: no plane extracted"<<std::endl;
    }
  else
    std::cout<<"plane extracted, point size: "<<inliers->indices.size()<<std::endl;

  //extract plane and scene-without-plane
  pointCloudT::Ptr scene_no_plane(new pointCloudT);
  pcl::ExtractIndices<pointT> extractor;
  extractor.setInputCloud(cloud);
  extractor.setIndices(inliers);
  extractor.setNegative(true);
  extractor.filter(*scene_no_plane);
  std::cout<<"scene extracted, point size: "<<scene_no_plane->points.size()<<std::endl;

  //euclidean cluster
  pcl::search::KdTree<pointT>::Ptr tree(new pcl::search::KdTree<pointT>);
  tree->setInputCloud(scene_no_plane);
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pointT> clusterExtrac;
  clusterExtrac.setInputCloud(scene_no_plane);
  clusterExtrac.setSearchMethod(tree);
  clusterExtrac.setClusterTolerance(0.02);
  clusterExtrac.setMinClusterSize(300);
  clusterExtrac.setMaxClusterSize(25000);
  clusterExtrac.extract(cluster_indices);
  if(cluster_indices.size()==0)
    {
      std::cerr<<"ERROR: no cluster extracted"<<std::endl;
      return -1;
    }
  else
    std::cout<<"extracted "<<cluster_indices.size()<<" clusters"<<std::endl;

  //extract the clusters
  pcl::ExtractIndices<pointT> extc;
  extc.setInputCloud(scene_no_plane);
  extc.setNegative(false);

  clusters.clear();
  std::vector<pcl::PointIndices>::iterator iter;
  int idx=0;
  for(iter=cluster_indices.begin();iter!=cluster_indices.end();++iter)
    {
      pcl::PointIndices _indices=*iter;
      pcl::PointIndices::Ptr cluster=boost::make_shared<pcl::PointIndices>(_indices);
      //std::cout<<"    cluster #"<<++idx<<" size: "<<cluster->indices.size()<<std::endl;
      pointCloudT::Ptr tmpCloud(new pointCloudT);
      extc.setIndices(cluster);
      extc.filter(*tmpCloud);
      clusters.push_back(tmpCloud);
    }

  return 1;
}

/**
 * @brief calculateModel
 * @param model
 * @param model_descriptors
 * @return
 */
int calculateModel(pointCloudPtr & model,
                   pcl::PointCloud<pcl::Normal>::Ptr model_normals,
                   pcl::PointCloud<descriptorT>::Ptr & model_descriptors)
{
  if(model->points.size()==0){
      std::cout<<"WARNING: model is empty!"<<std::endl;
      return -1;
    }
  //compute normals
  pcl::NormalEstimationOMP<pointT, pcl::Normal> norm_est;
  norm_est.setKSearch (10);
  norm_est.setInputCloud (model);
  norm_est.compute (*model_normals);

  //downsample to get keypoints
  pointCloudT::Ptr model_keypoints(new pointCloudT);
  double sample_leaf=0.004;
  pcl::VoxelGrid<pointT> sampleGrid;
  sampleGrid.setLeafSize(sample_leaf,sample_leaf,sample_leaf);
  sampleGrid.setInputCloud(model);
  sampleGrid.filter(*model_keypoints);
  std::cout << "Model total points: " << model->size ()
            << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

  //compute descriptors
  pcl::SHOTEstimationOMP<pointT, pcl::Normal, descriptorT> descr_est;
  descr_est.setRadiusSearch (0.04);
  descr_est.setInputCloud (model_keypoints);
  descr_est.setInputNormals (model_normals);
  descr_est.setSearchSurface (model);
  descr_est.compute (*model_descriptors);
  std::cout<<"model descriptors size "<<model_descriptors->points.size()<<std::endl;
}

/**
 * @brief findModelSceneCorrespondencs
 * @param model
 * @param model_descriptors
 * @param scene
 * @return
 */
int findModelSceneCorrespondencs(pointCloudPtr& model,
                                 pcl::PointCloud<descriptorT>::Ptr & model_descriptors,
                                 pointCloudPtr& scene)
{
  if(model->points.size()==0 || scene->points.size()==0){
      std::cout<<"WARNING: model or scene is empty!"<<std::endl;
      return 0;
    }
  //compute normals
  pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimationOMP<pointT, pcl::Normal> norm_est;
  norm_est.setKSearch (10);
  norm_est.setInputCloud (scene);
  norm_est.compute (*scene_normals);

  //downsample to get keypoints
  pointCloudT::Ptr scene_keypoints(new pointCloudT);

  double sample_leaf=0.004;
  pcl::VoxelGrid<pointT> sampleGrid;
  sampleGrid.setLeafSize(sample_leaf,sample_leaf,sample_leaf);

  sample_leaf=0.004;
  sampleGrid.setLeafSize(sample_leaf,sample_leaf,sample_leaf);
  sampleGrid.setInputCloud(scene);
  sampleGrid.filter(*scene_keypoints);
  std::cout << "Scene total points: " << scene->size ()
            << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;

  //compute descriptors
  typedef pcl::SHOT352 descriptorT;
  pcl::PointCloud<descriptorT>::Ptr scene_descriptors (new pcl::PointCloud<descriptorT> ());
  pcl::SHOTEstimationOMP<pointT, pcl::Normal, descriptorT> descr_est;
  descr_est.setRadiusSearch (0.01);

  descr_est.setInputCloud (scene_keypoints);
  descr_est.setInputNormals (scene_normals);
  descr_est.setSearchSurface (scene);
  descr_est.compute (*scene_descriptors);
  std::cout<<"scene descriptors size "<<scene_descriptors->points.size()<<std::endl;

  //find model-scene correspondences
  pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

  pcl::KdTreeFLANN<descriptorT> match_search;
  match_search.setInputCloud (model_descriptors);

  /*  For each scene keypoint descriptor, find nearest neighbor into the model
      keypoints descriptor cloud and add it to the correspondences vector.*/
  for (size_t i = 0; i < scene_descriptors->size (); ++i)
  {
    std::vector<int> neigh_indices (1);
    std::vector<float> neigh_sqr_dists (1);
    if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
    {
      continue;
    }
    int found_neighs = match_search.nearestKSearch (
          scene_descriptors->at (i),
          1,
          neigh_indices, neigh_sqr_dists);

    /*  add match only if the squared descriptor distance is less
        than 0.25 (SHOT descriptor distances are between 0 and 1 by design)*/
    if(found_neighs == 1 && neigh_sqr_dists[0] < 0.5f)
    {
      pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      model_scene_corrs->push_back (corr);
    }
  }

  std::cerr << "model--scene Correspondences found: "<<model_scene_corrs->size()<<std::endl;
  return model_scene_corrs->size();
}

/**
 * @brief cloud
 */
sensor::Kinect2Grabber kinect2_grabber;
pointCloudPtr cloud(new pointCloudT);
pointCloudPtr model(new pointCloudT);
pointCloudPtr bestMatchCluster(new pointCloudT);
vector<pointCloudPtr> clusters;
pcl::PointCloud<descriptorT>::Ptr model_descriptors(new pcl::PointCloud<descriptorT>);
pcl::PointCloud<pcl::Normal>::Ptr model_normals(new pcl::PointCloud<pcl::Normal>);
pointCloudPtr icp_aligned_model(new pointCloudT);
pointCloudPtr sac_aligned_model(new pointCloudT);
pointCloudPtr prev_bestMatch_model(new pointCloudT);
Eigen::Matrix4f sac_transform=Eigen::Matrix4Xf::Identity(4,4);
Eigen::Matrix4f icp_transform=Eigen::Matrix4Xf::Identity(4,4);

const string modelFileName="//home//wangy//dev//3dvision_ws//"
                           "projects//dataset//milkbox//model//milkbox.pcd";
bool isUpdateTarget=false;
bool isUpdateScene=false;
bool isClustersUpdated=false;
bool isRegistrationUpdated=false;
bool isExit=false;

/**
 * @brief registerModelToScene
 * @param model
 * @param scene
 * @return
 */
int registerModelToScene(pointCloudPtr & model,
                         pcl::PointCloud<pcl::Normal>::Ptr & model_normals1,
                         pointCloudPtr & scene,
                         Eigen::Matrix4f & sac_transform,
                         Eigen::Matrix4f & icp_transform)
{
  pointCloudPtr sac_aligned_model(new pointCloudT);
  pointCloudPtr icp_aligned_model(new pointCloudT);

  normalCloudT::Ptr scene_normals(new normalCloudT);
  normalCloudT::Ptr model_normals(new normalCloudT);
  pcl::NormalEstimationOMP<pointT,normalT> normal_est;
  pcl::search::KdTree<pointT>::Ptr tree (new pcl::search::KdTree<pointT> ());
  normal_est.setSearchMethod(tree);
  normal_est.setRadiusSearch(0.01);
  normal_est.setInputCloud(scene);
  normal_est.compute(*scene_normals);
  normal_est.setInputCloud(model);
  normal_est.compute(*model_normals);

  //feature estimation
  featureCloudPtr model_features(new featureCloudT);
  featureCloudPtr scene_features(new featureCloudT);

  pcl::console::print_highlight ("Estimating features...\n");
  pcl::FPFHEstimationOMP<pointT,normalT,featureT> fest;
  fest.setRadiusSearch (0.01);
  fest.setInputCloud (model);
  fest.setInputNormals (model_normals);
  fest.compute (*model_features);
  fest.setInputCloud (scene);
  fest.setInputNormals (scene_normals);
  fest.compute (*scene_features);

  // Perform alignment
  pcl::console::print_highlight ("Starting alignment...\n");
  pcl::SampleConsensusPrerejective<pointT,pointT,featureT> align;
  align.setInputSource (model);
  align.setSourceFeatures (model_features);
  align.setInputTarget (scene);
  align.setTargetFeatures (scene_features);
  align.setMaximumIterations (10000); // Number of RANSAC iterations
  align.setNumberOfSamples (3); // Number of points to sample for generating/prerejecting a pose
  align.setCorrespondenceRandomness (5); // Number of nearest features to use
  align.setSimilarityThreshold (0.45f); // Polygonal edge length similarity threshold
  align.setMaxCorrespondenceDistance (0.6); // Inlier threshold
  align.setInlierFraction (0.15f); // Required inlier fraction for accepting a pose hypothesis
  align.align (*sac_aligned_model);

  if (align.hasConverged ())
  {
    sac_transform=align.getFinalTransformation();
    // Print results
    printf ("\n");
    Eigen::Matrix4f transformation = align.getFinalTransformation ();
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
    pcl::console::print_info ("\n");

  }
  else
  {
    pcl::console::print_error ("Alignment has not converged!\n");
  }


  //icp
  pcl::IterativeClosestPoint<pointT,pointT> icp;
  icp.setInputSource(sac_aligned_model);
  icp.setInputTarget(scene);
  icp.setMaxCorrespondenceDistance(0.6);
  icp.setTransformationEpsilon(1e-10);
  icp.setEuclideanFitnessEpsilon(0.005);
  icp.setMaximumIterations(10000);
  icp.setUseReciprocalCorrespondences(true);
  icp.align(*icp_aligned_model);

  if(icp.hasConverged()){
      //prev_bestMatch_model=icp_aligned_model;
      icp_transform=icp.getFinalTransformation();
      std::cout<<"the score of icp estimation is : "<<icp.getFitnessScore()<<std::endl;
      std::cout<<"is icp converged: "<<icp.hasConverged()<<std::endl;
      std::cout<<"icp transform matrix:"<<std::endl<<icp.getFinalTransformation()<<std::endl;
    }
  else{
      pcl::console::print_error("ICP Alignment stopped without convergence!\n");
    }

  return 1;
}


/**
 * @brief threadFunc
 * @return
 */
void * detectThreadFunc(void*)
{
  while(!isExit){
      if(!isClustersUpdated)
        continue;

      //find correspondences between model and object in the scene clusters
      map<int,int,greater<int>> corresRecord;
      for(int i=0;i<clusters.size();++i){
          pointCloudPtr scene_obj=clusters.at(i);
          int corrNum=findModelSceneCorrespondencs(model,model_descriptors,scene_obj);
          corresRecord.insert(make_pair(corrNum,i));
          std::cout<<"corresRecord add: "<<i<<"  "<<corrNum<<std::endl;
        }

      if(corresRecord.size()==0){
          std::cout<<"WARNING: no correspondence found, exit"<<std::endl;
          continue;
        }

      map<int,int>::iterator it=corresRecord.begin();
      int best_corr_index=((pair<int,int>)(*it)).second;
      int best_corr=((pair<int,int>)(*it)).first;
      std::cerr<<"the best correspondence: #"<<best_corr_index
              <<" corres: "<<best_corr<<std::endl;
      bestMatchCluster=clusters.at(best_corr_index);
      isUpdateTarget=true;
    }
}

/**
 * @brief getCloudThreadFunc
 * @return
 */
void * getCloudThreadFunc(void*)
{
  while(!isExit){
      kinect2_grabber.getPointCloud(cloud);
      filterCloud(cloud);
      isUpdateScene=true;

      segmentCloud(cloud,clusters);
      if(clusters.size()==0){
          continue;
        }

      isClustersUpdated=true;
    }
}

/**
 * @brief registerThreadFunc
 * @return
 */
void * registerThreadFunc(void*)
{
  while(!isExit){
      if(isUpdateTarget){
          registerModelToScene(model,model_normals,bestMatchCluster,sac_transform,icp_transform);
          isRegistrationUpdated=true;
        }
    }
}

int main()
{
  //initialize the kinect2 sensor
  std::cout << "getting cloud..." << std::endl;
  kinect2_grabber.start();
  kinect2_grabber.getPointCloud(cloud);
  filterCloud(cloud);

  //load model clouds--currently a milkbox
  if(pcl::io::loadPCDFile(modelFileName,*model)==-1){
      std::cerr<<"load model cloud error"<<std::endl;
      return -1;
    }
  std::cout<<"load model, size: "<<model->points.size()<<std::endl;

  calculateModel(model,model_normals, model_descriptors);

  //down sample
  downSample(model,0.01);
  std::cout<<"downsample model, size: "<<model->points.size()<<std::endl;

  //define viewers
  pcl::visualization::PCLVisualizer viewer("point cloud from kinect2");
  pcl::visualization::PointCloudColorHandlerCustom<pointT> colorH(model,200,10,10);
  pcl::visualization::PointCloudColorHandlerCustom<pointT> target_colorH(model,10,200,10);
  viewer.addPointCloud(cloud,"scene_cloud");
  pcl::visualization::PCLVisualizer viewer2("matched target cloud");
  viewer2.addPointCloud(model,colorH,"model_cloud");
  pcl::transformPointCloud(*model,*sac_aligned_model,sac_transform);
  viewer2.addPointCloud(sac_aligned_model,target_colorH,"sac_registered_model");
  //viewer2.addPointCloud(icp_aligned_model,target_colorH,"icp_registered_model");
  pcl::visualization::PCLVisualizer viewer3("register results");
  //viewer3.addPointCloud(icp_aligned_model,target_colorH,"icp_registered_model");
  viewer3.addPointCloud(sac_aligned_model,target_colorH,"registered_model");

  pthread_t getCloudThread_id;
  int ret2=pthread_create(&getCloudThread_id,NULL,getCloudThreadFunc,NULL);
  if(ret2!=0){
    std::cerr<<"create thread "<<getCloudThread_id<<" failed!"<<std::endl;
    return -1;
  }
  pthread_detach(getCloudThread_id);

  pthread_t detectThread_id;
  int ret1=pthread_create(&detectThread_id,NULL,detectThreadFunc,NULL);
  if(ret1!=0){
    std::cerr<<"create thread "<<detectThread_id<<" failed!"<<std::endl;
    return -1;
  }
  pthread_detach(detectThread_id);

  pthread_t registerThread_id;
  int ret3=pthread_create(&registerThread_id,NULL,registerThreadFunc,NULL);
  if(ret3!=0){
    std::cerr<<"create thread "<<registerThread_id<<" failed!"<<std::endl;
    return -1;
  }
  pthread_detach(registerThread_id);

  //main loop
  while(!viewer.wasStopped()){
      if(isUpdateScene){
          viewer.updatePointCloud(cloud,"scene_cloud");
          isUpdateScene=false;
        }

      if(isUpdateTarget){
          viewer2.removePointCloud("target");
          viewer2.addPointCloud(bestMatchCluster,"target");

          //sac_transform=Eigen::Matrix4Xf::Identity(4,4);
          Eigen::Matrix4f transform=sac_transform*icp_transform;
          pcl::transformPointCloud(*model,*sac_aligned_model,transform);
          viewer2.removePointCloud("sac_registered_model");
          viewer3.removePointCloud("sac_registered_model");
          viewer2.addPointCloud(sac_aligned_model,target_colorH,"sac_registered_model");
          viewer3.addPointCloud(sac_aligned_model,target_colorH,"sac_registered_model");

          isUpdateTarget=false;
        }
      if(isRegistrationUpdated){
//          pcl::transformPointCloud(*model,*sac_aligned_model,sac_transform);
//          viewer2.removePointCloud("sac_registered_model");
//          viewer3.removePointCloud("sac_registered_model");
//          viewer2.addPointCloud(sac_aligned_model,"sac_registered_model");
//          viewer3.addPointCloud(sac_aligned_model,"sac_registered_model");
          isRegistrationUpdated=false;
        }

//      std::stringstream ss;
//      std::string cloud_name;
//      if(isClustersUpdated){
//          viewer3.removeAllPointClouds();
//          for(int i=0;i<clusters.size();++i){
//              ss.clear();
//              ss<<"cluster_"<<i;
//              ss>>cloud_name;
//              viewer3.addPointCloud(clusters.at(i),cloud_name);
//            }
//        }

      viewer.spinOnce();
      viewer2.spinOnce();
      viewer3.spinOnce();
    }

  return 0;
}
