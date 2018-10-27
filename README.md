# milkbox
***Milkbox*** is an object alignment demo using pcl standard pipeline; 

Assume that we have a known object CAD model(e.g. a milkbox), and its point cloud gathered by some kind of RGBD sensor, we can pick out the known object and estimate its transformation matrix through this demo.

However, the implementation of this demo may be naive somehow, and its purpose is just to verify the standard pipeline. To run the demo, a kinect v2 is needed as depth sensor, a real milkbox and a CAD model of milkbox is needed (a simplest milkbox CAD model is just a cylinder with proper size). You can also substitute the milkbox with some other object, what you should prepare is just the real object and its CAD model.

![](https://github.com/windz0629/milkbox/blob/master/milkbox.png)

### Pipeline

**1) Basic preprocess:** `grab pointCloud` --> `filter` --> `downsample` 

**2) Find best match cluster:** --> `compute normals` --> `downsample` --> `compute SHOT352 descriptors` --> `find model-scene correspondences`

**3) align model to pointCloud:** --> `FPFH estimation` --> `SAC alignment` --> `ICP alignment` --> `get transformation`

### Compile & run

#### Dependencies

* PCL 1.8
* VTK
* Eigen3
* freenect2
* Boost
* CUDA
* Opencv

#### Compile

```
git clone https://github.com/windz0629/milkbox.git
cd milkbox
mkdir build && cd build
cmake ..
make
```

#### run the executable

```
./milkbox
```