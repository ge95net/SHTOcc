
## Abstract
3D occupancy prediction has attracted much attention in the field of autonomous driving due to its powerful geometric perception and object recognition capabilities. However, existing methods have not explored the most essential distribution patterns of voxels, resulting in unsatisfactory results. This paper first explores the inter-class distribution and geometric distribution of voxels, thereby solving the long-tail problem caused by the inter-class distribution and the poor performance caused by the geometric distribution. Specifically, this paper proposes SHTOcc (\textbf{S}parse \textbf{H}ead-\textbf{T}ail Occupancy), which uses sparse head-tail voxel construction to accurately identify and balance key voxels in the head and tail classes, while using decoupled learning to reduce the model's bias towards the dominant (head) category and enhance the focus on the tail class. Experiments show that significant improvements have been made on multiple baselines: SHTOcc reduces GPU memory usage by 42.2\%, increases inference speed by 58.6\%, and improves accuracy by about 7\%, verifying its effectiveness and efficiency.

![image](https://github.com/user-attachments/assets/7e973e8b-1b5e-4ce2-aa49-a60a9053ee0b)


## Results
![image](https://github.com/user-attachments/assets/3f2d2819-ccd8-432b-ac0b-34cd2ba2a0ea)


## Getting Started
Following the ReadMe file in each folder:
- [SHTOcc-SparseOcc](SHTOcc-SparseOcc/README.md)
- [SHTOcc-Symphonies](SHTOcc-Symphonies/README.md)


## Acknowledgement

This project is developed based on the following open-sourced projects: [SparseOcc](https://github.com/VISION-SJTU/SparseOcc), [Symphonies](https://github.com/hustvl/Symphonies/tree/main), [Occformer](https://github.com/zhangyp15/OccFormer), [COTR](https://github.com/NotACracker/COTR?tab=readme-ov-file). Thanks for their excellent work.
