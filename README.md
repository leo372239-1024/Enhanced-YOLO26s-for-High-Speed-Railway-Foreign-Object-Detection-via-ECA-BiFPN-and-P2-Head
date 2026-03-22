# Enhanced-YOLO26s-for-High-Speed-Railway-Foreign-Object-Detection-via-ECA-BiFPN-and-P2-Head
项目基于 YOLO26s 的改进版本，数据集用的是 RailFOD23 铁路异物检测数据集 。模型创新技术架构是通过引入 ECA 注意力机制、BiFPN 颈部网络 以及 P2 高分辨率检测头，提升YOLO26s模型对铁路场景中微小异物（如塑料袋、气球、鸟巢等）的检测性能，改进模型相较于Baseline，Precision提升3.47%，Recall 提升2.2%，F1提升0.6%，mAP50-95 提升2.2%，达到了77.1%。
   - 项目名称：基于ECA注意力机制、BiFPN特征融合与P2高分辨率检测头改进的YOLO26s高铁异物目标检测：
- 项目简介：项目基于 YOLO26s 的改进版本，数据集用的是 RailFOD23 铁路异物检测数据集 。模型创新技术架构是通过引入 ECA 注意力机制、BiFPN 颈部网络 以及 P2 高分辨率检测头，提升YOLO26s模型对铁路场景中微小异物（如塑料袋、气球、鸟巢等）的检测性能，改进模型相较于Baseline，Precision提升3.47%，Recall 提升2.2%，F1提升0.6%，mAP50-95 提升2.2%，达到了77.1%。


- 模型工具包软件版本：ultralytics               8.3.55
- 模型名称：YOLO26s
- 数据集：RailFOD23 铁路异物检测数据集
- 硬件：intel i5-12500H Intel(R) Iris(R) Xe Graphics  NVIDIA GeForce RTX 3050 Laptop GPU  
  领域：目标检测
  AI模型训练流程：
- 准备数据集：RailFOD23 铁路异物检测数据集
- 训练与验证模型（train,vaildation）：
- 预测/推理（predict）:
- 导出部署（export）：

YOLO模型

- 操作模式mode
  train: 训练模型。从头开始训练或使用预训练权重进行微调。需要指定数据集配置文件（yaml）。 | 自定义数据集训练、迁移学习。 |
  val:验证模型。在验证集上评估已训练模型的性能（计算 mAP, precision, recall 等指标）。 | 模型调优、监控过拟合、选择最佳checkpoint。 |
  predict:推理/预测。使用模型对新的图像、视频或流数据进行预测。这是部署阶段最常用的模式。 | 实时检测、批量处理图片、视频分析。 |
  export:将训练好的 PyTorch 模型 (.pt) 转换为其他格式以便部署。 | 部署到边缘设备、Web端、移动端。支持 ONNX, TensorRT, CoreML,       OpenVINO, Paddle 等格式。 |
  track:目标跟踪。在视频流中进行预测并关联帧之间的对象，赋予每个对象唯一的 ID。 | 交通流量统计、行为分析、多目标跟踪。 |
- 任务类型task
  Detect目标检测:识别图像中的物体类别，并用水平边界框 (Horizontal Bounding Box) 框出位置。
  Segment实例分割：不仅检测物体，还生成每个物体的像素级掩膜 (Mask)，区分同一类别的不同个体。
  Classify图像分类:判断整张图像属于哪个类别（不包含位置信息）
  Pose姿态估计:检测物体的关键点（如人的关节：眼、肩、肘、膝等），并描绘骨架。
  OBB Oriented Bounding Boxes 旋转目标检测:针对细长或不规则排列的物体，使用旋转边界框进行更紧密的贴合（包含角度信息）。
- 使用方法: yolo mode='predict' task='detect' model='yolo11n.pt' source='77.png' device='cuda'
  #task省略的话默认就是detect


数据集结构：
_datasets
    _icon
        __images#存放图片
            train#训练集注意要和labels里的对应
            val#验证集注意要和labels里的对应
       _labels#存放标注信息
            train#注意要和images里的对应
            val#注意要和images里的对应


- 性能指标
  - Precision (P) = 正确检测目标数 / 检测目标数   =检测到的目标的准确率
  - Recall (R)   =正确检测目标数 / 实际目标数   =正确检测率
  - AP (Average Precision): 平均精度.PR 曲线下面的面积就是
  - mAP:所有类别平均就是mAP
  - mAP50     ：  loU=0.5时平均精度
  - mAP50-95   ：   COCO标准平均精度
  - F1 ： F1 = 2 * P * R / (P + R)，综合P和R给出的分数

- 性能图表
  - Precision-Recall 曲线：PR_curve.png，曲线越靠 右上角越好
  - Precision-Confidence 曲线:P_curve.png，
    Precision
    1.0 |                    ________
    |                 __/
    0.9 |              __/
    |           __/
    0.8 |        __/
    |     __/
    0.7 |__/
    |
    +---------------------------
    0   0.2   0.4   0.6   0.8   1
           Confidence
  - Recall-Confidence 曲线：R_curve.png，
    Recall
    1.0 |_________
    |        \
    0.9 |         \
    |          \
    0.8 |           \
    |            \
    0.7 |             \__
    |
    +---------------------------
    0   0.2   0.4   0.6   0.8   1
           Confidence
  - F1-Confidence 曲线：F1_curve.png，
    F1
    1.0 |
    |         /\
    0.9 |        /  \
    |       /    \
    0.8 |______/      \____
    |
    +---------------------------
    0   0.2   0.4   0.6   0.8   1
           Confidence
  - 混淆矩阵图：confusion_matrix.png，理想情况为对角线
        Predicted
    A   B   C
    True   A     ███
    B         ███
    C             ███
  - 归一化混淆矩阵：confusion_matrix_normalized.png，每一行加起来 = 1，更容易看分类错误比例

结果保存：

1. Baseline 模型 (官方 YOLO26s)
   最佳权重：runs/baseline/weights/best.pt
   最后一次权重：runs/baseline/weights/last.pt
2. 改进模型 (YOLO26s + ECA + BiFPN + P2)
   最佳权重：runs/improved/weights/best.pt
   最后一次权重：runs/improved/weights/last.pt
3. 测试结果 (指标对比与可视化)
   测试评估结果：分别保存在 runs/baseline_test/ 和 runs/improved_test/ 目录下。
   4.汇总对比结果：程序运行结束后，会自动将关键指标和论文图表提取并汇总到根目录下的 results/ 文件夹中（包含 metrics.csv、metrics.xlsx 和 plots/ 图表）。
