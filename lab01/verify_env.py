# -*- coding: utf-8 -*-
# 环境验证脚本(实验指南 2.2 节第(4)步要求)
import torch, sklearn, matplotlib, numpy

print("Python 环境: Anaconda 虚拟环境 llm_course")
print("PyTorch:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())  # 无 GPU 显示 False 也可正常完成本实验
print("scikit-learn:", sklearn.__version__)
print("NumPy:", numpy.__version__)
print("Matplotlib:", matplotlib.__version__)
