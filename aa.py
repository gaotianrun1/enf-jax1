import torch
# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA is available.")
    print(f"CUDA version: {torch.version.cuda}")
    # 可以进一步创建一个张量并将其移动到CUDA设备上测试
    x = torch.tensor([1.0, 2.0]).to('cuda')
    print(x)
else:
    print("CUDA is not available.")