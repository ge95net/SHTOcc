import torch
from collections import defaultdict
from prettytable import PrettyTable
import timm
def compare_keys(model, weight_path):
    # 加载模型和权重文件
    model_state = model.state_dict()
    try:
        weights = torch.load(weight_path, map_location='cpu')
    except Exception as e:
        print(f"加载权重文件失败: {e}")
        return
   
    for key, value in weights.items():
        # 删除前缀 "models."
        new_key = key.replace('net.', '', 1)  # 只替换第一个出现的 "models."
        new_weights[new_key] = value
    # 提取键名和形状
    weights = new_weights
    model_keys = set(model_state.keys())
    weight_keys = set(weights.keys())
    
    # 分类键名差异
    key_diff = defaultdict(list)
    key_diff["Matched"] = list(model_keys & weight_keys)
    key_diff["Missing in Weights"] = list(model_keys - weight_keys)
    key_diff["Extra in Weights"] = list(weight_keys - model_keys)
    
    # 检查匹配键的形状是否一致
    shape_mismatch = []
    for key in key_diff["Matched"]:
        if model_state[key].shape != weights[key].shape:
            shape_mismatch.append(key)
    
    # 输出结果
    table = PrettyTable()
    table.field_names = ["Category", "Keys"]
    table.align["Category"] = "l"
    table.align["Keys"] = "l"
    
    for category, keys in key_diff.items():
        if keys:
            table.add_row([category, "\n".join(keys)])
    
    print(table)
    
    if shape_mismatch:
        print("\n[Warning] 以下键名匹配但形状不一致:")
        for key in shape_mismatch:
            print(f"  - {key}: Model Shape {model_state[key].shape} vs Weight Shape {weights[key].shape}")
    else:
        print("\n[OK] 所有匹配键的形状一致!")

# 使用示例
if __name__ == "__main__":
    from timm import create_model
    from ssc_pl.models.Lseg import LSegNet
    # 初始化模型（以 vit_large_patch16_384 为例）
    # model = create_model("vit_large_patch16_384", pretrained=False)
    
    # # 指定权重文件路径
    # weight_path = "/ailab/user/wangxuhong/yuqiucheng/Symphonies/ckpts/pytorch_model.bin"

    labels = ['empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist',
            'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence',
            'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign']
    model = LSegNet(
        labels=labels,
        backbone='clip_vitl16_384',
        features=256,
        crop_size=480,
        arch_option=0,
        block_depth=0,
        activation='lrelu')

    weight_path = 'ckpts/demo_e200.ckpt'
    
    # 执行比对
    compare_keys(model, weight_path)

    