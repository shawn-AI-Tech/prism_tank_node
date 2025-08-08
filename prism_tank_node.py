import numpy as np
from PIL import Image
import torch

class PrismTank:
    """光棱坦克图像混合节点，使用对角线掩码混合两张图像，包含自定义格式转换函数"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "color_level": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_images"
    CATEGORY = "图像混合/特殊效果"

    @staticmethod
    def tensor_to_pil(tensor):
        """
        将ComfyUI的图像张量转换为PIL图像
        ComfyUI张量格式: (1, height, width, 3)，值范围[0, 1]，float32类型
        """
        # 移除批次维度并转换为numpy数组
        arr = tensor.squeeze(0).cpu().numpy()
        # 将值范围从[0,1]转换为[0,255]并转为uint8类型
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        # 转换为PIL图像
        return Image.fromarray(arr)
    
    @staticmethod
    def pil_to_tensor(pil_img):
        """
        将PIL图像转换为ComfyUI的图像张量格式
        输出格式: (height, width, 3)，值范围[0, 1]，float32类型
        """
        # 转换为numpy数组
        arr = np.array(pil_img, dtype=np.float32)
        # 将值范围从[0,255]转换为[0,1]
        arr = arr / 255.0
        # 转换为张量
        return torch.from_numpy(arr)

    def create_diagonal_mask(self, height, width):
        """创建对角线掩码"""
        mask = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if (x % 2 == 0 and y % 2 == 0) or (x % 2 == 1 and y % 2 == 1):
                    mask[y, x] = 1
        return mask
    
    def blend_images(self, image1, image2, color_level=20):
        """混合两张图像"""
        # 使用自定义转换函数将张量转换为PIL图像
        img1 = self.tensor_to_pil(image1)
        img2 = self.tensor_to_pil(image2)
        
        # 转换为numpy数组进行处理
        img1_np = np.array(img1, dtype=np.float32)
        img2_np = np.array(img2, dtype=np.float32)
        
        # 图像预处理
        img1_np = img1_np / (255 / color_level)
        img2_np = img2_np / (255 / (255 - color_level)) + color_level
        
        # 确保两张图片尺寸一致
        if img1_np.shape != img2_np.shape:
            img2 = img2.resize((img1_np.shape[1], img1_np.shape[0]))
            img2_np = np.array(img2, dtype=np.float32)
        
        # 获取图像尺寸并创建掩码
        height, width = img1_np.shape[:2]
        mask = self.create_diagonal_mask(height, width)
        
        # 扩展掩码维度以匹配图像的通道数
        if len(img1_np.shape) == 3:  # 彩色图像
            mask = mask[:, :, np.newaxis]
        
        # 应用掩码进行图像混合
        blended = img1_np * mask + img2_np * (1 - mask)
        
        # 确保像素值在有效范围内，并转换回uint8类型
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # 转换回PIL图像再转为ComfyUI所需的张量格式
        blended_img = Image.fromarray(blended)
        tensor_img = self.pil_to_tensor(blended_img).unsqueeze(0)  # 添加批次维度
        
        return (tensor_img,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "Prism Tank": PrismTank
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Prism Tank": "光棱坦克 (Prism Tank)"
}
    
