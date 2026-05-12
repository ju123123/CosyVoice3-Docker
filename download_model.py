# -*- coding: utf-8 -*-
"""
Fun-CosyVoice3-0.5B-2512 模型下载脚本
作者：凌封
来源：https://aibook.ren (AI全书)
"""
import os
import sys

def main():
    """下载 Fun-CosyVoice3-0.5B-2512 模型"""
    from modelscope import snapshot_download
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    
    print("=" * 50)
    print("Fun-CosyVoice3-0.5B-2512 模型下载")
    print("=" * 50)
    
    # 1. 下载主模型
    model_id = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    local_dir = os.path.join(models_dir, "Fun-CosyVoice3-0.5B")
    
    if os.path.exists(os.path.join(local_dir, "cosyvoice3.yaml")):
        print(f"✓ 主模型已存在: {local_dir}")
    else:
        print(f"正在下载主模型: {model_id}")
        print(f"保存路径: {local_dir}")
        snapshot_download(model_id, local_dir=local_dir)
        print(f"✓ 主模型下载完成")
    
    # 2. 下载 ttsfrd 模型 (可选，用于更好的文本标准化)
    ttsfrd_id = "iic/CosyVoice-ttsfrd"
    ttsfrd_dir = os.path.join(models_dir, "CosyVoice-ttsfrd")
    
    print(f"\n[可选] 文本标准化模型: {ttsfrd_id}")
    print("该模型可提升文本处理效果，但非必需。")
    
    try:
        user_input = input("是否下载? [y/N]: ").strip().lower()
        if user_input == 'y':
            if os.path.exists(os.path.join(ttsfrd_dir, "resource.zip")):
                print(f"✓ ttsfrd 模型已存在: {ttsfrd_dir}")
            else:
                print(f"正在下载: {ttsfrd_id}")
                snapshot_download(ttsfrd_id, local_dir=ttsfrd_dir)
                print(f"✓ ttsfrd 模型下载完成")
                print("\n提示: 如需启用 ttsfrd，请手动执行:")
                print(f"  cd {ttsfrd_dir}")
                print("  unzip resource.zip -d .")
                print("  pip install ttsfrd_dependency-0.1-py3-none-any.whl")
                print("  pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl")
        else:
            print("跳过 ttsfrd 下载 (将使用 wetext 作为替代)")
    except EOFError:
        # 非交互模式，跳过可选下载
        print("非交互模式，跳过可选模型下载")
    
    print("\n" + "=" * 50)
    print("模型下载完成！")
    print("=" * 50)
    print(f"\n模型路径: {local_dir}")
    print("下一步: ./start_server.sh")


if __name__ == "__main__":
    main()
