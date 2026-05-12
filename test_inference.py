# -*- coding: utf-8 -*-
"""
Fun-CosyVoice3-0.5B-2512 本地推理测试
验证模型加载和 GPU 显存占用
作者：凌封
来源：https://aibook.ren (AI全书)
"""
import os
import sys
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'third_party', 'Matcha-TTS'))

import torch
import torchaudio
import numpy as np



# 设置 cuDNN 库路径 (ONNX Runtime GPU 加速)
try:
    import nvidia.cudnn
    cudnn_lib = os.path.join(nvidia.cudnn.__path__[0], 'lib')
    if os.path.isdir(cudnn_lib):
        os.environ['LD_LIBRARY_PATH'] = f"{cudnn_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
except ImportError:
    pass

def main():
    parser = argparse.ArgumentParser(description="CosyVoice 推理测试")
    parser.add_argument("--use_vllm", action="store_true", help="使用 vLLM 加速")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Fun-CosyVoice3-0.5B-2512 本地推理测试")
    print("=" * 60)
    
    # 1. 检查 GPU
    print("\n[1/4] 检查 GPU 环境")
    if torch.cuda.is_available():
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ CUDA 版本: {torch.version.cuda}")
        print(f"  ✓ 初始显存占用: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
    else:
        print("  ⚠ 未检测到 GPU，将使用 CPU 推理 (速度较慢)")
    
    # 2. 加载模型
    print("\n[2/4] 加载模型")
    model_dir = os.path.join(SCRIPT_DIR, "models", "Fun-CosyVoice3-0.5B")
    
    if not os.path.exists(os.path.join(model_dir, "cosyvoice3.yaml")):
        print(f"  ✗ 模型未找到: {model_dir}")
        print("  请先运行: python download_model.py")
        return
    
    from cosyvoice.cli.cosyvoice import AutoModel
    
    start_time = time.time()
    print(f"  > vLLM 加速: {'已启用 (utilization=0.2)' if args.use_vllm else '未启用'}")
    
    try:
        cosyvoice = AutoModel(model_dir=model_dir, load_vllm=args.use_vllm)
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        if args.use_vllm:
             print("  请检查是否已安装 vllm: pip install vllm==0.9.0")
        return

    load_time = time.time() - start_time
    
    print(f"  ✓ 模型加载完成，耗时: {load_time:.1f}s")
    print(f"  ✓ 采样率: {cosyvoice.sample_rate}Hz")
    
    if torch.cuda.is_available():
        print(f"  ✓ 模型加载后显存占用: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
    
    # 3. 检查参考音频
    print("\n[3/4] 加载参考音频")
    prompt_wav_path = os.path.join(SCRIPT_DIR, "official", "asset", "zero_shot_prompt.wav")
    
    if not os.path.exists(prompt_wav_path):
        prompt_wav_path = os.path.join(SCRIPT_DIR, "asset", "prompt.wav")
    
    if os.path.exists(prompt_wav_path):
        print(f"  ✓ 参考音频: {prompt_wav_path}")
    else:
        print(f"  ⚠ 未找到参考音频，跳过推理测试")
        return
    
    # CosyVoice3 需要 "You are a helpful assistant.<|endofprompt|>" 前缀
    prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
    
    # 4. 推理测试
    print("\n[4/4] 推理测试")
    
    test_sentences = [
        "你好，我是小智，你的智能助手",
        "有啥需要我帮忙的吗？",
        "浙江省的省会是杭州市。",
        "看来你有别的事情要忙，我先走啦，需要我时再呼唤我哦"
    ]

    output_dir = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    # [重要优化] 预先计算并缓存音色特征
    # 这一步会执行 load_wav, extract_feat 等耗时操作，并将结果存入 spk2info
    print(f"  正在预计算并缓存参考音频特征 (ID: test_user)...")
    cosyvoice.add_zero_shot_spk(prompt_text, prompt_wav_path, "test_user")
    
    for idx, test_text in enumerate(test_sentences):
        print(f"\n[{idx+1}/{len(test_sentences)}] 测试文本: {test_text}")
        print("-" * 50)
        
        start_time = time.time()
        first_chunk_time = None
        total_samples = 0
        audio_chunks = []
        
        # [优化] 使用 zero_shot_spk_id 调用，直接使用缓存特征，跳过 I/O 和特征提取
        for i, result in enumerate(cosyvoice.inference_zero_shot(
            test_text, 
            prompt_text, 
            prompt_wav_path, # 此参数将被忽略
            stream=True,
            zero_shot_spk_id="test_user"
        )):
            if first_chunk_time is None:
                first_chunk_time = time.time() - start_time
            
            audio_tensor = result['tts_speech']
            audio_chunks.append(audio_tensor)
            total_samples += audio_tensor.shape[-1]
            
            # [模拟服务端优化] GPU PCM 转换
            _ = (audio_tensor * 32768).to(torch.int16).cpu().numpy().tobytes()
        
        total_time = time.time() - start_time
        audio_duration = total_samples / cosyvoice.sample_rate
        rtf = total_time / audio_duration if audio_duration > 0 else 0
        
        # 打印结果
        print(f"  ⚡ 首帧延迟: \033[1;32m{first_chunk_time * 1000:.0f} ms\033[0m" if first_chunk_time else "  ⚡ 首帧延迟: N/A")
        print(f"  ⏱️  总耗时:   {total_time:.3f} s")
        print(f"  🎵 音频时长: {audio_duration:.2f} s")
        print(f"  🚀 RTF:      {rtf:.3f}")
        
        # 保存音频
        if audio_chunks:
            # import torch # 已在全局导入
            full_audio = torch.cat(audio_chunks, dim=-1)
            filename = f"test_output_{idx+1}.wav"
            output_path = os.path.join(output_dir, filename)
            torchaudio.save(output_path, full_audio, cosyvoice.sample_rate)
            print(f"  💾 已保存:   {filename}")
    
    if torch.cuda.is_available():
        print(f"\n  ✓ 推理后显存占用: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
        print(f"  ✓ 显存峰值: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f}GB")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
