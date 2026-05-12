# -*- coding: utf-8 -*-
"""
CosyVoice TTS 客户端测试脚本
测试 OpenAI 兼容 audio/speech 请求
作者：凌封
来源：https://aibook.ren (AI全书)
"""
import os
import sys
import time
import argparse
import requests

def test_health(base_url: str):
    """测试健康检查接口"""
    print("\n[1] 健康检查")
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code == 200:
            print(f"  ✓ 服务正常: {resp.json()}")
            return True
        else:
            print(f"  ✗ 服务异常: {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ 连接失败: {e}")
        return False


def test_audio_speech(
        base_url: str,
        text: str,
        ref_audio: str,
        ref_text: str,
        response_format: str,
        stream: bool,
        output_path: str
):
    """测试 OpenAI 兼容 TTS 接口"""
    print("\n[2] OpenAI audio/speech 测试")
    print(f"  文本: {text}")
    print(f"  参考音频: {ref_audio}")
    print(f"  参考文本: {ref_text}")
    print(f"  格式: {response_format}")
    print(f"  流式: {stream}")
    
    start_time = time.time()
    first_chunk_time = None
    total_bytes = 0
    
    suffix = f".{response_format}"
    if not output_path.endswith(suffix):
        output_path += suffix
    
    try:
        resp = requests.post(
            f"{base_url}/v1/audio/speech",
            json={
                "model": "tts-1",
                "input": text,
                "task_type": "Base",
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "response_format": response_format,
                "stream": stream
            },
            stream=True,
            timeout=60
        )
        
        if resp.status_code != 200:
            print(f"  ✗ 请求失败: {resp.status_code} - {resp.text}")
            return False
        
        sample_rate = int(resp.headers.get("X-Sample-Rate", 24000))
        channels = int(resp.headers.get("X-Channels", 1))
        bits = int(resp.headers.get("X-Bits", 16))
        
        print(f"  采样率: {sample_rate}Hz, 通道: {channels},位深: {bits}bit")
        
        with open(output_path, "wb") as audio_file:
            for chunk in resp.iter_content(chunk_size=4800):
                if chunk:
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - start_time
                        print(f"  ⚡ 首帧延迟: {first_chunk_time * 1000:.0f}ms")
                    
                    audio_file.write(chunk)
                    total_bytes += len(chunk)
        
        total_time = time.time() - start_time
        
        print(f"  ✓ 接收完成")
        print(f"  ✓ 数据量: {total_bytes / 1024:.1f}KB")
        print(f"  ✓ 总耗时: {total_time:.2f}s")
        print(f"  ✓ 音频已保存: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 请求异常: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="CosyVoice TTS 客户端测试")
    parser.add_argument("--url", type=str, default="http://localhost:10096", help="服务地址")
    parser.add_argument("--text", type=str, default="你好，我是小智，很高兴为您服务。", help="测试文本")
    parser.add_argument("--ref_audio", type=str, default="asset/longyingwan_woman.wav", help="参考音频路径")
    parser.add_argument("--ref_text", type=str, default="我们将为全球城市的可持续发展贡献力量。", help="参考音频对应文本")
    parser.add_argument("--response_format", type=str, default="wav", help="输出格式: wav/mp3/flac/pcm/aac/opus")
    parser.add_argument("--stream", action="store_true", help="流式返回PCM；需要 --response_format pcm")
    parser.add_argument("--output", type=str, default="output/client_test", help="输出文件路径")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CosyVoice TTS 客户端测试")
    print("=" * 60)
    print(f"服务地址: {args.url}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 测试健康检查
    if not test_health(args.url):
        print("\n服务不可用，请先启动服务: ./start_server.sh")
        return
    
    # 测试 OpenAI 兼容 TTS
    test_audio_speech(
        args.url,
        args.text,
        args.ref_audio,
        args.ref_text,
        args.response_format,
        args.stream,
        args.output
    )
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
