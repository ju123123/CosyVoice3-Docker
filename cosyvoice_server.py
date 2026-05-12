# -*- coding: utf-8 -*-
import argparse
import logging
import time
import numpy as np
import io
import wave
import base64
import tempfile
import shutil
from urllib.parse import urlparse, unquote
from urllib.request import urlopen, Request
from typing import Generator, Optional
import threading
import torch
import torchaudio
from pydantic import BaseModel, Field, root_validator
import os
import sys

# 设置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'third_party', 'Matcha-TTS'))

from fastapi import FastAPI, HTTPException, UploadFile, Form, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# 多音色配置 (用户可修改此部分)
# ============================================================================
# 格式: {"id": "音色ID", "file": "文件名", "prompt_text": "音频中说的话"}
# 文件放在 deploy/cosyvoice/asset/ 目录下
# 要求音频清晰（5~15秒为佳，不要过长），保存为wav格式，采集率16kHz,单声道，内容任意，但是和prompt_text里的内容完全一致，必须一字不差！
# CosyVoice3 的 prompt_text 必须以 "You are a helpful assistant.<|endofprompt|>" 开头
# ============================================================================
VOICE_CONFIGS = [
    {
        "id": "default",  # 默认音色
        "file": "zero_shot_prompt.wav",  # asset/zero_shot_prompt.wav
        "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
    },
    # 添加更多音色示例 (取消注释并修改):
    {
        "id": "longyingcheng",
        "file": "longyingcheng_man.wav",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>真不好意思，从小至今，他还从来没有被哪一位异性朋友亲吻过呢。"
    },
    {
        "id": "longyingwan",
        "file": "longyingwan_woman.wav",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>我们将为全球城市的可持续发展贡献力量。"
    },
    {
        "id": "longyingmu",
        "file": "longyingmu_woman.wav",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>您好，我是智能电话助手，很高兴为您服务。请问您需要咨询业务预约办理还是查询信息？"
    }
]
# ============================================================================

# 全局变量
cosyvoice = None
inference_lock = threading.Lock()

# 多音色缓存: {voice_id: {"file": path, "prompt_text": text}}
voice_cache = {}
# 音色加载失败原因: {voice_id: {"message": str, ...}}
voice_load_errors = {}
default_voice_id = "default"  # 默认使用的音色ID

# 输出采样率 (可通过 --output_sample_rate 配置)
# 默认 16000 以兼容小智平台
output_sample_rate = 16000
cosyvoice3_prompt_prefix = "You are a helpful assistant.<|endofprompt|>"
supported_response_formats = {"wav", "mp3", "flac", "pcm", "aac", "opus"}

# [注意] CosyVoice API 不支持传入 Tensor，必须传路径或文件对象，因此移除 resampler_cache 优化


app = FastAPI(title="CosyVoice TTS Server", version="1.0.0")


class OpenAISpeechRequest(BaseModel):
    input: str = Field(..., description="要合成的文本")
    model: Optional[str] = Field(default=None, description="OpenAI 兼容字段，当前服务不按 model 切换模型")
    voice: str = Field(default=default_voice_id, description="预置音色ID，task_type=CustomVoice 时使用")
    response_format: str = Field(default="wav", description="音频格式: wav, mp3, flac, pcm, aac, opus")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="播放速度，非流式输出时生效")
    task_type: Optional[str] = Field(default=None, description="TTS任务类型: CustomVoice, VoiceDesign, Base")
    language: str = Field(default="Auto", description="语言标识，当前仅记录日志")
    instructions: str = Field(default="", description="声音风格/情感指令，当前仅记录日志")
    max_new_tokens: int = Field(default=2048, ge=1, description="兼容字段，当前仅记录日志")
    initial_codec_chunk_frames: Optional[int] = Field(default=None, ge=1, description="兼容字段，当前仅记录日志")
    stream: bool = Field(default=False, description="是否流式返回原始PCM；要求 response_format=pcm")
    ref_audio: Optional[str] = Field(default=None, description="Base任务参考音频: URL、data URL、file:// URI 或本地路径")
    ref_text: Optional[str] = Field(default=None, description="Base任务参考音频转写文本，不需要包含 CosyVoice3 前缀")
    x_vector_only_mode: Optional[bool] = Field(default=None, description="兼容字段，CosyVoice3当前不支持仅说话人向量模式")

    @root_validator(pre=True)
    def accept_legacy_reference_names(cls, values):
        if "ref_audio" not in values and "reference_aduio" in values:
            values["ref_audio"] = values.pop("reference_aduio")
        if "ref_audio" not in values and "reference_audio" in values:
            values["ref_audio"] = values.pop("reference_audio")
        if "ref_text" not in values and "reference_text" in values:
            values["ref_text"] = values.pop("reference_text")
        if "reference_aduio" in values:
            values.pop("reference_aduio")
        elif "reference_audio" in values:
            values.pop("reference_audio")
        if "reference_text" in values:
            values.pop("reference_text")
        return values

    class Config:
        extra = "forbid"


# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def generate_audio_stream(
        text: str,
        voice_id: str = None,
        prompt_text: str = None,
        prompt_wav=None,
        stream: bool = True,
        use_spk_cache: bool = False
) -> Generator[bytes, None, None]:
    """
    生成流式音频数据

    Args:
        text: 要合成的文本
        voice_id: 音色ID (从预配置音色中选择参考音频)
        prompt_text: 自定义音色的提示文本 (与 prompt_wav 配合使用)
        prompt_wav: 自定义音色的参考音频 (与 prompt_text 配合使用)
        stream: 是否流式输出
        use_spk_cache: voice_id 音色是否使用预缓存 speaker 特征
    """
    global cosyvoice, voice_cache

    with inference_lock:
        try:
            # 确定使用哪个音色
            spk_id = ""
            actual_prompt_text = prompt_text
            actual_prompt_wav = prompt_wav

            # 优先使用 voice_id (预配置音色)
            if voice_id and voice_id in voice_cache:
                voice_info = voice_cache[voice_id]
                actual_prompt_text = voice_info["prompt_text"]
                actual_prompt_wav = voice_info["file"]
                if use_spk_cache:
                    spk_id = voice_id
                    logger.info(f"⚡ 使用预缓存音色: {voice_id} (零I/O/计算)")
                else:
                    spk_id = ""
                    logger.info(f"使用音色文件实时提取特征: {voice_id} -> {actual_prompt_wav}")
            elif voice_id and voice_id not in voice_cache:
                raise ValueError(f"音色 '{voice_id}' 未加载，可用音色: {list(voice_cache.keys())}")
            elif prompt_text and prompt_wav:
                # 使用自定义音色 (无缓存，需实时计算)
                logger.info("使用自定义音色 (实时计算特征)")
            else:
                # 使用默认音色
                if default_voice_id in voice_cache:
                    spk_id = default_voice_id
                    voice_info = voice_cache[default_voice_id]
                    actual_prompt_text = voice_info["prompt_text"]
                    actual_prompt_wav = voice_info["file"]
                    logger.info(f"⚡ 使用默认音色: {default_voice_id}")

            if isinstance(actual_prompt_wav, str) and not os.path.exists(actual_prompt_wav):
                raise FileNotFoundError(f"参考音频文件不存在: {actual_prompt_wav}")

            logger.info(
                f"调用cosyvoice.inference_zero_shot: {text}, actual_prompt_text: {actual_prompt_text}, "
                f"actual_prompt_wav:{actual_prompt_wav}, stream={stream}, zero_shot_spk_id={spk_id or None}"
            )
            infer_kwargs = {
                "stream": stream
            }
            if spk_id:
                infer_kwargs["zero_shot_spk_id"] = spk_id

            for result in cosyvoice.inference_zero_shot(
                    text,
                    actual_prompt_text,
                    actual_prompt_wav,
                    **infer_kwargs
            ):
                audio_tensor = result['tts_speech']

                # [GPU 重采样] 如果输出采样率与模型原生不同，进行重采样
                if output_sample_rate != cosyvoice.sample_rate:
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor,
                        orig_freq=cosyvoice.sample_rate,
                        new_freq=output_sample_rate
                    )

                # [GPU 版本] PCM 16bit 转换
                # 1. GPU 进行乘法 (* 32768)
                # 2. GPU 进行类型转换 (float -> int16)
                # 3. 传输 int16 (2 bytes) 到 CPU
                yield (audio_tensor * 32768).to(torch.int16).cpu().numpy().tobytes()
        except Exception as e:
            logger.error(f"TTS 生成失败: {e}")
            raise


def pcm16_bytes_to_wav_bytes(
        pcm_bytes: bytes,
        sample_rate: int,
        channels: int = 1,
        sample_width: int = 2
) -> bytes:
    """
    将 PCM16 bytes 封装为 WAV bytes，不落盘。
    """
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

    wav_buffer.seek(0)
    return wav_buffer.read()


def encode_audio_response(pcm_bytes: bytes, response_format: str, sample_rate: int) -> tuple[bytes, str, str]:
    """将 PCM16 mono 音频编码为 OpenAI audio/speech 兼容的响应格式。"""
    fmt = response_format.lower().strip()

    if fmt == "pcm":
        return pcm_bytes, "application/octet-stream", "pcm"

    if fmt == "wav":
        return pcm16_bytes_to_wav_bytes(pcm_bytes, sample_rate), "audio/wav", "wav"

    if fmt == "aac":
        return encode_aac_response(pcm_bytes, sample_rate)

    format_map = {
        "mp3": ("MP3", "MPEG_LAYER_III", "audio/mpeg", "mp3"),
        "flac": ("FLAC", "PCM_16", "audio/flac", "flac"),
        "opus": ("OGG", "OPUS", "audio/ogg", "opus"),
    }

    if fmt not in format_map:
        raise HTTPException(
            status_code=400,
            detail="response_format 仅支持: mp3, wav, pcm, flac, opus, aac"
        )

    import soundfile as sf

    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    sf_format, subtype, media_type, extension = format_map[fmt]
    out = io.BytesIO()
    sf.write(out, audio, sample_rate, format=sf_format, subtype=subtype)
    out.seek(0)
    return out.read(), media_type, extension


def encode_aac_response(pcm_bytes: bytes, sample_rate: int) -> tuple[bytes, str, str]:
    """使用 PyAV 将 PCM16 mono 编码为 AAC/ADTS。"""
    try:
        import av
    except ImportError as e:
        raise HTTPException(status_code=400, detail="response_format=aac 需要安装 PyAV") from e

    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    out = io.BytesIO()
    container = av.open(out, mode="w", format="adts")
    stream = container.add_stream("aac", rate=sample_rate)
    stream.layout = "mono"

    frame = av.AudioFrame.from_ndarray(audio.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = sample_rate
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    out.seek(0)
    return out.read(), "audio/aac", "aac"


def apply_speed_to_pcm(pcm_bytes: bytes, speed: float, sample_rate: int) -> bytes:
    if abs(speed - 1.0) < 1e-3:
        return pcm_bytes

    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    new_freq = max(1, int(sample_rate / speed))
    sped = torchaudio.functional.resample(audio_tensor, orig_freq=sample_rate, new_freq=new_freq)
    sped = torch.clamp(sped.squeeze(0), -1.0, 1.0)
    return (sped * 32768).to(torch.int16).cpu().numpy().tobytes()


def build_cosyvoice3_prompt_text(reference_text: str) -> str:
    text = reference_text.strip()
    if text.startswith(cosyvoice3_prompt_prefix):
        return text
    return f"{cosyvoice3_prompt_prefix}{text}"


def resolve_local_audio_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def prepare_ref_audio(ref_audio: str) -> tuple[str, Optional[str]]:
    """解析 vLLM-Omni ref_audio，返回本地路径和需要清理的临时文件路径。"""
    value = ref_audio.strip()
    parsed = urlparse(value)

    if parsed.scheme in ("http", "https"):
        suffix = os.path.splitext(parsed.path)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            req = Request(value, headers={"User-Agent": "cosyvoice-tts-server"})
            with urlopen(req, timeout=30) as resp:
                tmp.write(resp.read())
            return tmp.name, tmp.name

    if parsed.scheme == "file":
        return unquote(parsed.path), None

    if value.startswith("data:"):
        header, _, payload = value.partition(",")
        if ";base64" not in header or not payload:
            raise HTTPException(status_code=400, detail="ref_audio data URL 必须是 base64 格式")
        mime = header[5:].split(";")[0]
        suffix = {
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3",
            "audio/flac": ".flac",
            "audio/ogg": ".ogg",
        }.get(mime, ".wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(base64.b64decode(payload))
            return tmp.name, tmp.name

    return resolve_local_audio_path(value), None


def reference_audio_error_response(ref_audio: str, resolved_path: str) -> dict:
    return {
        "error": {
            "message": f"参考音频不存在，请检查 ref_audio 路径：{ref_audio}",
            "type": "invalid_request_error",
            "param": "ref_audio",
            "code": "reference_audio_not_found"
        },
        "resolved_path": resolved_path
    }


def error_response(status_code: int, message: str, param: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": code
            }
        }
    )


def save_upload_to_temp_file(upload_file: UploadFile) -> str:
    suffix = os.path.splitext(upload_file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        upload_file.file.seek(0)
        shutil.copyfileobj(upload_file.file, tmp)
        return tmp.name


def pcm_stream_response(audio_generator: Generator[bytes, None, None]) -> StreamingResponse:
    return StreamingResponse(
        audio_generator,
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(cosyvoice.sample_rate if cosyvoice else output_sample_rate),
            "X-Channels": "1",
            "X-Bits": "16",
            "X-Audio-Format": "pcm"
        }
    )


def get_official_inference_method(method_name: str):
    if cosyvoice is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    if not hasattr(cosyvoice, method_name):
        raise HTTPException(status_code=404, detail=f"当前模型不支持 {method_name}")
    return getattr(cosyvoice, method_name)


def official_pcm_generator(method_name: str, method, *args, cleanup_files: Optional[list[str]] = None) -> Generator[bytes, None, None]:
    """
    官方 /inference_* 兼容接口使用的裸 PCM16 流。
    保持官方示例行为：直接返回模型原生采样率的 PCM，不套 WAV 头。
    """
    if cleanup_files is None:
        cleanup_files = []

    with inference_lock:
        try:
            logger.info(f"官方兼容接口调用: {method_name}")
            for result in method(*args):
                audio_tensor = result["tts_speech"]
                yield (audio_tensor * (2 ** 15)).to(torch.int16).cpu().numpy().tobytes()
        except Exception as e:
            logger.exception(f"官方兼容接口 {method_name} 生成失败")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            for path in cleanup_files:
                if path and os.path.exists(path):
                    os.unlink(path)


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    method = get_official_inference_method("inference_sft")
    return pcm_stream_response(
        official_pcm_generator("inference_sft", method, tts_text, spk_id)
    )


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(
        tts_text: str = Form(),
        prompt_text: str = Form(),
        prompt_wav: UploadFile = File()
):
    method = get_official_inference_method("inference_zero_shot")
    prompt_wav_path = save_upload_to_temp_file(prompt_wav)
    return pcm_stream_response(
        official_pcm_generator(
            "inference_zero_shot",
            method,
            tts_text,
            prompt_text,
            prompt_wav_path,
            cleanup_files=[prompt_wav_path]
        )
    )


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    method = get_official_inference_method("inference_cross_lingual")
    prompt_wav_path = save_upload_to_temp_file(prompt_wav)
    return pcm_stream_response(
        official_pcm_generator(
            "inference_cross_lingual",
            method,
            tts_text,
            prompt_wav_path,
            cleanup_files=[prompt_wav_path]
        )
    )


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(
        tts_text: str = Form(),
        spk_id: str = Form(),
        instruct_text: str = Form()
):
    method = get_official_inference_method("inference_instruct")
    return pcm_stream_response(
        official_pcm_generator("inference_instruct", method, tts_text, spk_id, instruct_text)
    )


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(
        tts_text: str = Form(),
        instruct_text: str = Form(),
        prompt_wav: UploadFile = File()
):
    method = get_official_inference_method("inference_instruct2")
    prompt_wav_path = save_upload_to_temp_file(prompt_wav)
    return pcm_stream_response(
        official_pcm_generator(
            "inference_instruct2",
            method,
            tts_text,
            instruct_text,
            prompt_wav_path,
            cleanup_files=[prompt_wav_path]
        )
    )


@app.get("/health")
async def health_check():
    """健康检查"""
    # import torch # 已在全局导入

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024 ** 3:.2f}GB",
            "gpu_memory_cached": f"{torch.cuda.memory_reserved(0) / 1024 ** 3:.2f}GB"
        }

    return JSONResponse({
        "status": "ok",
        "model": "Fun-CosyVoice3-0.5B-2512",
        "model_sample_rate": cosyvoice.sample_rate if cosyvoice else None,
        "output_sample_rate": output_sample_rate,
        "available_voices": list(voice_cache.keys()),
        "unavailable_voices": voice_load_errors,
        "default_voice": default_voice_id,
        **gpu_info
    })


@app.get("/v1/audio/voices")
async def list_audio_voices():
    return JSONResponse({
        "voices": list(voice_cache.keys()),
        "uploaded_voices": []
    })


@app.post("/v1/audio/speech")
async def openai_audio_speech(req: OpenAISpeechRequest):
    """
    OpenAI 兼容 TTS 接口。

    入参 JSON:
      - input: 要合成的文本
      - model/voice/response_format/speed: OpenAI speech 标准字段
      - task_type/language/instructions/max_new_tokens/initial_codec_chunk_frames/stream: vLLM-Omni 扩展字段
      - ref_audio/ref_text/x_vector_only_mode: vLLM-Omni Base 声音克隆字段
    """
    if cosyvoice is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    text = req.input.strip() if req.input else ""
    if not text:
        raise HTTPException(status_code=400, detail="input 不能为空")

    task_type = req.task_type or ("Base" if req.ref_audio or req.ref_text else "CustomVoice")
    task_type_map = {
        "customvoice": "CustomVoice",
        "voicedesign": "VoiceDesign",
        "base": "Base",
    }
    normalized_task_type = task_type_map.get(task_type.lower())
    if normalized_task_type is None:
        return error_response(
            400,
            "task_type 仅支持 CustomVoice、VoiceDesign、Base",
            "task_type",
            "unsupported_task_type"
        )
    task_type = normalized_task_type

    response_format = req.response_format.lower().strip()
    if response_format not in supported_response_formats:
        return error_response(
            400,
            "response_format 仅支持: wav, mp3, flac, pcm, aac, opus",
            "response_format",
            "unsupported_response_format"
        )
    if req.stream and response_format != "pcm":
        return error_response(
            400,
            "stream=true 时 response_format 必须为 pcm",
            "response_format",
            "stream_requires_pcm"
        )
    if req.stream and abs(req.speed - 1.0) >= 1e-3:
        return error_response(
            400,
            "stream=true 时暂不支持 speed 调整，请使用 speed=1.0",
            "speed",
            "stream_speed_unsupported"
        )
    if req.x_vector_only_mode:
        return error_response(
            400,
            "当前 CosyVoice3 服务不支持 x_vector_only_mode=true，请同时传 ref_audio 和 ref_text",
            "x_vector_only_mode",
            "x_vector_only_mode_unsupported"
        )
    if task_type == "VoiceDesign":
        return error_response(
            400,
            "当前 CosyVoice3 服务不支持 VoiceDesign，请使用 task_type=Base 并传 ref_audio/ref_text",
            "task_type",
            "voice_design_unsupported"
        )
    if task_type != "Base" and (req.ref_audio or req.ref_text):
        return error_response(
            400,
            "ref_audio/ref_text 仅支持 task_type=Base；如需声音克隆请使用 task_type=Base",
            "task_type",
            "reference_fields_require_base"
        )

    temp_ref_audio_path = None
    prompt_text = None
    prompt_wav = None
    voice_id = None

    if task_type == "Base":
        ref_audio = req.ref_audio.strip() if req.ref_audio else ""
        if not ref_audio:
            raise HTTPException(status_code=400, detail="task_type=Base 时 ref_audio 不能为空")

        ref_text = req.ref_text.strip() if req.ref_text else ""
        if not ref_text:
            raise HTTPException(status_code=400, detail="task_type=Base 时 ref_text 不能为空")

        try:
            prompt_wav, temp_ref_audio_path = prepare_ref_audio(ref_audio)
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"参考音频读取失败: {e}")
            return error_response(400, f"参考音频读取失败：{e}", "ref_audio", "reference_audio_load_failed")

        if not os.path.exists(prompt_wav):
            return JSONResponse(
                status_code=400,
                content=reference_audio_error_response(ref_audio, prompt_wav)
            )
        prompt_text = build_cosyvoice3_prompt_text(ref_text)
    else:
        voice_id = req.voice.strip() if req.voice else default_voice_id
        if voice_id not in voice_cache:
            return error_response(
                400,
                f"音色 '{voice_id}' 未加载，可用音色: {list(voice_cache.keys())}",
                "voice",
                "voice_not_found"
            )
        prompt_wav = voice_cache[voice_id]["file"]
        prompt_text = voice_cache[voice_id]["prompt_text"]
        if not os.path.exists(prompt_wav):
            return JSONResponse(
                status_code=400,
                content=reference_audio_error_response(prompt_wav, prompt_wav)
            )

    logger.info(
        f"TTS 请求: text='{text[:50]}...', task_type='{task_type}', voice='{voice_id}', "
        f"ref_audio='{prompt_wav}', response_format='{response_format}', stream={req.stream}, "
        f"speed={req.speed}, language='{req.language}', max_new_tokens={req.max_new_tokens}, "
        f"initial_codec_chunk_frames={req.initial_codec_chunk_frames}"
    )

    start_time = time.time()

    try:
        if req.stream:
            def pcm_stream_generator():
                first_chunk = True
                total_bytes = 0
                try:
                    for chunk in generate_audio_stream(
                            text=text,
                            voice_id=voice_id,
                            prompt_text=prompt_text,
                            prompt_wav=prompt_wav,
                            stream=True,
                            use_spk_cache=False
                    ):
                        if first_chunk:
                            logger.info(f"⚡ 首帧延迟: {(time.time() - start_time) * 1000:.0f}ms")
                            first_chunk = False
                        total_bytes += len(chunk)
                        yield chunk
                    logger.info(
                        f"✅ TTS 完成: 总耗时 {(time.time() - start_time) * 1000:.0f}ms, "
                        f"PCM={total_bytes / 1024:.1f}KB, format=pcm, stream=true"
                    )
                finally:
                    if temp_ref_audio_path and os.path.exists(temp_ref_audio_path):
                        os.unlink(temp_ref_audio_path)

            return StreamingResponse(
                pcm_stream_generator(),
                media_type="application/octet-stream",
                headers={
                    "X-Sample-Rate": str(output_sample_rate),
                    "X-Channels": "1",
                    "X-Bits": "16",
                    "X-Task-Type": task_type,
                    "X-Audio-Format": "pcm",
                    "X-Stream": "true"
                }
            )

        pcm_chunks = []
        first_chunk = True
        for chunk in generate_audio_stream(
                text=text,
                voice_id=voice_id,
                prompt_text=prompt_text,
                prompt_wav=prompt_wav,
                stream=False,
                use_spk_cache=False
        ):
            if first_chunk:
                logger.info(f"⚡ 首帧延迟: {(time.time() - start_time) * 1000:.0f}ms")
                first_chunk = False
            pcm_chunks.append(chunk)

        if not pcm_chunks:
            raise RuntimeError("没有生成任何音频数据")

        pcm_bytes = b"".join(pcm_chunks)
        pcm_bytes = apply_speed_to_pcm(pcm_bytes, req.speed, output_sample_rate)
        audio_bytes, media_type, extension = encode_audio_response(
            pcm_bytes=pcm_bytes,
            response_format=response_format,
            sample_rate=output_sample_rate
        )

        logger.info(
            f"✅ TTS 完成: 总耗时 {(time.time() - start_time) * 1000:.0f}ms, "
            f"PCM={len(pcm_bytes) / 1024:.1f}KB, 输出={len(audio_bytes) / 1024:.1f}KB, "
            f"format={extension}"
        )

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{extension}"',
                "X-Sample-Rate": str(output_sample_rate),
                "X-Channels": "1",
                "X-Bits": "16",
                "X-Task-Type": task_type,
                "X-Audio-Format": extension,
                "X-Stream": "false"
            }
        )
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.warning(f"TTS 参考音频不存在: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "param": "ref_audio",
                    "code": "reference_audio_not_found"
                },
                "resolved_path": prompt_wav
            }
        )
    except Exception as e:
        logger.exception("TTS 生成失败")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if not req.stream and temp_ref_audio_path and os.path.exists(temp_ref_audio_path):
            os.unlink(temp_ref_audio_path)


def register_cosyvoice_vllm_model():
    """按官方示例注册 CosyVoice vLLM 模型类。"""
    from vllm import ModelRegistry
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

    try:
        ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
        logger.info("已注册 vLLM 模型: CosyVoice2ForCausalLM")
    except ValueError as e:
        if "already" not in str(e).lower():
            raise
        logger.info("vLLM 模型 CosyVoice2ForCausalLM 已注册")


def load_model(
        model_dir: str,
        device: str = "cuda",
        fp16: bool = False,
        use_vllm: bool = False,
        load_trt: bool = False
):
    """加载 CosyVoice 模型"""
    global cosyvoice, voice_cache, voice_load_errors

    from cosyvoice.cli.cosyvoice import AutoModel

    logger.info(f"正在加载模型: {model_dir}")
    logger.info(f"设备: {device}, FP16: {fp16}, TensorRT: {load_trt}, vLLM加速: {use_vllm}")

    if use_vllm:
        try:
            register_cosyvoice_vllm_model()
        except ImportError:
            logger.error("启用 vLLM 失败: 未找到 vllm 库。请先安装: pip install vllm==0.9.0")
            sys.exit(1)

    start_time = time.time()
    try:
        # CosyVoice3 官方示例: AutoModel(..., load_trt=True, load_vllm=True, fp16=False)
        cosyvoice = AutoModel(
            model_dir=model_dir,
            fp16=fp16,
            load_trt=load_trt,
            load_vllm=use_vllm
        )
    except TypeError as e:
        if "load_vllm" in str(e):
            logger.error("当前 CosyVoice 版本似乎不支持 vLLM，请确保使用最新代码")
        raise e

    logger.info(f"模型加载完成，耗时: {time.time() - start_time:.1f}s")
    logger.info(f"模型采样率: {cosyvoice.sample_rate}Hz, 输出采样率: {output_sample_rate}Hz")

    # ========== 加载多音色配置 ==========
    voice_cache.clear()
    voice_load_errors.clear()
    asset_dir = os.path.join(SCRIPT_DIR, "asset")
    official_asset_dir = os.path.join(SCRIPT_DIR, "official", "asset")

    logger.info(f"⚡ 正在加载 {len(VOICE_CONFIGS)} 个音色配置...")

    for voice_config in VOICE_CONFIGS:
        voice_id = voice_config["id"]
        voice_file = voice_config["file"]
        prompt_text = voice_config["prompt_text"]

        # 查找音频文件
        voice_path = os.path.join(asset_dir, voice_file)
        searched_paths = [voice_path]
        if not os.path.exists(voice_path):
            # 尝试从官方目录查找
            voice_path = os.path.join(official_asset_dir, voice_file)
            searched_paths.append(voice_path)

        if not os.path.exists(voice_path):
            logger.warning(f"❌ 音色 '{voice_id}' 的文件未找到: {voice_file}")
            voice_load_errors[voice_id] = {
                "message": "参考音频文件不存在，请确认音频已放到 asset/ 或 official/asset/ 目录",
                "file": voice_file,
                "searched_paths": searched_paths,
                "code": "reference_audio_not_found"
            }
            continue

        try:
            # 缓存音色特征
            cosyvoice.add_zero_shot_spk(prompt_text, voice_path, voice_id)

            # 保存到 voice_cache
            voice_cache[voice_id] = {
                "file": voice_path,
                "prompt_text": prompt_text
            }
            voice_load_errors.pop(voice_id, None)
            logger.info(f"✅ 音色 '{voice_id}' 加载成功: {voice_file} 文件路径: {voice_path}")
        except Exception as e:
            logger.warning(f"❌ 音色 '{voice_id}' 加载失败: {e}")
            voice_load_errors[voice_id] = {
                "message": "参考音频加载失败，请检查音频格式、采样率、声道和 prompt_text 是否匹配",
                "file": voice_path,
                "reason": str(e),
                "code": "reference_audio_load_failed"
            }

    logger.info(f"⚡ 音色加载完成，共 {len(voice_cache)} 个可用音色: {list(voice_cache.keys())}")

    # 预热推理 (使用第一个可用音色)
    first_voice_path = None
    if voice_cache:
        first_voice_id = list(voice_cache.keys())[0]
        first_voice_path = voice_cache[first_voice_id]["file"]
    warmup_model(first_voice_path, first_voice_id if voice_cache else None)

    return cosyvoice


def warmup_model(prompt_wav_path: str = None, voice_id: str = None):
    """预热模型，减少首次请求延迟"""
    global cosyvoice

    if cosyvoice is None:
        return

    logger.info("🔥 正在预热模型...")
    start_time = time.time()

    warmup_text = "预热测试"
    warmup_prompt_text = "预热"

    # 如果有参考音频，使用 zero-shot 预热
    if prompt_wav_path and os.path.exists(prompt_wav_path):
        try:
            # 使用指定的 voice_id 进行预热
            spk_id = voice_id if voice_id else "default"
            for _ in cosyvoice.inference_zero_shot(
                    warmup_text,
                    warmup_prompt_text,
                    prompt_wav_path,
                    stream=False,
                    zero_shot_spk_id=spk_id
            ):
                pass
            logger.info(f"✅ 模型预热完成，耗时: {time.time() - start_time:.1f}s")
        except Exception as e:
            logger.warning(f"预热失败 (不影响正常使用): {e}")
    else:
        logger.info("⏭ 跳过预热 (无参考音频)")


def main():
    global output_sample_rate

    parser = argparse.ArgumentParser(description="CosyVoice TTS Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=10096, help="监听端口")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/Fun-CosyVoice3-0.5B",
        help="模型目录路径"
    )
    parser.add_argument("--device", type=str, default="cuda", help="运行设备: cuda 或 cpu")
    parser.add_argument("--fp16", action="store_true", help="使用 FP16 推理 (节省显存)")
    parser.add_argument("--load_trt", action="store_true", help="[优化] 使用 TensorRT 加速推理")
    parser.add_argument("--use_vllm", action="store_true", help="[优化] 使用 vLLM 加速推理 (需 pip install vllm)")
    parser.add_argument(
        "--output_sample_rate",
        type=int,
        default=16000,
        choices=[16000, 24000],
        help="输出采样率: 16000 (兼容小智平台) 或 24000 (原生高质量)"
    )
    args = parser.parse_args()

    # 设置输出采样率
    output_sample_rate = args.output_sample_rate

    # 处理相对路径
    if not os.path.isabs(args.model_dir):
        args.model_dir = os.path.join(SCRIPT_DIR, args.model_dir)

    # 加载模型
    load_model(args.model_dir, args.device, args.fp16, args.use_vllm, args.load_trt)

    # 启动服务
    logger.info(f"服务已启动: http://{args.host}:{args.port}")
    logger.info(f"健康检查: http://{args.host}:{args.port}/health")
    logger.info(f"OpenAI 兼容 TTS 接口: POST http://{args.host}:{args.port}/v1/audio/speech")
    logger.info(f"📢 输出采样率: {output_sample_rate}Hz (模型原生: 24000Hz)")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
