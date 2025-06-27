"""
# WebAPI文档

` python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml `

## 执行参数:
    `-a` - `绑定地址, 默认"127.0.0.1"`
    `-p` - `绑定端口, 默认9880`
    `-c` - `TTS配置文件路径, 默认"GPT_SoVITS/configs/tts_infer.yaml"`

## 调用:

### 推理

endpoint: `/tts`
GET:
```
http://127.0.0.1:9880/tts?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```

POST:
```json
{
    "text": "",                   # str.(required) text to be synthesized
    "text_lang: "",               # str.(required) language of the text to be synthesized
    "ref_audio_path": "",         # str.(required) reference audio path
    "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
    "prompt_text": "",            # str.(optional) prompt text for the reference audio
    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
    "top_k": 5,                   # int. top k sampling
    "top_p": 1,                   # float. top p sampling
    "temperature": 1,             # float. temperature for sampling
    "text_split_method": "cut0",  # str. text split method, see text_segmentation_method.py for details.
    "batch_size": 1,              # int. batch size for inference
    "batch_threshold": 0.75,      # float. threshold for batch splitting.
    "split_bucket": True,         # bool. whether to split the batch into multiple buckets.
    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
    "streaming_mode": False,      # bool. whether to return a streaming response.
    "seed": -1,                   # int. random seed for reproducibility.
    "parallel_infer": True,       # bool. whether to use parallel inference.
    "repetition_penalty": 1.35,   # float. repetition penalty for T2S model.
    "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
    "super_sampling": False       # bool. whether to use super-sampling for audio when using VITS model V3.
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400

### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
```
http://127.0.0.1:9880/control?command=restart
```
POST:
```json
{
    "command": "restart"
}
```

RESP: 无


### 切换GPT模型

endpoint: `/set_gpt_weights`

GET:
```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
```
RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400


### 切换Sovits模型

endpoint: `/set_sovits_weights`

GET:
```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/s2G488k.pth
```

RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400

"""
import io
import json
import os
import sys
import time
import traceback
from threading import Lock
from traceback import print_exception
from typing import Generator

from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest

import nls

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response, UploadFile, File
from fastapi.responses import PlainTextResponse, StreamingResponse, JSONResponse
import uvicorn
import yaml
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from pydantic import BaseModel


# print(sys.path)
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="tts_infer.yaml", help="path of tts_infer.yaml")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
parser.add_argument("-v", "--version", type=str, default="v2", help="version")
parser.add_argument("--device", type=str, default="cpu", help="device")
parser.add_argument("--is_half", type=str, default="false", help="is_half")
parser.add_argument("--enable_aliyun_tts", action="store_true", help="enable aliyun tts")
args = parser.parse_args()
config_path = args.tts_config
# device = args.device
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "tts_infer.yaml"

with open(config_path, "r", encoding="utf-8", errors="strict") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

configs["version"] = args.version
configs[args.version]["device"] = args.device
configs[args.version]["is_half"] = args.is_half

configs["custom"] = configs[args.version]

tts_config = TTS_Config(configs)
print(tts_config)
tts_pipeline = TTS(tts_config)

APP = FastAPI()


class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False


### modify from https://github.com/RVC-Boss/GPT-SoVITS/pull/894/files
def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",  # 输入16位有符号小端整数PCM
            "-ar",
            str(rate),  # 设置采样率
            "-ac",
            "1",  # 单声道
            "-i",
            "pipe:0",  # 从管道读取输入
            "-c:a",
            "aac",  # 音频编码器为AAC
            "-b:a",
            "192k",  # 比特率
            "-vn",  # 不包含视频
            "-f",
            "adts",  # 输出AAC数据流格式
            "pipe:1",  # 将输出写入管道
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def check_params(req: dict):
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    streaming_mode: bool = req.get("streaming_mode", False)
    media_type: str = req.get("media_type", "wav")
    prompt_lang: str = req.get("prompt_lang", "")
    text_split_method: str = req.get("text_split_method", "cut5")

    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if text_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    elif text_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"text_lang: {text_lang} is not supported in version {tts_config.version}"},
        )
    if prompt_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}"},
        )
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    elif media_type == "ogg" and not streaming_mode:
        return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})

    if text_split_method not in cut_method_names:
        return JSONResponse(
            status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"}
        )

    return None


async def tts_handle(req: dict):
    """
    Text to speech handler.

    Args:
        req (dict):
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker synthesis
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 5,                   # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "media_type": "wav",          # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
                "streaming_mode": False,      # bool. whether to return a streaming response.
                "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
                "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.
                "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
                "super_sampling": False,       # bool. whether to use super-sampling for audio when using VITS model V3.
            }
    returns:
        StreamingResponse: audio stream response.
    """

    streaming_mode = req.get("streaming_mode", False)
    return_fragment = req.get("return_fragment", False)
    media_type = req.get("media_type", "wav")

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    if streaming_mode or return_fragment:
        req["return_fragment"] = True

    try:
        tts_generator = tts_pipeline.run(req)

        if streaming_mode:

            def streaming_generator(tts_generator: Generator, media_type: str):
                if_frist_chunk = True
                for sr, chunk in tts_generator:
                    if if_frist_chunk and media_type == "wav":
                        yield wave_header_chunk(sample_rate=sr)
                        media_type = "raw"
                        if_frist_chunk = False
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()

            # _media_type = f"audio/{media_type}" if not (streaming_mode and media_type in ["wav", "raw"]) else f"audio/x-{media_type}"
            return StreamingResponse(
                streaming_generator(
                    tts_generator,
                    media_type,
                ),
                media_type=f"audio/{media_type}",
            )

        else:
            sr, audio_data = next(tts_generator)
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        print_exception(e)
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(e)})


@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)


@APP.get("/tts")
async def tts_get_endpoint(
    text: str = None,
    text_lang: str = None,
    ref_audio_path: str = None,
    aux_ref_audio_paths: list = None,
    prompt_lang: str = None,
    prompt_text: str = "",
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut0",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    streaming_mode: bool = False,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: int = 32,
    super_sampling: bool = False,
    **kwargs,
):
    data_path = get_work_data_dir("uploaded_audio")
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": os.path.join(data_path, ref_audio_path) if ref_audio_path else "",
        "aux_ref_audio_paths": list(map(lambda p: os.path.join(data_path, p), aux_ref_audio_paths)) if aux_ref_audio_paths else [],
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower(),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty),
        "sample_steps": int(sample_steps),
        "super_sampling": super_sampling,
        **kwargs,
    }
    return await tts_handle(req)


@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.model_dump()
    return await tts_get_endpoint(**req)


@APP.post("/upload_refer_audio")
async def upload_refer_audio(audio_file: UploadFile = File(...)):
    try:
        upload_path = get_work_data_dir("uploaded_audio")
        os.makedirs(upload_path, exist_ok=True)
        save_path = os.path.join(upload_path, audio_file.filename)

        with open(save_path , "wb") as buffer:
            buffer.write(await audio_file.read())

        return PlainTextResponse(status_code=200, content=audio_file.filename)
    except Exception as e:
        return PlainTextResponse(status_code=400, content=str(e))


# @APP.get("/set_refer_audio")
# async def set_refer_aduio(refer_audio_path: str = None):
#     try:
#         tts_pipeline.set_ref_audio(refer_audio_path)
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"message": "set refer audio failed", "Exception": str(e)})
#     return JSONResponse(status_code=200, content={"message": "success"})
#

# @APP.post("/set_refer_audio")
# async def set_refer_aduio_post(audio_file: UploadFile = File(...)):
#     try:
#         # 检查文件类型，确保是音频文件
#         if not audio_file.content_type.startswith("audio/"):
#             return JSONResponse(status_code=400, content={"message": "file type is not supported"})

#         os.makedirs("uploaded_audio", exist_ok=True)
#         save_path = os.path.join("uploaded_audio", audio_file.filename)
#         # 保存音频文件到服务器上的一个目录
#         with open(save_path , "wb") as buffer:
#             buffer.write(await audio_file.read())

#         tts_pipeline.set_ref_audio(save_path)
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})
#     return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        tts_pipeline.init_t2s_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change gpt weight failed", "Exception": str(e)})

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        tts_pipeline.init_vits_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


# create aliyun client
aliyun_client = None
if args.enable_aliyun_tts:
    ak_id = os.getenv('ALIYUN_AK_ID')
    ak_secret = os.getenv('ALIYUN_AK_SECRET')
    region_id = os.getenv('ALIYUN_REGION_ID')
    if ak_id and ak_secret and region_id:
        aliyun_client = AcsClient(ak_id, ak_secret, region_id)
cached_aliyun_token = {}
cache_lock = Lock()


def refresh_aliyun_token():
    if not aliyun_client:
        return None

    token_request = CommonRequest()
    token_request.set_method('POST')
    token_request.set_domain(f'nls-meta.cn-shanghai.aliyuncs.com')
    token_request.set_version('2019-02-28')
    token_request.set_action_name('CreateToken')

    try:
        token_response = aliyun_client.do_action_with_exception(token_request)
        print(token_response)

        jss = json.loads(token_response)
        if 'Token' in jss and 'Id' in jss['Token']:
            token_info = jss['Token']
            print(f'token: {token_info["Id"]}, expireTime: {token_info["ExpireTime"]}')
            return token_info
    except Exception as e:
        print('Failed to refresh aliyun sts token!', e)
        return None


def get_aliyun_token():
    if not aliyun_client:
        return None

    global cached_aliyun_token
    with cache_lock:
        if 'Id' in cached_aliyun_token and 'ExpireTime' in cached_aliyun_token:
            expire_time = cached_aliyun_token['ExpireTime']
            if time.time() < expire_time:
                return cached_aliyun_token['Id']

        new_token_info = refresh_aliyun_token()
        if new_token_info:
            cached_aliyun_token['Id'] = new_token_info['Id']
            cached_aliyun_token['ExpireTime'] = new_token_info['ExpireTime']
            return cached_aliyun_token['Id']
        else:
            print("Unable to retrieve a new token.")
            return None


def get_work_data_dir(module: str) -> str:
    data_home = os.environ.get('DATA_HOME', '/workspace/data')
    data_path = os.path.join(data_home, module)
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    return data_path


def inference_by_aliyun(text: str,
                        voice: str,
                        emotion: str = None,
                        output_format: str = None,
                        request_id: str = '') -> bytes:
    api_url = os.getenv('ALIYUN_TTS_URL')
    if not api_url:
        print('Miss aliyun tts api url.', sys.stderr, flush=True)
        raise IllegalStateException()

    app_key = os.getenv('ALIYUN_TTS_APP_KEY')
    if not app_key:
        print('Miss aliyun tts app key.', sys.stderr, flush=True)
        raise IllegalStateException()

    sts_token = get_aliyun_token()
    if not sts_token:
        print('Miss aliyun sts token.', sys.stderr, flush=True)
        raise IllegalStateException()

    if emotion and voice.endswith('_emo'):
        text = f'<speak voice="{voice}"><emotion category="{emotion}">{text}</emotion></speak>'

    out = io.BytesIO()

    def on_metainfo(message, *custom_args):
        print("aliyun tts on_metainfo message=>{} args=>{}".format(message, custom_args))

    def on_error(message, *custom_args):
        print("aliyun tts on_error message=>{} args=>{}".format(message, custom_args))

    def on_close(*custom_args):
        print("aliyun tts on_close: args=>{}".format(custom_args))

    def on_data(data, *custom_args):
        try:
            out.write(data)
        except Exception as e:
            print("write data failed: ", e)

    def on_completed(message, *custom_args):
        print("aliyun tts on_completed:args=>{} message=>{}".format(custom_args, message))

    tts = nls.NlsSpeechSynthesizer(url=api_url,
                                   token=sts_token,
                                   appkey=app_key,
                                   on_metainfo=on_metainfo,
                                   on_data=on_data,
                                   on_completed=on_completed,
                                   on_error=on_error,
                                   on_close=on_close,
                                   callback_args=[request_id])
    tts.start(text=text,
              aformat=output_format,
              voice=voice,
              wait_complete=True)
    tts.shutdown()
    data = out.getvalue()
    out.close()
    return data


@APP.get('/ping')
def ping():
    return PlainTextResponse(status_code=200, content='pong')


def test_aliyun_token():
    token = get_aliyun_token()
    cached_token = get_aliyun_token()
    print(f'aliyun token: {get_aliyun_token()}, cached token: {cached_aliyun_token}')
    assert token == cached_token


def test_aliyun_tts():
    output_path = './test_aliyun_tts.mp3'

    try:
        os.remove(output_path)
    except Exception:
        pass

    f = open(output_path, 'wb')

    def test_on_metainfo(message, *custom_args):
        print("on_metainfo message=>{} args=>{}".format(message, custom_args))

    def test_on_error(message, *custom_args):
        print("on_error mesage=>{} args=>{}".format(message, custom_args))

    def test_on_close(*custom_args):
        print("on_close: args=>{}".format(custom_args))
        try:
            f.close()
        except Exception as e:
            print("close file failed since:", e)

    def test_on_data(data, *custom_args):
        try:
            f.write(data)
        except Exception as e:
            print("write data failed: ", e)

    def test_on_completed(message, *custom_args):
        print("on_completed:args=>{} message=>{}".format(custom_args, message))

    voice = 'zhimi_emo'
    text = '作为普通人，最好不要认为自己会是那少数的幸运儿。'
    emotion = 'sad'
    emo_text = f'<speak voice="{voice}"><emotion category="{emotion}">{text}</emotion></speak>'
    api_url = os.getenv('ALIYUN_TTS_URL')
    app_key = os.getenv('ALIYUN_TTS_APP_KEY')
    nls.enableTrace(True)
    tts = nls.NlsSpeechSynthesizer(url=api_url,
                                   token=get_aliyun_token(),
                                   appkey=app_key,
                                   on_metainfo=test_on_metainfo,
                                   on_data=test_on_data,
                                   on_completed=test_on_completed,
                                   on_error=test_on_error,
                                   on_close=test_on_close,
                                   callback_args=['test args'])
    tts.start(text=emo_text,
              aformat='mp3',
              voice=voice,
              wait_complete=True)
    tts.shutdown()
    assert os.path.exists(output_path)


class IllegalArgumentException(Exception):
    pass


class IllegalStateException(Exception):
    pass


if __name__ == "__main__":
    try:
        if host == "None":  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
