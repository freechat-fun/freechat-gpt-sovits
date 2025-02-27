#!flask/bin/python

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from threading import Lock
from typing import List

import ffmpeg
import numpy as np
import torch
import torchaudio
from flask import Flask, render_template, request, Response

from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest

import nls
from TTS.config import load_config
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.audio.numpy_transforms import save_wav
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager


def create_argparser():
    def convert_boolean(x):
        return x.lower() in ['true', '1', 'yes']

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--list_models',
        type=convert_boolean,
        nargs='?',
        const=True,
        default=False,
        help='list available pre-trained tts and vocoder models.',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='tts_models/en/ljspeech/tacotron2-DDC',
        help='Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>',
    )
    parser.add_argument('--vocoder_name', type=str, default=None, help='name of one of the released vocoder models.')

    # Args for running custom models
    parser.add_argument('--config_path', default=None, type=str, help='Path to model config file.')
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to model file.',
    )
    parser.add_argument(
        '--vocoder_path',
        type=str,
        help='Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).',
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument('--speakers_file_path', type=str, help='JSON file for multi-speaker model.', default=None)
    parser.add_argument('--vocab_path', type=str, help='Path to vocab file.', default=None)
    parser.add_argument('--port', type=int, default=5002, help='port to listen on.')
    parser.add_argument('--use_cuda', type=convert_boolean, default=False, help='true to use CUDA.')
    parser.add_argument('--debug', type=convert_boolean, default=False, help='true to enable Flask debug mode.')
    parser.add_argument('--show_details', type=convert_boolean, default=False, help='Generate model detail page.')
    parser.add_argument('--enable_aliyun_tts', type=convert_boolean, default=False, help='true to enable aliyun tts.')
    return parser


# parse the args
args = create_argparser().parse_args()

path = Path(__file__).parent / '../.models.json'
manager = ModelManager(path)

# update in-use models to the specified released models.
model_path = None
config_path = None
speakers_file_path = None
vocab_path = None
vocoder_path = None
vocoder_config_path = None

# CASE1: list pre-trained TTS models
if args.list_models:
    manager.list_models()
    sys.exit()

# CASE2: load pre-trained model paths
if args.model_name is not None and not args.model_path:
    model_path, config_path, model_item = manager.download_model(args.model_name)
    args.vocoder_name = model_item['default_vocoder'] if args.vocoder_name is None else args.vocoder_name

if args.vocoder_name is not None and not args.vocoder_path:
    vocoder_path, vocoder_config_path, _ = manager.download_model(args.vocoder_name)

# CASE3: set custom model paths
if args.model_path is not None:
    model_path = args.model_path
    config_path = args.config_path
    speakers_file_path = args.speakers_file_path
    vocab_path = args.vocab_path

# create the xtts model
config = XttsConfig()
config.load_json(config_path)
model = Xtts.init_from_config(config)
if args.use_cuda:
    model.load_checkpoint(config, checkpoint_dir=model_path, vocab_path=vocab_path, use_deepspeed=True)
    model.cuda()
else:
    model.load_checkpoint(config, checkpoint_dir=model_path, vocab_path=vocab_path)

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


def to_wav_file(wav: List[int] | torch.Tensor | np.ndarray) -> bytes:
    out = io.BytesIO()
    if torch.is_tensor(wav):
        wav = wav.cpu().numpy()
    if isinstance(wav, list):
        wav = np.array(wav)
    save_wav(wav=wav, path=out, sample_rate=config.model_args.output_sample_rate, pipe_out=None)
    data = out.getvalue()
    out.close()
    return data


def to_pcm_bytes(wav: List[int] | torch.Tensor | np.ndarray) -> io.BytesIO:
    if torch.is_tensor(wav):
        wav = wav.cpu().numpy()
    if isinstance(wav, list):
        wav = np.array(wav)
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wav_norm = wav_norm.astype(np.int16)
    return bytes(wav_norm.ravel().view('b').data)


def get_work_data_dir(module: str) -> str:
    data_home = os.environ.get("DATA_HOME")
    data_path = os.path.join(data_home, module) if data_home else get_user_data_dir(module)
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    return data_path


def convert_pcm(data: List[int] | torch.Tensor | np.ndarray, output_format: str = 'mp3',
                sample_rate: int = config.model_args.output_sample_rate, bitrate: str = '64k') -> bytes:
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    if isinstance(data, list):
        data = np.array(data)
    data_norm = data * (32767 / max(0.01, np.max(np.abs(data))))
    data_norm = data_norm.astype(np.int16)

    if data_norm.ndim == 1:
        channels = 1
    else:
        channels = data_norm.shape[1]
    bit_depth = data_norm.dtype.itemsize * 8
    pcm_format = f's{bit_depth}le'

    try:
        process = (
            ffmpeg
            .input('pipe:0', format=pcm_format, ar=sample_rate, ac=channels)
            .output('pipe:1', format=output_format, audio_bitrate=bitrate)
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True)
        )

        input_data = bytes(data_norm.ravel().view('b').data)
        stdout, stderr = process.communicate(input=input_data)
        if stderr:
            print("Error:", stderr.decode())
        else:
            print(f"Successfully converted pcm to {output_format}")

        return stdout
    except Exception as e:
        print(f"An error occurred: {e}")
        raise IllegalStateException(e)


def convert_audio_to_wav(input_file_path: str, output_file_path: str = 'output.wav') -> None:
    try:
        process = (
            ffmpeg
            .input(input_file_path)
            .output(output_file_path, format='wav')
            .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
        )

        stdout, stderr = process.communicate()

        if stderr:
            print("Error:", stderr.decode())
        else:
            print(f"Successfully converted {input_file_path} to {output_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise IllegalStateException(e)


class IllegalArgumentException(Exception):
    pass


class IllegalStateException(Exception):
    pass


# APIs
app = Flask(__name__)
lock = Lock()


def inference(output_format: str = None):
    request_id = request.headers.get('Request-Id', '')
    print(f' > [{request.method}] {request.path}')
    print(f' > Request id: {request_id}')

    text = request.headers.get('text') or request.values.get('text', '')
    speaker_idx = request.headers.get('speaker-id') or request.values.get('speaker_id', '')
    language_idx = request.headers.get('language-id') or request.values.get('language_id', '')
    speaker_wav = request.headers.get('speaker-wav') or request.values.get('speaker_wav', '')

    if speaker_idx:
        print(f' > Speaker Idx: {speaker_idx}')
        third_party_speaker = re.compile(r"\[(.+?)](.+?)(\|([^|]*))?")
        match = third_party_speaker.match(speaker_idx)

        if match:
            platform = match.group(1)
            speaker = match.group(2)
            emotion = match.group(4)
            if platform == 'aliyun' and aliyun_client:
                return inference_by_aliyun(text, speaker, emotion, output_format, request_id)
        gpt_cond_latent, speaker_embedding = model.speaker_manager.speakers[speaker_idx].values()
    elif speaker_wav:
        upload_folder = get_work_data_dir('wav')
        audio_path = os.path.join(upload_folder, speaker_wav)
        print(f' > Speaker Wav: {speaker_wav}')
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[audio_path])
    else:
        print('Miss speaker information.', sys.stderr, flush=True)
        raise IllegalArgumentException()

    print(f' > Model input: {text}')
    print(f' > Language Idx: {language_idx}')

    if gpt_cond_latent is None or speaker_embedding is None:
        print('The gpt_cond_latent and speaker_embedding should not be None.', sys.stderr, flush=True)
        raise IllegalStateException()

    result = model.inference(text, language_idx, gpt_cond_latent, speaker_embedding)
    if result is None or 'wav' not in result:
        print('Failed to inference.', sys.stderr, flush=True)
        raise IllegalStateException()

    if not output_format:
        return result['wav']
    elif 'wav' == output_format:
        return to_wav_file(result['wav'])
    else:
        return convert_pcm(result['wav'], output_format)


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


@app.route('/inference/wav', methods=['POST'])
def tts_wav():
    with lock:
        t0 = time.time()

        try:
            out = inference('wav')
        except IllegalArgumentException:
            return None, 400
        except IllegalStateException:
            return None, 500
        print(f'Inference of audio length {out.getbuffer().nbytes}, time: {time.time() - t0}', flush=True)
        return Response(out, mimetype='audio/wav', direct_passthrough=True)


@app.route('/inference/aac', methods=['POST'])
def tts_aac():
    with lock:
        t0 = time.time()

        try:
            out = inference('adts')
        except IllegalArgumentException:
            return None, 400
        except IllegalStateException:
            return None, 500
        print(f'Inference of audio length {len(out)}, time: {time.time() - t0}', flush=True)
        return Response(out, mimetype='audio/aac', direct_passthrough=True)


@app.route('/inference/mp3', methods=['POST'])
def tts_mp3():
    with lock:
        t0 = time.time()

        try:
            out = inference('mp3')
        except IllegalArgumentException:
            return None, 400
        except IllegalStateException:
            return None, 500
        print(f'Inference of audio length {len(out)}, time: {time.time() - t0}', flush=True)
        return Response(out, mimetype='audio/mpeg', direct_passthrough=True)


@app.route('/inference/data/stream', methods=['POST'])
def tts_stream():
    with lock:
        print(f' > [{request.method}] {request.path}')
        print(f' > Request id: {request.headers.get("Request-Id", "")}')

        text = request.headers.get('text') or request.values.get('text', '')
        speaker_idx = request.headers.get('speaker-id') or request.values.get('speaker_id', '')
        language_idx = request.headers.get('language-id') or request.values.get('language_id', '')
        speaker_wav = request.headers.get('speaker-wav') or request.values.get('speaker_wav', '')

        if speaker_idx:
            print(f' > Speaker Idx: {speaker_idx}')
            gpt_cond_latent, speaker_embedding = model.speaker_manager.speakers[speaker_idx].values()
        elif speaker_wav:
            upload_folder = get_work_data_dir('wav')
            audio_path = os.path.join(upload_folder, speaker_wav)
            print(f' > Speaker Wav: {speaker_wav}')
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[audio_path])
        else:
            return None, 400

        print(f' > Model input: {text}')
        print(f' > Language Idx: {language_idx}')

        def generate_chunks():
            print('Inference...')
            t0 = time.time()

            chunks = model.inference_stream(text, language_idx, gpt_cond_latent, speaker_embedding)
            for i, chunk in enumerate(chunks):
                t1 = time.time()
                print(f'Received chunk {i} of audio length {chunk.shape[-1]}, time: {t1 - t0}', flush=True)
                t0 = t1
                yield to_pcm_bytes(chunk)

        return Response(generate_chunks(), mimetype='audio/wav', direct_passthrough=True)


@app.route('/speaker/wav', methods=['POST'])
def upload_wav():
    print(f' > [{request.method}] {request.path}')
    print(f' > Request id: {request.headers.get("Request-Id", "")}', flush=True)

    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    input_filename = os.path.basename(file.filename)
    basename, ext = os.path.splitext(input_filename)
    if ext != '.wav':
        return 'Input file should be .wav', 400

    upload_folder = get_work_data_dir('wav')
    file_path = os.path.join(upload_folder, input_filename)
    file.save(file_path)

    return input_filename, 200


@app.route('/speaker/mp3', methods=['POST'])
def upload_mp3():
    print(f' > [{request.method}] {request.path}')
    print(f' > Request id: {request.headers.get("Request-Id", "")}', flush=True)

    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    input_filename = os.path.basename(file.filename)
    basename, ext = os.path.splitext(input_filename)
    if ext != '.mp3':
        return 'Input file should be .mp3', 400

    output_filename = f'{basename}.wav'
    tmp_dir = os.getenv('TMPDIR', '/tmp')
    upload_folder = get_work_data_dir('wav')
    input_file_path = os.path.join(tmp_dir, input_filename)
    output_file_path = os.path.join(upload_folder, output_filename)

    file.save(input_file_path)
    convert_audio_to_wav(input_file_path, output_file_path)

    return output_filename, 200


@app.route('/speaker/wav/<filename>', methods=['DELETE'])
def delete_file(filename):
    print(f' > [{request.method}] {request.path}')
    print(f' > Request id: {request.headers.get("Request-Id", "")}', flush=True)

    upload_folder = get_work_data_dir('wav')
    file_path = os.path.join(upload_folder, filename)

    try:
        os.remove(file_path)
        return f'File {filename} deleted successfully!', 200
    except FileNotFoundError:
        return f'File {filename} not found', 404


@app.route('/speaker/wav/exists/<filename>', methods=['GET'])
def exists_file(filename):
    print(f' > [{request.method}] {request.path}')
    print(f' > Request id: {request.headers.get("Request-Id", "")}', flush=True)

    upload_folder = get_work_data_dir('wav')
    file_path = os.path.join(upload_folder, filename)

    return (f'File {filename} exists!', 200) \
        if os.path.exists(file_path) \
        else (f'File {filename} not found!', 404)


@app.route('/details')
def details():
    print(f' > [{request.method}] {request.path}')
    print(f' > Request id: {request.headers.get("Request-Id", "")}', flush=True)

    if args.config_path is not None and os.path.isfile(args.config_path):
        model_config = load_config(args.config_path)
    else:
        if args.model_name is not None:
            model_config = load_config(config_path)
        else:
            model_config = None

    if args.vocoder_config_path is not None and os.path.isfile(args.vocoder_config_path):
        vocoder_config = load_config(args.vocoder_config_path)
    else:
        if args.vocoder_name is not None:
            vocoder_config = load_config(vocoder_config_path)
        else:
            vocoder_config = None

    return render_template(
        'details.html',
        show_details=args.show_details,
        model_config=model_config,
        vocoder_config=vocoder_config,
        args=args.__dict__,
    )


@app.route('/ping', methods=['GET'])
def ping():
    return 'pong', 200


def test_stream():
    gpt_cond_latent, speaker_embedding = model.speaker_manager.speakers['Claribel Dervla'].values()

    print("Inference...")
    t0 = time.time()
    chunks = model.inference_stream(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding
    )

    wav_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            print(f"Time to first chunk: {time.time() - t0}")
        print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
        wav_chunks.append(chunk)
    wav = torch.cat(wav_chunks, dim=0)
    torchaudio.save("xtts_streaming.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)


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


def main():
    app.run(debug=args.debug, host='::', port=args.port)


if __name__ == '__main__':
    main()
