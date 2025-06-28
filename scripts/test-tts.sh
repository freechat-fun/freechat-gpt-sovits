#!/usr/bin/env bash

set -e

source $(dirname ${BASH_SOURCE[0]})/setenv.sh

serv_addr=${ARGS[0]:-http://127.0.0.1:9880}
zh_text="蝼蚁！赶紧叫我大王，否则打得你们呱呱叫！"
en_text="Ants! Call me the king now, or I will beat you to a pulp!"
ref_audio=${PROJECT_PATH}/t_voice.mp3
ref_lang=en
ref_text="You think you were born? Hum, no, you were built."

ref_path=`curl -s "${serv_addr}/upload_refer_audio" \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36' \
  -H 'Accept: text/plain, */*' \
  -F audio_file=@${ref_audio}
`

# echo "ref_path=${ref_path}"

mkdir -p ${PROJECT_PATH}/local-data/output

request_body=`cat << EOF
{
    "text": "${zh_text}",
    "text_lang": "zh",
    "ref_audio_path": "${ref_path}",
    "prompt_lang": "${ref_lang}",
    "prompt_text": "${ref_text}",
    "text_split_method": "cut0",
    "media_type": "aac"
}
EOF
`

output_audio=${PROJECT_PATH}/local-data/output/zh_sample.aac

curl -X POST "${serv_addr}/tts" \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36' \
  -H 'Content-Type: application/json' \
  -H 'Accept: audio/*' \
  -d "${request_body}" \
  -sSo "${output_audio}"

request_body=`cat << EOF
{
    "text": "${en_text}",
    "text_lang": "en",
    "ref_audio_path": "${ref_path}",
    "prompt_lang": "${ref_lang}",
    "prompt_text": "${ref_text}",
    "text_split_method": "cut0",
    "media_type": "aac"
}
EOF
`

output_audio=${PROJECT_PATH}/local-data/output/en_sample.aac

curl -X POST "${serv_addr}/tts" \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36' \
  -H 'Content-Type: application/json' \
  -H 'Accept: audio/*' \
  -d "${request_body}" \
  -sSo "${output_audio}"
