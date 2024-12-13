#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/setenv.sh

speakers=${PROJECT_PATH}/xtts_v2_speaker_idxs.json
text=你好，我是狐狸猜的创建者，一个聪明人！
lang=zh-cn

if [[ -f ${speakers} ]]; then
  IFS=$'\n'
  mkdir -p ${PROJECT_PATH}/local-data/output
  for name in $(jq -r '.[]' "${speakers}"); do
    if [[ -z ${name} ]]; then
      continue
    fi

    output_wav=${PROJECT_PATH}/local-data/output/${name}.wav
    curl -X POST 'https://tts.freechat.fun/api/tts' \
      -H 'Origin: https://tts.freechat.fun' \
      -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36' \
      -H 'Content-Type: application/x-www-form-urlencoded' \
      -H 'Accept: audio/wav, text/plain, */*' \
      -H "speaker-id: ${name}" \
      -H "language-id: ${lang}" \
      -d "text=${text}" \
      -sSo "${output_wav}"

    echo "${name} -> ${output_wav}"
  done
fi

