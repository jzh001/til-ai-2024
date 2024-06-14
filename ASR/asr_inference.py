from transformers import pipeline

'''
docker tag anti-heroes-asr asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-anti-heroes/anti-heroes-asr:latest
docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-anti-heroes/anti-heroes-asr:latest

gcloud ai models upload --region asia-southeast1 --display-name 'anti-heroes-asr' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-anti-heroes/anti-heroes-asr:latest --container-health-route /health --container-predict-route /stt --container-ports 5001 --version-aliases default


TAG='tinyp'
docker tag anti-heroes-asr-$TAG asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-anti-heroes/anti-heroes-$TAG-asr:latest
docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-anti-heroes/anti-heroes-$TAG-asr:latest

gcloud ai models upload --region asia-southeast1 --display-name anti-heroes-$TAG-asr --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-anti-heroes/anti-heroes-$TAG-asr:latest --container-health-route /health --container-predict-route /stt --container-ports 5001 --version-aliases default
'''

#MODEL_PATH = "models/whisper-tiny-v1"
MODEL_PATH = "auto/model_dir"
transcriber = pipeline("automatic-speech-recognition", model=MODEL_PATH, device=0)

def predict(path_or_file):
    return transcriber(path_or_file)['text']

def parallel_predict(paths_or_files):
    result = transcriber(paths_or_files)
    result_texts = [x['text'] for x in result]
#    result_texts = [text.lower().replace('stand by', 'standby') for text in result_texts]
    return result_texts
