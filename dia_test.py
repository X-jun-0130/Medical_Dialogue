import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
os.chdir('/Nlp_2023/Dialogue_Bloom/')

from transformers import AutoTokenizer,AutoModelForCausalLM
from pydantic import BaseModel, validator
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

model_name = "./Bloom_Dia/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
model.eval()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_request: str = '外阴这两侧容易发红不疼不痒。怎么回事啊？\n希望获得的帮助: \n皮肤科医生看看这两侧容易发红怎么回事？\n怀孕情况: \n未怀孕\n患病多久: \n一周内\n过敏史: \n无（2018-06-12填写）\n既往病史: \n无（2018-06-12填写）'
    history: list = []


class ChatResponse(BaseModel):
    message: str = ''
    history: list = []
    
# '''
# inference
# '''
response_id = 'Doctor:'

def infer(model, payload, history):
    payload = 'Patient:'+ payload

    if len(history) > 0:
        input_text = '<s>' + ''.join([k[0]+k[1] for k in history]) + payload + response_id
    else:
        input_text = '<s>'  + payload + response_id

    
    his_length = len(''.join([k[0]+k[1] for k in history]))

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()
    logits = model.generate(input_ids, num_beams=1, top_k=3, repetition_penalty=1.1, max_length= his_length+150)
    out = tokenizer.decode(logits[0].tolist())
    out = out.replace(input_text, '')
    out = out.split('Patient:')[0]
    out = out.replace('Doctor:', '。')
    out = out.replace('</s>', '')

    history.append([payload, response_id + out])
    return out, history

@app.post(f"/ask",summary="发送一个请求给接口，返回一个答案",response_model=ChatResponse)
def ask(request_data: ChatRequest):
    inputs = request_data.user_request
    history = request_data.history
    response, history = infer(model, inputs, history)
    # print(response)
    # print(history)
    if len(tokenizer.encode(''.join([k[0]+k[1] for k in history]))) >= 500:
        return {"message": '内容超出限制，请重新开始话题。', "history": history}
    else:
        if response:
            return {"message": response, "history": history}
        else:
            return {"message": '很抱歉，我暂时无法回答这个问题。', "history": history}

if __name__ == '__main__':
    uvicorn.run('dia_test:app', host="0.0.0.0", port=5053)

