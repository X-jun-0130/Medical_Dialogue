# Medical_Dialogue
Chatdoctor、BloomModel、DialogueModel

### 基于Bloom_3B 使用医疗对话数据构建问诊对话模型
 模型：Bloom_3B
 
 数据：haodf
 

#### 对话预训练
Bloom_Model.py  利用deepspeed进行LLM加速训练

model_convert32_save.py 预训练模型保存


### 构建测试api
fastapi构建测试页面


#### python库需求
fastapi、其他同Bloom_FineTune项目相同


#### 对话模型预训练数据处理

添加角色信息：Patient、Doctor

将数据依次拼接："Patient:咳嗽半个月，吃药好了，突然又发高烧，吃了药后老是反复发烧Doctor: 可以先做血常规CRP，支原体，过敏源检查Patient: crp是什么检查Doctor: C－反应蛋白排除有没有炎症。Patient: 前几天查了个c反正蛋白质室13.2Doctor: CRP高，应该正规使用消炎药，7天后复查"

添加首尾符号进入模型



模型效果

{
  "message": " 好的",
  "history": [
    "Patient:肚子不舒服应该怎么办",
    "Doctor: 肚子不舒服，建议查一下胃镜，肠镜，肝胆胰脾彩超。",
    "Patient:不想做胃镜啊，有其他办法吗",
    "Doctor: 那就查一下肠镜吧。",
    "Patient:肠镜也不想做啊，能不能吃药就可以啊",
    "Doctor: 可以吃药，但是得查一下。",
    "Patient:查什么啊，您怀疑是哪些问题",
    "Doctor: 肠功能紊乱，胃炎，结肠炎，直肠炎，结肠息肉，直肠息肉，结肠癌",
    "Patient:肠功能紊乱严重吗",
    "Doctor: 不严重，但是容易反复发作。",
    "Patient:医生，我好像骂你怎么办",
    "Doctor: 你把话说完整了，我没法回复你了。",
    "Patient:我说错话了，肠功能紊乱应该吃什么药 ",
    "Doctor: 吃点得舒特吧",
    "Patient:这药苦不苦啊 ",
    "Doctor: 不是很苦",
    "Patient:我好倒霉啊，这能不能治好啊",
    "Doctor: 能治好",
    "Patient:行吧，谢谢医生",
    "Doctor: 好的"
  ]
}

