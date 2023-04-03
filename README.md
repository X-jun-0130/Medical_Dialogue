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
``
添加角色信息：Patient、Doctor
将数据依次拼接："Patient:咳嗽半个月，吃药好了，突然又发高烧，吃了药后老是反复发烧Doctor: 可以先做血常规CRP，支原体，过敏源检查Patient: crp是什么检查Doctor: C－反应蛋白排除有没有炎症。Patient: 前几天查了个c反正蛋白质室13.2Doctor: CRP高，应该正规使用消炎药，7天后复查"
添加首尾符号进入模型
``
