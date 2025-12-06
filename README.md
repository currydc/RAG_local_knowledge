1、安装环境：
（1）python版本：Python 3.12.10
（2）安装LibreOffice
（3）公网：pip install -r requirements.txt
（4）安装ollama，并安装模型（本次使用的是qwen3:1.7b模型）
（5）嵌入模型：北京智源人工智能研究院;BAAI/bge-large-zh-v1.5嵌入模型
2、启动flask：py .\flask_api.py
3、启动ollama服务：ollama serve
4、使用：
（1）build_promt.py中嵌入模型可使用api，替换本地模型（北京智源人工智能研究院;BAAI/bge-large-zh-v1.5嵌入模型），通过切换api_url、api_key
（2）flask_api.py中通过配置OPENAI_API_KEY、OPENAI_BASE_URL切换本地ollama模型与api
（3）qwen3系列支持在用户提示或系统消息中添加 /think 和 /no_think 来逐回合切换模型的思考模式，其它系列模型可能不支持切换思考模式
