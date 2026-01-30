# Qwen3-TTS FastAPI 部署文档

本文档介绍如何使用 FastAPI 部署 Qwen3-TTS 语音合成服务，支持**多GPU部署**和**高并发**。

## 目录

- [环境准备](#环境准备)
- [下载模型权重](#下载模型权重)
- [启动服务](#启动服务)
- [启动脚本参数说明](#启动脚本参数说明)
- [多GPU部署](#多gpu部署)
- [API接口文档](#api接口文档)
- [调用示例](#调用示例)
- [常见问题](#常见问题)

---

## 环境准备

### 1. 创建 conda 环境

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
```

### 2. 安装项目

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
pip install -e .
```

### 3. (可选) 安装 FlashAttention 2 加速

```bash
pip install -U flash-attn --no-build-isolation
```

---

## 下载模型权重

### 方式一：ModelScope（国内推荐）

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local_dir ./Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

### 方式二：Hugging Face

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir ./Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

---

## 启动服务

### 1. 修改配置

编辑 `start_server.sh`，设置模型路径和GPU：

```bash
MODEL_PATH="/your/path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
GPU_ID="0"        # 使用的GPU编号，多卡用逗号分隔如 "0,1"
WORKERS=4         # worker进程数
PORT="8009"       # 端口号
```

### 2. 启动

```bash
chmod +x start_server.sh
./start_server.sh
```

### 3. 验证服务

```bash
curl http://localhost:8009/health
```

返回：
```json
{"status": "healthy", "model_loaded": true, "cuda_available": true}
```

---

## 启动脚本参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 绑定地址 | `0.0.0.0` |
| `--port` | 端口号 | `8009` |
| `--model` | 模型路径 | 脚本内配置 |
| `--gpu` | GPU编号，多卡用逗号分隔 | `2` |
| `--workers` | worker进程数 | `1` |
| `--reload` | 开发模式，代码修改自动重启 | - |

### 示例

```bash
# 单卡启动
./start_server.sh --gpu 0 --workers 4

# 双卡启动，8个worker
./start_server.sh --gpu 2,3 --workers 8

# 指定模型路径
./start_server.sh --model /path/to/model --gpu 0,1 --workers 4
```

---

## 多GPU部署

### 工作原理

指定多个GPU时（如 `--gpu 2,3`）：
1. 设置 `CUDA_VISIBLE_DEVICES` 使程序只能看到指定的GPU
2. 每个 worker 根据进程ID自动分配到不同GPU
3. worker 均匀分布在所有可用GPU上

### 示例：2卡8进程

```bash
./start_server.sh --gpu 2,3 --workers 8
```

效果：
- 使用物理GPU 2 和 GPU 3
- 启动8个worker进程
- 每张卡分配4个worker

### 启动日志

```
============================================
Qwen3-TTS FastAPI Server
============================================
Host:       0.0.0.0
Port:       8009
Model:      /path/to/model
Workers:    8
GPU:        2,3
============================================

Worker PID 12345 using device: cuda:0
Worker PID 12346 using device: cuda:1
Worker PID 12347 using device: cuda:0
Worker PID 12348 using device: cuda:1
...
```

### 推荐配置

| GPU数量 | Worker数 | 适用场景 |
|---------|----------|----------|
| 1 | 2-4 | 开发测试 |
| 2 | 4-8 | 生产环境 |
| 4 | 8-16 | 高并发场景 |

---

## API接口文档

### 健康检查

```
GET /health
```

**返回：**
```json
{"status": "healthy", "model_loaded": true, "cuda_available": true}
```

---

### 获取支持的语言

```
GET /languages
```

**返回：**
```json
{"languages": ["auto", "chinese", "english", "japanese", "korean", ...]}
```

---

### 语音合成（单条）

```
POST /tts/voice_design
```

**请求参数：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | ✅ | - | 要合成的文本 |
| `instruct` | string | ✅ | - | 声音风格描述 |
| `language` | string | ❌ | `"Auto"` | 语言：Chinese, English, Japanese 等 |
| `max_new_tokens` | int | ❌ | `2048` | 最大生成token数 |
| `temperature` | float | ❌ | `0.9` | 采样温度 |
| `top_p` | float | ❌ | `1.0` | Top-p采样 |
| `top_k` | int | ❌ | `50` | Top-k采样 |
| `response_format` | string | ❌ | `"wav"` | `"wav"` 返回音频文件，`"base64"` 返回base64编码 |

**请求示例：**

```json
{
  "text": "你好，欢迎使用Qwen3语音合成服务。",
  "instruct": "温柔甜美的年轻女声",
  "language": "Chinese"
}
```

**返回：**
- `response_format="wav"`：直接返回音频文件
- `response_format="base64"`：返回JSON
  ```json
  {"audio_base64": "UklGRi...", "sample_rate": 24000, "format": "wav"}
  ```

---

### 批量语音合成

```
POST /tts/voice_design/batch
```

**请求参数：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `texts` | string[] | ✅ | 文本列表 |
| `instructs` | string[] | ✅ | 声音风格列表（与texts长度一致） |
| `languages` | string[] | ❌ | 语言列表（默认全部Auto） |

**请求示例：**

```json
{
  "texts": ["你好", "Hello"],
  "instructs": ["温柔女声", "Friendly male voice"],
  "languages": ["Chinese", "English"]
}
```

**返回：**

```json
{"audios": ["UklGRi...", "UklGRi..."], "sample_rate": 24000, "format": "wav"}
```

---

## 调用示例

### Python

```python
import requests

# 获取音频文件
response = requests.post(
    "http://localhost:8009/tts/voice_design",
    json={
        "text": "你好，欢迎使用Qwen3语音合成服务。",
        "instruct": "温柔甜美的年轻女声",
        "language": "Chinese"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### cURL

```bash
curl -X POST "http://localhost:8009/tts/voice_design" \
  -H "Content-Type: application/json" \
  -d '{"text": "你好", "instruct": "温柔女声", "language": "Chinese"}' \
  --output output.wav
```

---

## 声音风格示例

`instruct` 参数支持自然语言描述，示例：

| 描述 | 效果 |
|------|------|
| `温柔甜美的年轻女声` | 温柔甜美的女声 |
| `沉稳大气的男声` | 沉稳的男声 |
| `活泼开朗的少女声音` | 活泼的少女声 |
| `知性优雅的女声` | 知性女声 |
| `充满激情的演讲者声音` | 激情演讲风格 |

---

## 常见问题

### 模型加载失败

```
HTTPException: 503 - Model not loaded
```

**解决：** 检查模型路径是否正确：
```bash
ls -la /path/to/model/
# 应包含：config.json, model.safetensors 等文件
```

### 显存不足

```
RuntimeError: CUDA out of memory
```

**解决：**
1. 减少每张卡的worker数量
2. 使用更多GPU：`--gpu 0,1,2,3`
3. 减少请求中的 `max_new_tokens`

### Worker没有分布到多卡

**检查：** 查看启动日志中的设备分配：
```
Worker PID 12345 using device: cuda:0
Worker PID 12346 using device: cuda:1
```

如果都是 `cuda:0`，检查 `CUDA_VISIBLE_DEVICES` 是否正确设置。
