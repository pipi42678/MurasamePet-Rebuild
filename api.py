# -*- coding: utf-8 -*-
"""
MurasamePet API 服务
提供聊天、问答和视觉理解接口
"""

from fastapi import FastAPI, Request
from datetime import datetime
import uvicorn
import json
import platform
import sys
import httpx
import os
from openai import OpenAI
from Murasame.utils import get_config

# 确保标准输出使用 UTF-8 编码，防止中文乱码
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 检测平台
IS_MACOS = platform.system() == "Darwin"

# 延迟初始化的全局变量
ENGINE = None
DEVICE = None


def init_engine():
    """延迟初始化引擎和设备（按需导入大型库）"""
    global ENGINE, DEVICE
    
    if ENGINE is not None:
        return ENGINE, DEVICE
    
    if IS_MACOS:
        # 在 macOS 上强制要求 MLX
        print("🍎 检测到 macOS 系统，初始化 MLX 引擎...")
        try:
            from mlx_lm.utils import load
            from mlx_lm.generate import generate
            ENGINE = "mlx"
            DEVICE = "mlx"  # MLX 会自动使用 Apple Silicon GPU (Metal)
            print("✅ MLX 引擎加载成功 (Apple Silicon GPU 加速)")
        except ImportError as e:
            print(f"❌ 严重错误：macOS 系统需要 MLX 但未找到该库！")
            print(f"导入错误详情: {e}")
            print()
            print("🔍 解决方案：")
            print("1. 安装 MLX: pip install mlx-lm")
            print("2. 或确保您使用的 Python 环境支持 MLX")
            print()
            print("🚨 程序退出：macOS 系统必须使用 MLX 以获得最佳性能")
            exit(1)
    else:
        # 在非 macOS 系统上使用 PyTorch
        print("🖥️ 检测到非 macOS 系统，初始化 PyTorch 引擎...")
        import torch
        ENGINE = "torch"
        # 检测设备优先级：MPS > CUDA > CPU
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEVICE = "mps"
            print("✅ PyTorch 引擎加载成功 (使用 MPS 加速)")
        elif torch.cuda.is_available():
            DEVICE = "cuda"
            print("✅ PyTorch 引擎加载成功 (使用 CUDA 加速)")
        else:
            DEVICE = "cpu"
            print("⚠️ PyTorch 引擎加载成功 (使用 CPU，性能可能较慢)")
    
    return ENGINE, DEVICE

api = FastAPI()

config = get_config()
model_path = config.get("server", {}).get("llm", {}).get("local", {}).get("path", {}).get("MACOS" if IS_MACOS else "Win", None)
if model_path is None:
    print(f"❌ 严重错误：未配置模型路径 ({'MACOS' if IS_MACOS else 'Win'}): {model_path}")
    print("🔧 请在 config.json 中配置正确的模型路径。")
    exit(1)

max_seq_length = 2048


def load_model_and_tokenizer():
    # 先初始化引擎（按需导入大型库）
    config = get_config()
    engine, device = init_engine()
    
    print(f"📂 模型加载路径: {model_path}")
    print(f"⚙️ 推理引擎: {engine} | 计算设备: {device}")

    if IS_MACOS:
        # 在 macOS 上使用已合并的 MLX 模型
        from mlx_lm.utils import load
        from mlx_lm.generate import generate
        
        print("🍎 正在加载合并后的 MLX 模型 (Qwen3-14B-Murasame-Chat-MLX-Int4)...")

        # 检查合并后的模型是否存在
        # 检查 MLX 模型必需文件（配置文件和模型权重）
        required_static_files = ["tokenizer.json", "config.json"]
        missing_files = [f for f in required_static_files if not os.path.exists(os.path.join(model_path, f))]

        # 检查模型权重文件（单个 model.safetensors 或分片 model-*.safetensors）
        has_model_weights = False
        try:
            if os.path.exists(os.path.join(model_path, "model.safetensors")):
                has_model_weights = True
            else:
                if any(f.startswith("model-") and f.endswith(".safetensors") for f in os.listdir(model_path)):
                    has_model_weights = True
        except FileNotFoundError:
             # 如果 model_path 不存在，os.listdir 会报错
             pass

        if missing_files or not has_model_weights:
            if not has_model_weights:
                missing_files.append("model.safetensors (或 model-*.safetensors)")
            
            print(f"❌ 严重错误：在 {model_path} 中缺少以下 MLX 模型文件: {', '.join(missing_files)}")
            print("💡 请确保已为 macOS 下载了正确的合并模型，而不是 Windows 使用的 LoRA 文件。")
            print("   - 运行 'python download.py' 脚本来获取正确的模型。")
            exit(1)

        try:
            print("🔄 正在从磁盘读取模型文件...")
            # 直接加载合并后的完整模型（不需要单独的 base_model 和 adapter）
            model, tokenizer = load(model_path)
            print("✅ 合并 MLX 模型加载成功！")
            print(f"   📍 模型路径: {model_path}")
            print(f"   🏷️ 模型类型: Qwen3-14B + Murasame LoRA (已合并, Int4 量化)")
            print(f"   🚀 已启用 Apple Silicon GPU 加速")

        except Exception as e:
            print(f"❌ 严重错误：无法加载合并 MLX 模型！")
            print(f"错误详情: {e}")
            print()
            print("🔍 可能的原因：")
            print("1. 模型文件损坏或不完整")
            print("2. 下载的模型版本与 MLX 不兼容")
            print("3. 缺少必需的 MLX 依赖")
            print()
            print("💡 解决方案：")
            print("1. 重新运行 download.py 确保合并模型正确下载")
            print("2. 检查 MLX 和 mlx-lm 是否正确安装 (pip install mlx-lm)")
            print("3. 验证 ./models/Murasame 目录中的模型文件")
            print()
            print("🚨 程序退出：应用需要合并 MLX 模型才能运行")
            exit(1)
    else:
        # 在非 macOS 系统上使用 PyTorch (按需导入)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        print("🔧 正在加载完整模型...")

        try:
            print("🔄 正在准备完整模型...")

            if not os.path.exists(model_path):
                print(f"❌ 严重错误：模型路径不存在: {model_path}")
                print("💡 请确认完整模型是否已下载")
                exit(1)

            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            # device_map = "auto" if device == "cuda" else "cpu"
            bnb_quant_type = config.get('server', {}).get('llm', {}).get('local', {}).get('bnb_quant_type', None)
            bnb_config = None

            # 配置量化
            if device == "cuda" and bnb_quant_type:
                if bnb_quant_type == "int8":
                    print("📦 配置 8-bit 量化...")
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                elif bnb_quant_type in ["nf4", "fp4"]:
                    print(f"📦 配置 4-bit ({bnb_quant_type}) 量化...")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type=bnb_quant_type,
                        bnb_4bit_compute_dtype=torch_dtype
                    )
            else:
                # 如果是 CPU，或者 quant_type 为 None，强制设为 None
                if device == "cpu" and bnb_quant_type:
                    print("⚠️  警告: CPU 模式不支持 bitsandbytes 量化，将切换为默认 dtype 加载。")
                bnb_config = None

            if device == "cpu":
                print("⚠️  警告: 在 CPU 上加载 14B 模型需要大量内存 (通常 > 32GB)，请确保可用内存充足。")

            print(f"📦 正在加载模型: {model_path}")
            if bnb_config:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map=device,
                    trust_remote_code=True,
                    dtype=torch_dtype
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    trust_remote_code=True,
                    dtype=torch_dtype
                )

            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            print("✅ 完整模型加载成功！")
            print(f"   📍 模型: {model_path}")
            # print(f"   📍 适配器: {adapter_path}")
            print(f"   🏷️ 推理设备: {device}")
        except Exception as e:
            print(f"❌ 严重错误：无法加载完整模型！")
            print(f"错误详情: {e}")
            print()
            print("🔍 可能的原因：")
            print("1. 模型文件损坏或不完整")
            print("2. 缺少必需的 PyTorch 依赖")
            print()
            print("💡 解决方案：")
            print("重新运行 download.py 确保模型文件正确下载")
            print()
            print("🚨 程序退出：应用需要完整模型才能运行")
            exit(1)

    return model, tokenizer


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_request(prompt):
    print(f'📥 [{get_current_time()}] 收到请求: {prompt}')


def log_response(response):
    print(f'📤 [{get_current_time()}] 回复生成: {response}')


def parse_request(json_post_list):
    prompt = json_post_list.get('prompt', '')
    history = json_post_list.get('history', [])
    return prompt, history


def create_response(response_text, history, status=200):
    return {
        "response": response_text,
        "history": history,
        "status": status,
        "time": get_current_time()
    }


def call_openai_chat(base_url, api_key, model, messages, max_tokens=2048, temperature=0.7, connect_timeout=20.0, read_timeout=60.0):
    """
    通用接口调用，支持 DeepSeek, OpenRouter, SiliconFlow, Ollama 等
    """
    # 如果是本地兼容接口（如 Ollama），api_key 可能是空，OpenAI SDK 需要一个非空占位符
    final_api_key = api_key if api_key and api_key.strip() else "dummy-key"

    # 确保 base_url 格式正确
    if not base_url:
        raise ValueError("API endpoint (base_url) cannot be empty")

    client = OpenAI(
        base_url=base_url,
        api_key=final_api_key,
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=httpx.Timeout(None, connect=connect_timeout, read=read_timeout)
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"❌ OpenAI SDK 调用出错: {str(e)}")
        raise e


@api.post("/chat")
async def create_chat(request: Request):
    """
    主对话接口 (Murasame)
    支持：
    1. 云端/API模式 (通过 server.llm.cloud 配置)
    2. 本地权重模式 (通过加载 models/Murasame)
    """
    json_post_list = await request.json()
    prompt, history = parse_request(json_post_list)
    log_request(prompt)

    # 将当前 prompt 临时加入历史用于推理
    messages = history + [{'role': 'user', 'content': prompt}]

    config = get_config()
    llm_config = config.get('server', {}).get('llm', {})
    enable_local = llm_config.get('enable_local', False)
    connect_timeout = llm_config.get('timeout', {}).get('connect', 20.0)
    read_timeout = llm_config.get('timeout', {}).get('read', 60.0)

    reply = ""

    # --- 1. 云端/API 模式 ---
    if not enable_local:
        cloud_conf = llm_config.get('cloud', {})
        api_key = cloud_conf.get('api_key', '')
        endpoint = cloud_conf.get('endpoint', '')
        model_name = cloud_conf.get('model', '')

        print(f"🌐 [主模型] 调用 API: {model_name} @ {endpoint}")
        try:
            reply = call_openai_chat(
                base_url=endpoint,
                api_key=api_key,
                model=model_name,
                messages=messages,
                max_tokens=2048,
                temperature=0.7,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
        except Exception as e:
            return create_response(f"API Error: {str(e)}", history, status=500)

    # --- 2. 本地权重模式 (Murasame) ---
    else:
        try:
            print(f"💬 使用 {ENGINE} 引擎进行推理...")
            print(f"📊 最大生成长度: {json_post_list.get('max_new_tokens', 2048)} tokens")

            text = tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            print("✅ 聊天模板应用完成")

            max_new_tokens = int(json_post_list.get('max_new_tokens', 2048))
            max_new_tokens = max(1, max_new_tokens)
            temperature = float(json_post_list.get('temperature', 0.7))
            top_p = float(json_post_list.get('top_p', 0.9))
            top_p = max(0.01, min(top_p, 1.0))

            # 推理
            print("🤖 正在生成回复...")
            if ENGINE == "mlx":
                from mlx_lm.generate import generate
                response = generate(
                    model, tokenizer,
                    prompt=text,
                    max_tokens=max_new_tokens,
                    verbose=False
                )
                reply = response.strip()
            else:
                import torch
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                )
                encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "temperature": max(0.01, temperature),
                    "top_p": top_p,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        **generation_kwargs,
                    )
                generated_tokens = generated[0, encoded["input_ids"].shape[-1]:]
                reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
        except Exception as e:
            return create_response(f"API Error: {str(e)}", history, status=500)


    # 更新历史记录
    history.append({'role': 'user', 'content': prompt})
    history.append({"role": "assistant", "content": reply})
    print(f"✅ 回复生成完成 (长度: {len(reply)} 字符)")
    log_response(reply)
    return create_response(reply, history)


@api.post("/llm")
async def create_llm_chat(request: Request):
    """
    辅助/问答接口
    统一使用 call_openai_chat，根据配置决定连接云端还是本地 API (如 Ollama)
    """
    json_post_list = await request.json()
    prompt, history = parse_request(json_post_list)
    role = json_post_list.get('role', 'user')

    if prompt:
        history = history + [{'role': role, 'content': prompt}]

    config = get_config()
    helper_config = config.get('server', {}).get('helper', {})
    enable_local = helper_config.get('enable_local', False)
    connect_timeout = helper_config.get('timeout', {}).get('connect', 20.0)
    read_timeout = helper_config.get('timeout', {}).get('read', 60.0)

    # 根据开关选择配置源
    if enable_local:
        conf = helper_config.get('local', {})
        target_name = "本地API"
    else:
        conf = helper_config.get('cloud', {})
        target_name = "云端API"

    endpoint = conf.get('endpoint', '')
    api_key = conf.get('api_key', '')
    model_name = conf.get('model', '')

    print(f"🔧 [辅助模型] 调用 {target_name}: {model_name}")

    try:
        final_response = call_openai_chat(
            base_url=endpoint,
            api_key=api_key,
            model=model_name,
            messages=history,
            max_tokens=4096,
            temperature=0.7,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )
    except Exception as e:
        return create_response(f"Helper Error: {str(e)}", history, status=500)

    history.append({'role': 'assistant', 'content': final_response})
    log_response(final_response)
    return create_response(final_response, history)


@api.post("/vl")
async def create_vl_chat(request: Request):
    """
    视觉理解接口 (VL)
    支持 Base64 图片上传，自动兼容 OpenAI Vision 格式
    """
    json_post_list = await request.json()
    prompt, history = parse_request(json_post_list)
    log_request(f"{prompt} [图片任务]")

    image_b64 = json_post_list.get('image')

    # 自动修正 Base64 前缀
    if image_b64 and not image_b64.startswith('data:image/'):
        # 默认使用 jpeg，因为体积更小
        image_b64 = f"data:image/png;base64,{image_b64}"

    config = get_config()
    vl_config = config.get('server', {}).get('vl', {})
    enable_local = vl_config.get('enable_local', False)
    connect_timeout = vl_config.get('timeout', {}).get('connect', 20.0)
    read_timeout = vl_config.get('timeout', {}).get('read', 60.0)

    # 选择配置
    if enable_local:
        conf = vl_config.get('local', {})
        target_name = "本地API"
    else:
        conf = vl_config.get('cloud', {})
        target_name = "云端API"

    endpoint = conf.get('endpoint', '')
    api_key = conf.get('api_key', '')
    model_name = conf.get('model', '')

    print(f"👁️ [视觉模型] 调用 {target_name}: {model_name}")

    # --- 构建消息体 ---
    # VL 任务通常单轮，我们构建一个临时的 messages 列表发给 API
    api_messages = []

    # 1. 复制历史（纯文本）
    if history:
        api_messages.extend(history)

    # 2. 构建当前多模态消息
    current_content = [{"type": "text", "text": prompt}]
    if image_b64:
        current_content.append({
            "type": "image_url",
            "image_url": {"url": image_b64}
        })

    api_messages.append({
        "role": "user",
        "content": current_content
    })

    try:
        final_response = call_openai_chat(
            base_url=endpoint,
            api_key=api_key,
            model=model_name,
            messages=api_messages,
            max_tokens=2048,    
            temperature=0.7,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )
    except Exception as e:
        error_msg = f"VL Error: {str(e)}"
        print(f"❌ {error_msg}")
        # 发生错误时，返回无修改的 history
        return create_response(error_msg, history, status=500)

    print("✅ 视觉识别完成")
    log_response(final_response)

    # --- 处理返回的历史 ---
    # 关键：我们只将【纯文本】的 prompt 存入历史返回给客户端
    # 这样主模型的上下文里就不会包含巨大的 Base64 数据
    history.append({'role': 'user', 'content': prompt})
    history.append({'role': 'assistant', 'content': final_response})

    return create_response(final_response, history)


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 MurasamePet API 服务启动中...")
    print("=" * 60)

    config = get_config()

    # 检查主模型配置
    llm_conf = config.get('server', {}).get('llm', {})
    enable_local_llm = llm_conf.get('enable_local', False)

    if enable_local_llm:
        print(f"🏠 [配置] 主模型启用本地权重模式，开始加载模型...")
        model, tokenizer = load_model_and_tokenizer()
    else:
        print(f"🌐 [配置] 主模型启用云端 API 模式，跳过本地权重加载。")
        # 打印一下云端配置信息供检查
        cloud = llm_conf.get('cloud', {})
        print(f"   - 端点: {cloud.get('endpoint')}")
        print(f"   - 模型: {cloud.get('model')}")

    print("=" * 60)
    print("✅ 模型加载完成，启动 FastAPI 服务器...")
    print(f"🌐 服务地址: http://0.0.0.0:28565")
    print(f"📡 可用端点:")
    print(f"   - POST /chat   (主对话接口)")
    print(f"   - POST /llm   (通用问答接口)")
    print(f"   - POST /vl   (视觉理解接口)")
    print("=" * 60)

    uvicorn.run(api, host='0.0.0.0', port=28565, workers=1)