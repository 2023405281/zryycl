import requests
import json
import time

def call_doubao_api(input_text: str, token: str, timeout: int = 30) -> dict:
    """
    调用豆包API进行智能问答交互
    :param input_text: 用户输入的提问文本
    :param token: 豆包API的token（你的专属token已适配）
    :param timeout: 请求超时时间（默认30秒）
    :return: 包含响应内容和状态码的字典
    """
    # 1. 配置API基础信息（豆包官方API端点，适配通用对话接口）
    API_URL = "https://ark.cn-beijing.volces.com/api/v3chat/completions"
    HEADERS = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }

    # 2. 构造请求体（适配豆包API的参数格式）
    request_data = {
        "model": "doubao-pro",  # 豆包专业版模型，可根据需求替换为其他模型
        "messages": [
            {
                "role": "user",
                "content": input_text
            }
        ],
        "temperature": 0.7,  # 回答随机性（0-1，值越低越精准）
        "max_tokens": 2000,  # 最大回复长度
        "stream": False      # 非流式输出，直接返回完整结果
    }

    try:
        # 3. 发送POST请求调用API
        response = requests.post(
            url=API_URL,
            headers=HEADERS,
            data=json.dumps(request_data, ensure_ascii=False),
            timeout=timeout
        )

        # 4. 处理响应结果
        response.raise_for_status()  # 抛出HTTP状态码异常（如401/500等）
        result = response.json()

        # 解析核心回复内容
        if "choices" in result and len(result["choices"]) > 0:
            answer = result["choices"][0]["message"]["content"].strip()
            return {
                "code": 200,
                "response": answer,
                "request_id": result.get("id", ""),
                "usage": result.get("usage", {})  # 令牌使用量（可选）
            }
        else:
            return {
                "code": 400,
                "response": "豆包API返回格式异常，未找到回复内容",
                "raw_result": result
            }

    # 5. 异常处理（覆盖常见错误场景）
    except requests.exceptions.Timeout:
        return {
            "code": 408,
            "response": f"请求超时（{timeout}秒），请检查网络或重试"
        }
    except requests.exceptions.ConnectionError:
        return {
            "code": 503,
            "response": "网络连接失败，请检查网络或API地址是否正确"
        }
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP错误：{e.response.status_code} - {e.response.reason}"
        # 针对性处理常见HTTP错误
        if e.response.status_code == 401:
            error_msg = "Token无效或过期，请检查你的豆包token是否正确"
        elif e.response.status_code == 429:
            error_msg = "请求频率过高，请稍后重试"
        return {
            "code": e.response.status_code,
            "response": error_msg
        }
    except Exception as e:
        return {
            "code": 500,
            "response": f"调用豆包API时发生未知错误：{str(e)}"
        }


# 测试代码（单独运行该文件时验证API是否可用）
if __name__ == "__main__":
    # 使用你的专属token测试
    TEST_TOKEN = "dc2bd008-7c45-4744-ba12-ec6754d8c1a1"
    TEST_TEXT = "请介绍一下智能问答系统的核心架构"
    
    print("正在调用豆包API...")
    start_time = time.time()
    result = call_doubao_api(TEST_TEXT, TEST_TOKEN)
    end_time = time.time()

    print(f"\n===== 调用结果 =====")
    print(f"状态码：{result['code']}")
    print(f"耗时：{end_time - start_time:.2f}秒")
    print(f"回复内容：\n{result['response']}")