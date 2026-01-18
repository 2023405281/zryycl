/**
 * 智能问答系统前端交互逻辑
 * 功能：处理表单提交、调用后端接口、渲染结果
 */
document.addEventListener('DOMContentLoaded', function() {
    // 获取页面元素
    const form = document.getElementById('text-process-form');
    const inputText = document.getElementById('input-text');
    const taskType = document.getElementById('task-type');
    const resultContent = document.getElementById('result-content');
    const loadingTip = document.getElementById('loading-tip');

    // 初始化：隐藏加载提示和结果区域
    loadingTip.style.display = 'none';
    resultContent.style.display = 'none';

    // 表单提交事件
    form.addEventListener('submit', function(e) {
        // 阻止表单默认提交行为
        e.preventDefault();

        // 1. 获取并校验输入
        const text = inputText.value.trim();
        const task = taskType.value;
        
        if (!text) {
            alert('请输入要处理的文本！');
            return;
        }

        // 2. 显示加载提示，清空旧结果
        loadingTip.style.display = 'block';
        resultContent.style.display = 'none';
        resultContent.innerText = '';

        // 3. 调用后端接口
        fetch('/api/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,        // 与后端接口参数名一致
                task_type: task    // 与后端接口参数名一致
            })
        })
        .then(response => {
            // 检查响应状态是否正常
            if (!response.ok) {
                throw new Error(`HTTP错误，状态码：${response.status}`);
            }
            return response.json(); // 解析JSON响应
        })
        .then(data => {
            // 4. 隐藏加载提示，渲染结果
            loadingTip.style.display = 'none';
            
            if (data.code === 200) {
                // 成功：显示结果
                resultContent.style.display = 'block';
                resultContent.innerText = data.data.content;
            } else {
                // 失败：提示错误信息
                alert(`处理失败：${data.msg}`);
            }
        })
        .catch(error => {
            // 5. 网络/接口异常处理
            loadingTip.style.display = 'none';
            console.error('请求失败详情：', error);
            alert('接口请求失败，请检查后端服务是否正常运行！');
        });
    });
});