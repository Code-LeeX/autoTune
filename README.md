# AI Pitch Correction Tool (Melodyne Clone)

一个类似Auto-Tune/Melodyne的AI修音工具，使用最新的深度学习技术提供高质量的音高修正。

## 🎯 核心特性

- **高精度音高检测**: 使用CREPE模型进行基频提取
- **自然音色保持**: 通过DDSP技术保留原始音色特征
- **直观的可视化界面**: 实时波形和音高曲线显示
- **灵活的参数控制**: 可调节修音强度、平滑度等参数
- **手动编辑功能**: 支持精细的音符级别调整
- **多格式支持**: 支持WAV/MP3/FLAC输入输出

## 🏗️ 技术架构

### 前端 (React)
- React 18 + TypeScript
- Web Audio API 音频处理
- Canvas 波形可视化
- Material-UI 组件库

### 后端 (Python FastAPI)
- **CREPE**: 神经网络音高检测
- **DDSP**: 可微分数字信号处理
- **PyWorld**: 备选音频分析工具
- FastAPI + Uvicorn 服务器

### 核心算法
1. **音高提取**: CREPE CNN模型 → f0曲线 + confidence
2. **智能量化**: 保持颤音特征的12平均律对齐
3. **音频合成**: DDSP重建 → 保持音色的修音输出

## 🚀 快速开始

### 环境要求
- Python 3.8+
- Node.js 16+
- 8GB+ RAM (推荐16GB)
- Apple Silicon / CUDA GPU (可选)

### 安装步骤

1. **后端设置**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **前端设置**
```bash
cd frontend
npm install
```

3. **启动服务**
```bash
# 后端 (端口 8000)
cd backend && python main.py

# 前端 (端口 3000)
cd frontend && npm start
```

## 📁 项目结构

```
├── backend/                 # Python FastAPI后端
│   ├── models/              # AI模型管理
│   ├── audio_processing/    # 音频处理核心
│   ├── api/                 # REST API路由
│   ├── utils/               # 工具函数
│   └── config/              # 配置管理
├── frontend/                # React前端
│   ├── src/components/      # React组件
│   ├── src/utils/           # 音频工具
│   └── src/api/             # API调用
├── docs/                    # 项目文档
└── scripts/                 # 构建脚本
```

## 🎵 使用方法

1. **上传音频**: 支持拖拽或选择音频文件
2. **参数调节**: 设置修音强度、目标音阶等
3. **实时预览**: 查看音高曲线和修正效果
4. **手动编辑**: 精细调整特定音符
5. **导出结果**: 多种格式高质量输出

## 🔧 开发指南

### API接口
- `POST /upload` - 音频文件上传
- `POST /process` - 音频处理
- `GET /download/{id}` - 结果下载
- `WebSocket /ws` - 实时处理进度

### 核心算法
详见 `docs/algorithm.md` 中的技术细节说明

## 📄 许可证

MIT License - 详见 LICENSE 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！