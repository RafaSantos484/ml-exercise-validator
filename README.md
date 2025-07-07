# 🏋️ ML Exercise Validator

Solução web que utiliza visão computacional e aprendizado de máquina para validar automaticamente a execução de exercícios físicos, fornecendo feedback em tempo real diretamente no navegador do usuário.

## 📸 Visão Geral

Este projeto permite que usuários realizem exercícios físicos, como a prancha alta, enquanto recebem validações automáticas sobre a correção da postura. A validação é feita com base na detecção de pontos do corpo utilizando o BlazePose (MediaPipe) e modelos de inferência treinados previamente, como KNN, SVM, Random Forest, regressão logística, além de um modelo empírico baseado em regras como intervalos em que os ângulos se encontram. Os modelos de ML treinados em Python podem ser carregados no ambiente JavaScript dos navegadores utilizando o formato [ONNX](https://onnx.ai/).

## 🚀 Tecnologias Utilizadas

- React
- TypeScript
- MediaPipe
- onnxruntime-web
- WebAssembly (WASM)
- HTML5 Video API

## 🧩 Software design

![Class diagram](https://github.com/user-attachments/assets/200a6963-d85f-4e9a-b078-771bac0ca454)

Os principais design patterns usados foram:
- Factory Method
- Singleton
- Strategy
- Facade
